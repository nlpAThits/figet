
from pyflann import *
from sklearn.neighbors import NearestNeighbors
import numpy as np
from figet.utils import get_logging
from figet.hyperbolic import poincare_distance, hyperbolic_distance_numpy
import torch
from functools import cmp_to_key
from figet.evaluate import COARSE

log = get_logging()


def poincare_distance_wrapper(a, b):
    return poincare_distance(a[1], b[1])


class kNN(object):
    """
    In this class I should do all the calculations related to the precision.
    This is:
        - Analyze with which top-k I cover most of the cases.
        - Analyze in which position is the right candidate (on average)
    """

    def __init__(self, type2vec, knn_hyper=False):
        self.type2vec = type2vec
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.flann = FLANN()
        self.params = self.flann.build_index(type2vec.cpu().numpy(), algorithm='autotuned', target_precision=0.99,
                                             build_weight=0.01, memory_weight=0, sample_fraction=0.25)
        self.knn_hyper = knn_hyper

    def _query_index(self, predictions, k):
        predictions = predictions.detach()
        if not self.knn_hyper:
            if k > len(self.type2vec):
                k = len(self.type2vec)
            indexes, _ = self.flann.nn_index(predictions.detach().cpu().numpy(), k, checks=self.params["checks"])
            return torch.from_numpy(indexes).to(self.device).long()

        factor = 10
        neighbors = factor * k if factor * k <= len(self.type2vec) else len(self.type2vec)
        indexes, _ = self.flann.nn_index(predictions.detach().cpu().numpy(), neighbors, checks=self.params["checks"])
        result = []
        for idx in indexes:
            idx_and_tensors = list(zip(idx, [tensor for tensor in self.type2vec[idx]]))
            sorted_idx_and_tensors = sorted(idx_and_tensors, key=cmp_to_key(poincare_distance_wrapper))
            result.append([sorted_idx_and_tensors[i][0] for i in range(neighbors)])
        return torch.LongTensor(result).to(self.device)

    def neighbors(self, predictions, type_indexes, k):
        try:
            indexes = self._query_index(predictions, k)
        except ValueError:
            log.debug("EXPLOTO TODO!")
            log.debug(predictions)

        return indexes  # , self._one_hot_true_types(indexes, type_indexes)

    def _one_hot_true_types(self, neighbor_indexes, type_indexes):
        """
        :param neighbor_indexes: batch x k
        :param type_indexes: batch x type_len
        :return: batch x k with a one hot vector describing where the right type is
        """
        one_hot = torch.zeros(neighbor_indexes.shape).to(self.device)
        for i in range(len(neighbor_indexes)):
            neighbors = neighbor_indexes[i]
            types = type_indexes[i]
            for t in types:
                j = np.where(t.item() == neighbors)[0]
                if len(j):
                    one_hot[i][j] = 1.0
        return one_hot

    def precision_at(self, predictions, types, k):
        if k > len(self.type2vec):
            k = len(self.type2vec)
            log.info("WARNING: k should be less or equal than len(type2vec). Otherwise is asking precision at the "
                     "full dataset")

        indexes = self._query_index(predictions, k)

        total_precision = 0
        for i in range(len(predictions)):
            true_types = set(j.item() for j in types[i])
            neighbors = set(x for x in indexes[i])
            total_precision += 1 if true_types.intersection(neighbors) else 0
        return total_precision

    def type_positions(self, predictions, types):
        indexes = self._query_index(predictions, len(self.type2vec))
        types_positions = []
        closest_true_neighbor = []
        for i in range(len(types)):
            true_types = types[i].tolist()
            neighbors = indexes[i]
            positions = [np.where(neighbors == true_type)[0].item() for true_type in true_types]
            types_positions.extend(positions)
            closest_true_neighbor.append(min(positions))
        return types_positions, closest_true_neighbor


class TypeAssigner(object):
    def __init__(self, type_dict, type2vec):
        self.type_dict = type_dict
        self.coarse = list(COARSE)
        self.coarse.remove("entity")
        coarse_tensors = torch.stack([type2vec[type_dict.label2idx[label]] for label in self.coarse])

        self.coarse_knn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric=hyperbolic_distance_numpy)
        self.coarse_knn.fit(coarse_tensors)

    def get_closest_coarse(self, pred_embeds):
        indexes = self.coarse_knn.kneighbors(pred_embeds.detach(), n_neighbors=1, return_distance=False)
        return [self.type_dict.label2idx[self.coarse[i.item()]] for i in indexes]

    def assign_types(self, pred_embeds, neighbor_indexes, type_indexes, hierarchy=None, threshold=0.5):
        """
        :param predictions: batch x k
        :param neighbor_indexes: batch x k
        :param type_indexes: batch x type_len
        :return: list of pairs of predicted type indexes, and true type indexes
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        closest_coarse = self.get_closest_coarse(pred_embeds)
        result = []
        for i in range(len(pred_embeds)):
            # predicted_indexes = (predictions[i] >= threshold).nonzero()
            # if len(predicted_indexes) == 0:
            #     predicted_indexes = predictions[i].max(0)[1].unsqueeze(0)
            # predicted_types = neighbor_indexes[i][predicted_indexes]

            predicted_types = neighbor_indexes[i]
            parents = [closest_coarse[i]]

            # if hierarchy:
            #     for predicted_type in predicted_types:
            #         parents += hierarchy.get_parents_id(predicted_type.item())

            types_set = set(parents).union(set([i.item() for i in predicted_types]))

            result.append([type_indexes[i], torch.LongTensor(list(types_set)).to(device)])

        return result

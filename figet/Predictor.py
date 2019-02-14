
from pyflann import *
import numpy as np
from figet.utils import get_logging
from figet.Constants import COARSE_FLAG, FINE_FLAG, UF_FLAG
from figet.hyperbolic import poincare_distance
import torch
from functools import cmp_to_key

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

    def __init__(self, type2vec, type_vocab, knn_hyper=False):
        self.type2vec = type2vec
        self.type_vocab = type_vocab
        self.knn_hyper = knn_hyper
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.neighs_per_granularity = {COARSE_FLAG: 1, FINE_FLAG: 2, UF_FLAG: 3}

        self.granularity_ids = {COARSE_FLAG: list(type_vocab.get_coarse_ids()),
                                FINE_FLAG: list(type_vocab.get_fine_ids()),
                                UF_FLAG: list(type_vocab.get_ultrafine_ids())}
        self.granularity_ids[COARSE_FLAG].remove(type_vocab.label2idx["entity"])

        self.granularity_sets = {gran_flag: set(ids) for gran_flag, ids in self.granularity_ids.items()}
        self.knn_searchers = {}
        self.checks = {}

        for granularity in [COARSE_FLAG, FINE_FLAG, UF_FLAG]:
            ids = self.granularity_ids[granularity]
            type_vectors = type2vec[ids]
            gran_flann = FLANN()
            params = gran_flann.build_index(type_vectors.cpu().numpy(), algorithm='autotuned', target_precision=0.99,
                                            build_weight=0.01, memory_weight=0, sample_fraction=0.25)
            self.knn_searchers[granularity] = gran_flann
            self.checks[granularity] = params["checks"]

    def _query_index(self, predictions, gran_flag, k=-1):
        """
        :param predictions:
        :param gran_flag:
        :param k: amount of neighbors to retrieve
        :return:
        """
        max_neighbors = len(self.granularity_ids[gran_flag])
        if k == -1:
            k = self.neighs_per_granularity[gran_flag]
        if k > max_neighbors:
            k = max_neighbors

        knn_searcher = self.knn_searchers[gran_flag]
        checks = self.checks[gran_flag]
        predictions = predictions.detach().cpu().numpy()

        if not self.knn_hyper:
            indexes, _ = knn_searcher.nn_index(predictions, k, checks=checks)
            mapped_indexes = self.map_indices_to_type2vec(indexes, gran_flag)
            return torch.LongTensor(mapped_indexes).to(self.device)

        factor = 10
        requested_neighbors = factor * k if factor * k <= max_neighbors else max_neighbors
        indexes, _ = knn_searcher.nn_index(predictions, requested_neighbors, checks=checks)
        mapped_indexes = self.map_indices_to_type2vec(indexes, gran_flag)
        result = []
        for idx in mapped_indexes:
            idx_and_tensors = list(zip(idx, [tensor for tensor in self.type2vec[idx]]))
            sorted_idx_and_tensors = sorted(idx_and_tensors, key=cmp_to_key(poincare_distance_wrapper))
            result.append([sorted_idx_and_tensors[i][0] for i in range(k)])

        return torch.LongTensor(result).to(self.device)

    def map_indices_to_type2vec(self, indexes, gran_flag):
        """
        :param indexes: batch x n
        :param gran_flag:
        :return:
        """
        original_ids = self.granularity_ids[gran_flag]
        result = []
        for index in indexes:
            if type(index) == np.int32: index = [index]
            row = [original_ids[value] for value in index]
            result.append(row)
        return result

    def neighbors(self, predictions, k, gran_flag):
        try:
            indexes = self._query_index(predictions, gran_flag, k)
        except ValueError:
            log.debug("EXPLOTO TODO!")
            log.debug(predictions)

        return indexes      # , self._one_hot_true_types(indexes, type_indexes)

    def type_positions(self, predictions, types, granularity_flag):
        indexes = self._query_index(predictions, granularity_flag, k=len(self.type2vec))
        gran_ids_set = self.granularity_sets[granularity_flag]
        types_positions = []
        closest_true_neighbor = []
        for i in range(len(types)):
            true_types = [x for x in types[i].tolist() if x in gran_ids_set]
            if not true_types:
                continue
            neighbors = indexes[i]
            positions = [np.where(neighbors == true_type)[0].item() for true_type in true_types]
            types_positions.extend(positions)
            closest_true_neighbor.append(min(positions))

        return types_positions, closest_true_neighbor


    # def _one_hot_true_types(self, neighbor_indexes, type_indexes):
    #     """
    #     :param neighbor_indexes: batch x k
    #     :param type_indexes: batch x type_len
    #     :return: batch x k with a one hot vector describing where the right type is
    #     """
    #     one_hot = torch.zeros(neighbor_indexes.shape).to(self.device)
    #     for i in range(len(neighbor_indexes)):
    #         neighbors = neighbor_indexes[i]
    #         types = type_indexes[i]
    #         for t in types:
    #             j = np.where(t.item() == neighbors)[0]
    #             if len(j):
    #                 one_hot[i][j] = 1.0
    #     return one_hot

    # def precision_at(self, predictions, types, k):
    #     if k > len(self.type2vec):
    #         k = len(self.type2vec)
    #         log.info("WARNING: k should be less or equal than len(type2vec). Otherwise is asking precision at the "
    #                  "full dataset")
    #
    #     indexes = self._query_index(predictions, k)
    #
    #     total_precision = 0
    #     for i in range(len(predictions)):
    #         true_types = set(j.item() for j in types[i])
    #         neighbors = set(x for x in indexes[i])
    #         total_precision += 1 if true_types.intersection(neighbors) else 0
    #     return total_precision


def assign_types(predictions, neighbor_indexes, type_indexes, predictor, threshold=0.5, gran_flag=COARSE_FLAG):
    """
    :param predictions: batch x k
    :param neighbor_indexes: batch x k
    :param type_indexes: batch x type_len
    :return: list of pairs of predicted type indexes, and true type indexes
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    result = []
    parents = predictor.neighbors(predictions, 1, gran_flag=COARSE_FLAG) if gran_flag != COARSE_FLAG else None

    for i in range(len(neighbor_indexes)):

        predicted_types = neighbor_indexes[i]

        types_set = set([j.item() for j in predicted_types])
        if gran_flag != COARSE_FLAG:
            item_parents = parents[i]
            types_set = types_set.union(set(item_parents.tolist()))

        result.append([type_indexes[i], torch.LongTensor(list(types_set)).to(device)])

    return result


def assign_all_granularities_types(neighbor_indexes, type_indexes, hierarchy):
    """
    :param neighbor_indexes: list of neighbors for all granularities
    :param type_indexes:
    :return:
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    result = []
    for i in range(len(neighbor_indexes[0])):
        types_set = set()
        for neigh_idx in neighbor_indexes:
            types_set = types_set.union(set([j.item() for j in neigh_idx[i]]))

        # parents = []
        # if hierarchy:
        #     for predicted_type in types_set:
        #         parents += hierarchy.get_parents_id(predicted_type)
        #
        # types_set = types_set.union(set(parents))

        result.append((type_indexes[i], torch.LongTensor(list(types_set)).to(device)))

    return result

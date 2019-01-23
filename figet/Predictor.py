
from annoy import AnnoyIndex
import numpy as np
from figet.utils import get_logging
from figet.hyperbolic import hyperbolic_distance_list, poincare_distance
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

    def __init__(self, type2vec):
        self.type2vec = type2vec
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.annoy_index = AnnoyIndex(type2vec[0].size(0))
        for i, v in enumerate(type2vec):
            self.annoy_index.add_item(i, v)

        trees = len(type2vec) / 50 if len(type2vec) > 500 else 10
        self.annoy_index.build(trees)

    def query_index(self, predictions, k):
        predictions = predictions.detach()
        neighbors = 2 * k if 2 * k <= len(self.type2vec) else len(self.type2vec)
        result = []
        for pred in predictions:
            indexes = self.annoy_index.get_nns_by_vector(pred, n=neighbors)
            idx_and_tensors = list(zip(indexes, [tensor for tensor in self.type2vec[indexes]]))
            sorted_idx_and_tensors = sorted(idx_and_tensors, key=cmp_to_key(poincare_distance_wrapper))
            result.append([sorted_idx_and_tensors[i][0] for i in range(k)])

        return torch.LongTensor(result).to(self.device)

    def neighbors(self, predictions, type_indexes, k):
        try:
            indexes = self.query_index(predictions, k)
        except ValueError:
            log.debug("EXPLOTO TODO!")
            log.debug(predictions)

        return indexes, self._one_hot_true_types(indexes, type_indexes)

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

        indexes = self.query_index(predictions, k)

        total_precision = 0
        for i in range(len(predictions)):
            true_types = set(j.item() for j in types[i])
            neighbors = set(x for x in indexes[i])
            total_precision += 1 if true_types.intersection(neighbors) else 0
        return total_precision

    def type_positions(self, predictions, types):
        indexes = self.query_index(predictions, len(self.type2vec))
        types_positions = []
        closest_true_neighbor = []
        for i in range(len(types)):
            true_types = types[i].tolist()
            neighbors = indexes[i]
            positions = [np.where(neighbors == true_type)[0].item() for true_type in true_types]
            types_positions.extend(positions)
            closest_true_neighbor.append(min(positions))
        return types_positions, closest_true_neighbor


def assign_types(predictions, neighbor_indexes, type_indexes, hierarchy=None, threshold=0.5):
    """
    :param predictions: batch x k
    :param neighbor_indexes: batch x k
    :param type_indexes: batch x type_len
    :return: list of pairs of predicted type indexes, and true type indexes
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    result = []
    for i in range(len(predictions)):
        predicted_indexes = (predictions[i] >= threshold).nonzero()
        if len(predicted_indexes) == 0:
            predicted_indexes = predictions[i].max(0)[1].unsqueeze(0)

        predicted_types = neighbor_indexes[i][predicted_indexes]

        parents = []
        if hierarchy:
            for predicted_type in predicted_types:
                parents += hierarchy.get_parents_id(predicted_type.item())

        types_set = set(parents).union(set([i.item() for i in predicted_types]))

        result.append([type_indexes[i], torch.LongTensor(list(types_set)).to(device)])

    return result

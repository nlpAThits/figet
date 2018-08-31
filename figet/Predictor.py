
from sklearn.neighbors import NearestNeighbors
import numpy as np
from figet.utils import get_logging
import torch

log = get_logging()


class kNN(object):
    """
    In this class I should do all the calculations related to the precision.
    This is:
        - Analyze with which top-k I cover most of the cases.
        - Analyze in which position is the right candidate (on average)
    """

    def __init__(self, type_dict, type2vec, metric=None):
        self.type_dict = type_dict      # If I don't use them, then why to store them?
        self.type2vec = type2vec
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if metric:
            self.neigh = NearestNeighbors(n_neighbors=5, algorithm='ball_tree', metric=metric)
        else:
            self.neigh = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
        self.neigh.fit(type2vec)

    def neighbors(self, predictions, type_indexes, k):
        indexes = self.neigh.kneighbors(predictions.detach(), n_neighbors=k, return_distance=False)
        indexes = torch.from_numpy(indexes).to(self.device).long()

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

        indexes = self.neigh.kneighbors(predictions.detach(), n_neighbors=k, return_distance=False)

        if types.size(1) != 1: types = types[:,-1]
        total_precision = 0
        for i in range(len(predictions)):
            true_types = set(i.item() for i in [types[i]])
            neighbors = set(x for x in indexes[i])
            total_precision += 1 if true_types.intersection(neighbors) else 0
        return total_precision

    def true_types_position(self, predictions, types):
        indexes = self.neigh.kneighbors(predictions.detach(), n_neighbors=len(self.type2vec), return_distance=False)
        if types.size(1) != 1: types = types[:, -1]
        types_positions = []
        for i in range(len(types)):
            true_type = types[i].item()
            neighbors = indexes[i]
            position = np.where(neighbors == true_type)
            types_positions.append(position[0].item())
        return types_positions


def assign_types(predictions, type_indexes, threshold=0.5):
    """
    :param predictions:
    :param truth:
    :return: list of pairs of predicted type indexes, and true type indexes
    """
    result = []
    for i in range(len(predictions)):
        predicted_indexes = (predictions[i] >= threshold).nonzero()

        if len(predicted_indexes) == 0:
            predicted_indexes = predictions[i].max(0)[1].unsqueeze(0)

        result.append([type_indexes[i], predicted_indexes.long()])

    return result

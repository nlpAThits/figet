
from sklearn.neighbors import NearestNeighbors
import numpy as np
from figet.utils import get_logging, hyperbolic_distance

log = get_logging()


class Predictor(object):
    """
    In this class I should do all the calculations related to the precision.
    This is:
        - Analyze with which top-k I cover most of the cases.
        - Analyze in which position is the right candidate (on average)
    """

    def __init__(self, type_dict, type2vec):
        self.type_dict = type_dict      # Si no los uso para nada, no hace falta que los guarde
        self.type2vec = type2vec
        self.neigh = NearestNeighbors(n_neighbors=5, algorithm='ball_tree', metric=hyperbolic_distance)
        self.neigh.fit(type2vec)

    def precision_at(self, predictions, types, k):
        if k > len(self.type2vec):
            k = len(self.type2vec)
            log.info("WARNING: k should be less or equal than len(type2vec). Otherwise is asking precision at the "
                     "full dataset")

        indexes = self.neigh.kneighbors(predictions, n_neighbors=k, return_distance=False)
        total_precision = 0
        for i in range(len(predictions)):
            true_types = set(i.item() for i in [types[i]])
            neighbors = set(x for x in indexes[i])
            total_precision += 1 if true_types.intersection(neighbors) else 0
        return total_precision

    def true_types_position(self, predictions, types):
        indexes = self.neigh.kneighbors(predictions, n_neighbors=len(self.type2vec), return_distance=False)
        types_positions = []
        for i in range(len(types)):
            true_type = types[i].item()
            neighbors = indexes[i]
            position = np.where(neighbors == true_type)
            types_positions.append(position[0].item())
        return types_positions

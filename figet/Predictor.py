
from figet.utils import get_logging
from sklearn.neighbors import KDTree

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
        self.tree = KDTree(type2vec)

    def precision_at(self, predictions, types, k):
        if k > len(self.type2vec):
            k = len(self.type2vec)
            log.info("WARNING: k should be less or equal than len(type2vec). Otherwise is asking precision at the "
                     "full dataset")

        distances, indexes = self.tree.query(predictions, k=k)
        total_precision = 0
        for i in range(len(predictions)):
            true_types = set(i.item() for i in [types[i]])
            neighbors = set(x for x in indexes[i])
            total_precision += 1 if true_types.intersection(neighbors) else 0
        return total_precision

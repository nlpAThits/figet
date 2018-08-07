
from sklearn.neighbors import KDTree


class Predictor(object):
    """
    In this class I should do all the calculations related to the precision.
    This is:
        - Analyze with which top-k I cover most of the cases.
        - Analyze in which position is the right candidate (on average)
    """

    def __init__(self, type_dict, type2vec):
        self.type_dict = type_dict
        self.type2vec = type2vec
        self.tree = KDTree(type2vec)

    def precision_at(self, predictions, types, k):
        distances, indexes = self.tree.query(predictions.numpy(), k=k)
        return sum([types[i] in indexes[i] for i in range(len(types))]) # algo asi...
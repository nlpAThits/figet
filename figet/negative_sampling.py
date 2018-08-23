
from operator import itemgetter
from figet.Loss import poincare_distance


class NegativeSampleContainer(object):

    def __init__(self, type2vec):
        self.index_and_distance = {}

        for idx, point in enumerate(type2vec):
            repeated = point.expand(len(type2vec), point.size()[0])
            distances = poincare_distance(repeated, type2vec)
            ordered_idx_and_dist = sorted(enumerate(distances), key=itemgetter(1), reverse=True)
            self.index_and_distance[idx] = ordered_idx_and_dist

    def get_indexes(self, idx, n, current_epoch=None, total_epochs=None):
        """
        :param n: amount of negative indexes
        :param current_epoch: if is not None, returns the corresponding batch, to make training harder. Pre: epoch >= 1
        :return: list of negative indexes
        """
        neg_batch = self._get_negative_batch(idx, current_epoch, total_epochs)

        return [item[0] for item in neg_batch[:n]]

    def get_distances(self, idx, n, current_epoch=None, total_epochs=None):
        """
        :param n: amount of negative indexes
        :param current_epoch: if is not None, returns the corresponding batch, to make training harder. Pre: epoch >= 1
        :return: list of distances to negative samples
        """
        neg_batch = self._get_negative_batch(idx, current_epoch, total_epochs)

        return [item[1] for item in neg_batch[:n]]

    def _get_negative_batch(self, idx, current_epoch=None, total_epochs=None):
        neg_batch = self.index_and_distance[idx]
        if current_epoch:
            neg_batch_size = int(len(neg_batch) / total_epochs)
            neg_batch = neg_batch[(current_epoch - 1) * neg_batch_size:]
        return neg_batch

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

        self.index_cache = {}
        self.distance_cache = {}
        self.current_epoch_cache = -1

    def get_indexes(self, idx, n, current_epoch=None, total_epochs=None):
        return self.update_caches(idx, n, current_epoch, total_epochs)[0]

    def get_distances(self, idx, n, current_epoch=None, total_epochs=None):
        return self.update_caches(idx, n, current_epoch, total_epochs)[1]

    def update_caches(self, idx, n, current_epoch=None, total_epochs=None):
        """
        :param n: amount of negative indexes
        :param current_epoch: if is not None, returns the corresponding batch, to make training harder. Pre: epoch >= 1
        :return: list of negative indexes, list of distances to negative samples
        """
        if current_epoch != self.current_epoch_cache:
            self.clear_cache()
            self.current_epoch_cache = current_epoch

        if idx in self.index_cache:
            return self.index_cache[idx], self.distance_cache[idx]

        neg_batch = self._get_negative_batch(idx, current_epoch, total_epochs)

        sub_neg_batch = neg_batch[:n]
        indexes = [item[0] for item in sub_neg_batch]
        distances = [item[1] for item in sub_neg_batch]

        self.index_cache[idx] = indexes
        self.distance_cache[idx] = distances
        return indexes, distances

    def _get_negative_batch(self, idx, current_epoch=None, total_epochs=None):
        neg_batch = self.index_and_distance[idx]
        if current_epoch:
            neg_batch_size = int(len(neg_batch) / total_epochs)
            neg_batch = neg_batch[(current_epoch - 1) * neg_batch_size:]
        return neg_batch

    def clear_cache(self):
        self.index_cache = {}
        self.distance_cache = {}
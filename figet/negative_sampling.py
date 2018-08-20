
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

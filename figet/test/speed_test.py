
import time
import random
import numpy as np
import torch
from torch.nn import HingeEmbeddingLoss, PairwiseDistance
from sklearn.neighbors import NearestNeighbors

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Loss import hyperbolic_distance_torch, hyperbolic_distance_numpy, poincare_distance


def timeit(func, *params):
    start_time = time.time()
    res = func(*params)
    print(time.time() - start_time)
    return res


def hinge_time_test():
    batch_size = 1000
    emb_dim = 10
    p = torch.Tensor([0.050] * emb_dim)
    q = torch.Tensor([0.049] * emb_dim)

    pred_batch = torch.Tensor(batch_size, emb_dim)
    true_batch = torch.Tensor(batch_size, emb_dim)
    y = torch.Tensor(batch_size).fill_(1)

    for i in range(batch_size):
        pred_batch[i] = p
        true_batch[i] = q

    pairwise_func = PairwiseDistance(p=2, eps=np.finfo(float).eps)
    hinge_loss = HingeEmbeddingLoss()

    print("Pairwise distance calculation")
    pair_distances = timeit(pairwise_func, pred_batch, true_batch)
    print("Hinge loss of pairwise distance")
    timeit(hinge_loss, pair_distances, y)

    print("Hyperbolic distance calculation")
    hyper_distances = timeit(poincare_distance, pred_batch, true_batch)
    print("Hinge loss of hyperbolic distance")
    timeit(hinge_loss, hyper_distances, y)


def knn_time_test():
    batch_size = 1000
    emb_dim = 10

    pred_batch = torch.Tensor(batch_size, emb_dim)
    true_batch = torch.Tensor(batch_size, emb_dim)
    y = torch.Tensor(batch_size).fill_(1)

    for i in range(batch_size):
        pred_batch[i] = torch.Tensor([random.random() for i in range(emb_dim)])
        true_batch[i] = torch.Tensor([random.random() for i in range(emb_dim)])

    knn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree', metric=hyperbolic_distance_numpy)
    knn.fit(true_batch)

    print("Numpy distance calculation")
    timeit(lambda x: knn.kneighbors(pred_batch, n_neighbors=10, return_distance=False), None)

    knn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree', metric=hyperbolic_distance_torch)
    knn.fit(true_batch)

    print("Torch distance calculation")
    timeit(lambda x: knn.kneighbors(pred_batch, n_neighbors=10, return_distance=False), None)


if __name__ == '__main__':
    knn_time_test()

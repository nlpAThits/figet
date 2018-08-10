
import time
import numpy as np
import torch
from torch.nn import HingeEmbeddingLoss, PairwiseDistance

import sys
sys.path.append("..")

from Loss import hyperbolic_distance_torch, hyperbolic_distance_batch


def timeit(func, *params):
    start_time = time.time()
    res = func(*params)
    print(time.time() - start_time)
    return res


def main():
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
    hyper_distances = timeit(hyperbolic_distance_batch, pred_batch, true_batch)
    print("Hinge loss of hyperbolic distance")
    timeit(hinge_loss, hyper_distances, y)


if __name__ == '__main__':
    main()

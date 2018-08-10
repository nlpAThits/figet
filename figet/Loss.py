import torch
import numpy as np
from numpy.linalg import norm
import math


def hyperbolic_distance_numpy(p, q):
    return hyperbolic_distance(norm(p), norm(q), norm(p - q))


def hyperbolic_distance_torch(p, q):
    return hyperbolic_distance(p.norm(), q.norm(), (p - q).norm())


def hyperbolic_distance(p_norm, q_norm, p_minus_q_norm):
    numerator = 2 * p_minus_q_norm * p_minus_q_norm
    denominator = (1 - p_norm * p_norm) * (1 - q_norm * q_norm)
    if denominator <= 0:
        denominator = np.finfo(float).eps
    return math.acosh(1 + numerator / denominator)


def hyperbolic_distance_batch(batch_p, batch_q, cuda=False):
    return batch_metric(batch_p, batch_q, metric=hyperbolic_distance_torch, cuda=cuda)


def batch_metric(batch_p, batch_q, metric, cuda=False):
    result = torch.FloatTensor(len(batch_p)).cuda() if cuda else torch.FloatTensor(len(batch_p))
    for i in range(len(batch_p)):
        result[i] = metric(batch_p[i], batch_q[i])
    return torch.autograd.Variable(result, requires_grad=True)

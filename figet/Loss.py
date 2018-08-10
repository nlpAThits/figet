import torch
import numpy as np
from numpy.linalg import norm
import math

eps = 1e-5


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
    result = torch.tensor(len(batch_p), dtype=torch.float, requires_grad=True)
    result = result.cuda() if cuda else result
    for i in range(len(batch_p)):
        result[i] = metric(batch_p[i], batch_q[i])
    return torch.autograd.Variable(result)


def poincare_distance(u, v):
    boundary = 1 - 1e-5
    squnorm = torch.clamp(torch.sum(u * u, dim=-1), 0, boundary)
    sqvnorm = torch.clamp(torch.sum(v * v, dim=-1), 0, boundary)
    sqdist = torch.sum(torch.pow(u - v, 2), dim=-1)
    x = sqdist / ((1 - squnorm) * (1 - sqvnorm)) * 2 + 1
    # arcosh
    z = torch.sqrt(torch.pow(x, 2) - 1)
    return torch.log(x + z)


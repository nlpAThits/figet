from figet import utils
import torch
from torch.autograd import Function
import numpy as np
from numpy.linalg import norm
import math

eps = 1e-5

log = utils.get_logging()


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
    """
    From: https://github.com/facebookresearch/poincare-embeddings/blob/master/model.py#L48
    """
    boundary = 1 - 1e-5
    squnorm = torch.clamp(torch.sum(u * u, dim=-1), 0, boundary)
    sqvnorm = torch.clamp(torch.sum(v * v, dim=-1), 0, boundary)
    sqdist = torch.sum(torch.pow(u - v, 2), dim=-1)
    x = sqdist / ((1 - squnorm) * (1 - sqvnorm)) * 2 + 1
    # arcosh
    z = torch.sqrt(torch.pow(x, 2) - 1)
    return torch.log(x + z)


class PoincareDistance(Function):
    boundary = 1 - eps

    @staticmethod
    def grad(x, v, sqnormx, sqnormv, sqdist):
        alpha = (1 - sqnormx)
        beta = (1 - sqnormv)
        z = 1 + 2 * sqdist / (alpha * beta)
        a = ((sqnormv - 2 * torch.sum(x * v, dim=-1) + 1) / torch.pow(alpha, 2)).unsqueeze(-1).expand_as(x)
        a = a * x - v / alpha.unsqueeze(-1).expand_as(v)
        z = torch.sqrt(torch.pow(z, 2) - 1)
        z = torch.clamp(z * beta, min=eps).unsqueeze(-1)
        return 4 * a / z.expand_as(x)

    @staticmethod
    def forward(ctx, u, v):
        squnorm = torch.clamp(torch.sum(u * u, dim=-1), 0, PoincareDistance.boundary)
        sqvnorm = torch.clamp(torch.sum(v * v, dim=-1), 0, PoincareDistance.boundary)
        sqdist = torch.sum(torch.pow(u - v, 2), dim=-1)
        ctx.save_for_backward(u, v, squnorm, sqvnorm, sqdist)
        x = sqdist / ((1 - squnorm) * (1 - sqvnorm)) * 2 + 1
        # arcosh
        z = torch.sqrt(torch.pow(x, 2) - 1)
        return torch.log(x + z)

    @staticmethod
    def backward(ctx, g):
        u, v, squnorm, sqvnorm, sqdist = ctx.saved_tensors
        g = g.unsqueeze(-1)
        gu = PoincareDistance.grad(u, v, squnorm, sqvnorm, sqdist)
        gv = PoincareDistance.grad(v, u, sqvnorm, squnorm, sqdist)
        u_grad, v_grad = g.expand_as(gu) * gu, g.expand_as(gv) * gv
        # log.debug("Gradient: u: {}, v: {}".format(u_grad, v_grad))
        return -u_grad, -v_grad
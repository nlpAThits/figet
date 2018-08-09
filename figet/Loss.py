import torch
import numpy as np
from numpy.linalg import norm
import math

import unittest


def hyperbolic_distance_numpy(p, q):
    numerator = 2 * norm(p - q)**2
    denominator = (1 - norm(p)**2) * (1 - norm(q)**2)
    if denominator <= 0:
        denominator = np.finfo(float).eps
    return math.acosh(1 + numerator / denominator)


def hyperbolic_distance_torch(p, q):
    numerator = 2 * ((p - q).norm())**2
    denominator = (1 - (p.norm())**2) * (1 - (q.norm())**2)
    if denominator <= 0:
        denominator = np.finfo(float).eps
    return math.acosh(1 + numerator / denominator)


def hyperbolic_distance_batch(batch_p, batch_q):
    return batch_metric(batch_p, batch_q, metric=hyperbolic_distance_torch)


def batch_metric(batch_p, batch_q, metric):
    result = torch.FloatTensor(len(batch_p))
    for i in range(len(batch_p)):
        result[i] = metric(batch_p[i], batch_q[i])
    return torch.autograd.Variable(result, requires_grad=True)


class TestHyperbolicDistance(unittest.TestCase):
    def test_torch_distance(self):
        p = torch.Tensor([0, 0.5])
        q = torch.Tensor([0.5, 0.5])
        self.assertAlmostEqual(hyperbolic_distance_torch(p, q), 1.4909, places=3)

        z = torch.Tensor([-0.822, 0.475])
        w = torch.Tensor([-0.95, 0.])
        self.assertAlmostEqual(hyperbolic_distance_torch(z, w), 4.630, places=2)

    def test_numpy_distance(self):
        p = np.array([0, 0.5])
        q = np.array([0.5, 0.5])
        self.assertAlmostEqual(hyperbolic_distance_numpy(p, q), 1.4909, places=3)

        z = np.array([-0.822, 0.475])
        w = np.array([-0.95, 0.])
        self.assertAlmostEqual(hyperbolic_distance_numpy(z, w), 4.630, places=2)

    def test_batch_distance(self):
        p = torch.Tensor([0, 0.5])
        q = torch.Tensor([0.5, 0.5])
        batch_size = 10
        batch_p = torch.Tensor(batch_size, 2)
        batch_q = torch.Tensor(batch_size, 2)
        for i in range(batch_size):
            batch_p[i] = p
            batch_q[i] = q

        result = hyperbolic_distance_batch(batch_p, batch_q)

        expected_distance = 1.4909
        for i in range(batch_size):
            self.assertAlmostEqual(result[i].item(), expected_distance, places=3)


if __name__ == '__main__':
    unittest.main()

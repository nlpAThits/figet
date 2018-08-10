
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
import unittest
import sys
sys.path.append("..")

from Loss import hyperbolic_distance_torch, hyperbolic_distance_numpy, hyperbolic_distance_batch


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

    def test_nearest_neighbor(self):
        neigh = NearestNeighbors(n_neighbors=5, algorithm='ball_tree', metric=hyperbolic_distance_numpy)
        p = np.array([0, 0.5])
        q = np.array([0.5, 0.5])
        neigh.fit([p])

        distances, indexes = neigh.kneighbors(q.reshape(1, -1), n_neighbors=1)

        self.assertAlmostEqual(distances.item(), 1.4909, places=3)


if __name__ == '__main__':
    unittest.main()

import unittest

import numpy as np

from main import generate_synthetic_data
from micrograd import Linear, Sequential, Tensor


class TestAutograd(unittest.TestCase):
    def test_matmul_exp_and_sum_propagate_gradients(self):
        inputs = Tensor([[1.0, 2.0]], requires_grad=True)
        weights = Tensor([[3.0], [4.0]], requires_grad=True)

        loss = (inputs @ weights).exp().sum(axis=0).sum()
        loss.backward()

        expected = np.exp(11.0)
        np.testing.assert_allclose(inputs.grad, [[3.0 * expected, 4.0 * expected]])
        np.testing.assert_allclose(weights.grad, [[expected], [2.0 * expected]])

    def test_addition_accumulates_and_unbroadcasts_both_gradients(self):
        left = Tensor([[1.0], [2.0]], requires_grad=True)
        right = Tensor([[3.0, 4.0]], requires_grad=True)

        (left + right).sum().backward()

        np.testing.assert_allclose(left.grad, [[2.0], [2.0]])
        np.testing.assert_allclose(right.grad, [[2.0, 2.0]])

    def test_sequential_runs_layers_and_returns_all_parameters(self):
        model = Sequential(Linear(2, 3), Linear(3, 1))
        output = model(Tensor([[1.0, -1.0]]))

        self.assertEqual(output.data.shape, (1, 1))
        self.assertEqual(len(model.parameters()), 4)

    def test_synthetic_data_has_requested_count_and_full_rings(self):
        samples, labels, one_hot = generate_synthetic_data(n_samples=5)

        self.assertEqual(samples.shape, (5, 2))
        self.assertEqual(labels.shape, (5,))
        self.assertEqual(one_hot.shape, (5, 2))
        self.assertEqual(np.count_nonzero(labels == 0), 3)
        self.assertEqual(np.count_nonzero(labels == 1), 2)
        self.assertTrue(np.allclose(one_hot.sum(axis=1), 1.0))


if __name__ == "__main__":
    unittest.main()
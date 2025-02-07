import unittest
import numpy as np

from task import PLSA


class TestPLSA(unittest.TestCase):
    def setUp(self):
        self.counts = np.array([[1, 2, 3], [4, 5, 6]])
        self.num_topics = 2
        self.seed = 42
        self.model = PLSA(self.counts, self.num_topics, self.seed)

    def test_initialization(self):
        # Check if Phi and Theta have the correct shape
        self.assertEqual(self.model.Phi.shape, (self.counts.shape[0], self.num_topics))
        self.assertEqual(self.model.Theta.shape, (self.num_topics, self.counts.shape[1]))

        # Check if Phi and Theta are normalized
        self.assertTrue(np.allclose(np.sum(self.model.Phi, axis=0), np.ones(self.num_topics)))
        self.assertTrue(np.allclose(np.sum(self.model.Theta, axis=0), np.ones(self.counts.shape[1])))

    def test_update_parameters(self):
        old_Phi = self.model.Phi.copy()
        old_Theta = self.model.Theta.copy()

        self.model._update_parameters()

        # Check if parameters are updated
        self.assertFalse(np.array_equal(old_Phi, self.model.Phi))
        self.assertFalse(np.array_equal(old_Theta, self.model.Theta))

    def test_perplexity(self):
        perplexity = self.model.perplexity()

        # Check if perplexity is a positive number
        self.assertGreater(perplexity, 0)


if __name__ == '__main__':
    unittest.main()

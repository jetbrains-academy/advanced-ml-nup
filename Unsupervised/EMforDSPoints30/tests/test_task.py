import numpy as np
import pandas as pd

import unittest

from task import DawidSkeneEM


class TestDawidSkeneEM(unittest.TestCase):
    def setUp(self):
        # Create a mock dataset
        data = {
            'user': ['user1', 'user2', 'user1', 'user3'],
            'object': ['obj1', 'obj1', 'obj2', 'obj2'],
            'label': ['label1', 'label1', 'label2', 'label2']
        }
        self.df = pd.DataFrame(data)

        self.num_usr = len(self.df['user'].unique())
        self.num_obj = len(self.df['object'].unique())
        self.num_lbl = len(self.df['label'].unique())

        self.model = DawidSkeneEM(self.df)

    def test_encode_data_and_get_counts(self):
        # Test if model.counts are correctly calculated and have the correct shape
        expected_shape = (self.num_usr, self.num_obj, self.num_lbl)

        model = self.model
        self.assertEqual(model.counts.shape, expected_shape, msg="The shape of self.counts is wrong.")

        # For example, user1 labeled obj1 with label1 once
        user1_idx = np.where(model.usr_arr == 'user1')[0]
        obj1_idx = np.where(model.obj_arr == 'obj1')[0]
        label1_idx = np.where(model.lbl_arr == 'label1')[0]
        self.assertEqual(model.counts[user1_idx, obj1_idx, label1_idx], 1,
                         msg="The value of self.counts is wrong.")

    def test_initialize_parameters(self):
        self.model.initialize_parameters()

        model = self.model
        self.assertEqual(model.pi.shape, (self.num_usr, self.num_lbl, self.num_lbl),
                         msg="The shape of self.pi is wrong.")

        self.assertEqual(model.rho.shape, (self.num_lbl,),
                         msg="The shape of self.rho is wrong.")

        self.assertTrue(np.all(model.pi >= 0) and np.all(model.pi <= 1),
                        msg="The values in pi should be probabilities.")

        # Test that values in rho are probabilities
        self.assertTrue(np.all(model.rho >= 0) and np.all(model.rho <= 1),
                        msg="The values in rho should be probabilities.")

    def test_e_step_output(self):
        # Initialize parameters for testing
        self.model.initialize_parameters()
        # Set some known values to pi and rho for predictable output
        self.model.pi = np.full((self.num_usr, self.num_lbl, self.num_lbl), 0.5)
        self.model.rho = np.full(self.num_lbl, 1.0 / self.num_lbl)
        posterior = self.model._e_step()

        self.assertEqual(posterior.shape, (self.num_obj, self.num_lbl),
                         msg="The shape of self.label_prob is wrong.")

        self.assertTrue(np.all(posterior >= 0) and np.all(posterior <= 1),
                        msg="The values in self.label_prob should be probabilities.")

        # Test that the sum of probabilities for each object is 1
        for i in range(self.num_obj):
            self.assertAlmostEqual(np.sum(posterior[i, :]), 1.0,
                                   msg="The values in self.label_prob should sum up to 1.")

    def test_m_step_updates(self):
        self.model.initialize_parameters()
        # Create a simple posterior distribution for testing
        posterior = np.zeros((self.num_obj, self.num_lbl))
        for i in range(self.num_obj):
            posterior[i, i % self.num_lbl] = 1  # Assigning deterministic posteriors for simplicity

        # Save old values of pi and rho for comparison
        old_pi = self.model.pi.copy()

        # Perform M-step
        self.model._m_step(posterior)

        self.assertFalse(np.array_equal(self.model.pi, old_pi),
                         msg="The values of self.pi should be updated.")

        self.assertTrue(np.all(self.model.pi >= 0) and np.all(self.model.pi <= 1),
                        msg="The values in self.pi should be probabilities.")

        self.assertTrue(np.all(self.model.rho >= 0) and np.all(self.model.rho <= 1),
                        msg="The values in self.rho should be probabilities.")

    def test_m_step_consistency(self):
        self.model.initialize_parameters()
        # Create a simple posterior distribution for testing
        posterior = np.zeros((self.num_obj, self.num_lbl))
        for i in range(self.num_obj):
            posterior[i, i % self.num_lbl] = 1  # Assigning deterministic posteriors for simplicity

        self.model._m_step(posterior)

        # Check if updates to rho are
        for k in range(self.num_lbl):
            self.assertAlmostEqual(self.model.rho[k], np.mean(posterior[:, k]),
                                   msg="The updates of self.rho should be consistent with posterior.")


if __name__ == '__main__':
    unittest.main()

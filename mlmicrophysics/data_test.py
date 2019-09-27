from .data import load_cam_output, uniform_stratify_data
import unittest
import pandas as pd
import numpy as np


class TestData(unittest.TestCase):
    def setUp(self):
        self.bad_path = "/blah/de/blah"
        return

    def test_load_cam_output(self):
        with self.assertRaises(FileNotFoundError):
            load_cam_output(self.bad_path)

    def test_uniform_stratify_data(self):
        samples = 50000
        random_seed = 1232
        np.random.seed(random_seed)
        output_labels = pd.DataFrame({"two_labels": np.random.random_integers(0, 2, samples),
                                      "three_labels": np.random.random_integers(-1, 2, samples)})
        scaled_output_values = pd.DataFrame({"two_labels": np.zeros(samples),
                                             "three_labels": np.zeros(samples)})
        category_size = 500
        num_bins = 10
        bins = {}
        for col in output_labels.columns:
            nonzero_samples = output_labels[col] != 0
            scaled_output_values.loc[nonzero_samples, col] = np.random.normal(size=np.count_nonzero(nonzero_samples))
            bins[col] = {}
            labels = np.unique(output_labels[col])
            for label in labels:
                if label != 0:
                    label_vals = scaled_output_values.loc[output_labels[col] == label, col].values
                    print(label_vals.min(), label_vals.max())
                    bins[col][label] = np.linspace(label_vals.min(), label_vals.max() + 0.1, num_bins)
        sampling_indices = uniform_stratify_data(output_labels, scaled_output_values, category_size, bins)
        for out_label in output_labels.columns:
            for label in sampling_indices[out_label].keys():
                self.assertEqual(sampling_indices[out_label][label].size, category_size)


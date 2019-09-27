from .plots import error_histogram, distribution_histogram
import numpy as np
import logging
import os
import unittest
from os.path import join, exists
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))


class PlotTester(unittest.TestCase):
    def setUp(self):
        self.observations = {"qrtend_TAU_1": np.random.normal(loc=-8, scale=3, size=10000),
                        "nctend_TAU_1": np.random.normal(loc=0, scale=10, size=10000)}
        self.predictions = {"qrtend_TAU_1": self.observations["qrtend_TAU_1"]
                                            + np.random.normal(loc=0, scale=1, size=10000),
                       "nctend_TAU_1": self.observations["nctend_TAU_1"]
                                       + np.random.normal(loc=0, scale=1, size=10000)}
        self.out_dir = "/tmp/"

    def test_error_histogram(self):
        out_path = join(self.out_dir, "error_histogram_test.png")
        error_histogram(self.observations, self.predictions, "TAU Bin", "ML Bin", out_path, dpi=80)
        self.assertTrue(exists(out_path))
        return

    def test_distribution_histogram(self):
        distributions = {"Observations": self.observations,
                         "Predictions": self.predictions}
        colors = {"Observations":"red", "Predictions": "blue"}
        out_path = join(self.out_dir, "distribution_histogram_test.png")
        distribution_histogram(distributions, np.array(["Observations", "Predictions"]),
                               np.array(["qrtend_TAU_1", "nctend_TAU_1"]), colors,
                               out_path)
        self.assertTrue(exists(out_path))
        return
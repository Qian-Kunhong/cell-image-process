import unittest

import numpy as np

from day2_trial import basc_a_threshold, validate_model_feature_names


class ModelATrialTests(unittest.TestCase):
    def test_basc_a_separates_clear_discontinuity(self):
        values = np.array([1.0, 1.2, 1.4, 1.6, 9.5, 10.0, 10.5, 11.0])
        threshold, binary = basc_a_threshold(values)
        self.assertGreater(threshold, 1.6)
        self.assertLess(threshold, 9.5)
        np.testing.assert_array_equal(binary, np.array([0, 0, 0, 0, 1, 1, 1, 1]))

    def test_model_feature_guard_rejects_oct4(self):
        with self.assertRaises(AssertionError):
            validate_model_feature_names(["area_px", "oct4_mean_intensity"])

    def test_model_feature_guard_accepts_dapi(self):
        validate_model_feature_names(["area_px", "dapi_mean_intensity"])


if __name__ == "__main__":
    unittest.main()

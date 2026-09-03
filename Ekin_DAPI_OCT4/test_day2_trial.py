import unittest

import numpy as np
import pandas as pd

from day2_trial import (
    COMPOSITE_FEATURES, MODEL_FEATURES, add_composite_features,
    basc_a_threshold, phenotype_colors, validate_model_feature_names,
)


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

    def test_model_feature_guard_rejects_markers_and_metadata(self):
        for name in ("yap_ratio", "AF488_mean", "ha_concentration_nM", "culture_day", "seeding_density"):
            with self.subTest(name=name), self.assertRaises(AssertionError):
                validate_model_feature_names(["area_px", name])

    def test_composites_are_finite_dimensionless_and_marker_independent(self):
        row = {name: 1.0 for name in MODEL_FEATURES}
        row.update({
            "image_id": "a", "area_px": 100.0, "perimeter_px": 40.0,
            "aspect_ratio": 2.0, "solidity": 0.9, "dapi_mean_intensity": 20.0,
            "dapi_std_intensity": 4.0, "dapi_intensity_range": 10.0,
            "nn1_distance_px": 20.0, "equivalent_diameter_px": 10.0,
            "local_crowding_area_fraction_proxy": 0.3,
            "neighbor_size_log_disagreement": 0.2,
            "neighbor_shape_disagreement": 0.1,
            "neighborhood_angular_asymmetry": 0.4,
            "oct4_mean_intensity": 999.0, "ha_concentration_nM": 5.0,
        })
        first = add_composite_features(pd.DataFrame([row]))
        row["oct4_mean_intensity"] = -999.0
        row["ha_concentration_nM"] = 0.0
        second = add_composite_features(pd.DataFrame([row]))
        np.testing.assert_allclose(first[COMPOSITE_FEATURES], second[COMPOSITE_FEATURES])
        self.assertTrue(np.isfinite(first[COMPOSITE_FEATURES].to_numpy()).all())

    def test_palette_is_global_not_subset_dependent(self):
        self.assertEqual(
            phenotype_colors(["Phenotype 3"])["Phenotype 3"],
            phenotype_colors(["Phenotype 1", "Phenotype 2", "Phenotype 3"])["Phenotype 3"],
        )


if __name__ == "__main__":
    unittest.main()

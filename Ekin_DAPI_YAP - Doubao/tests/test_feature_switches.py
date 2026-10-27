import unittest

from ekin_dapi_yap import model_a_pipeline as PIPELINE
import run_40x_analysis as CONTROL
import run_interpretable_phenotypes as INTERPRETABLE


class FeatureSwitchTests(unittest.TestCase):
    def test_no_intensity_profile_is_complete_and_explicit(self):
        self.assertEqual(set(CONTROL.FEATURE_SWITCHES), set(PIPELINE.CORE.ALL_SWITCHABLE_MODEL_FEATURES))
        disabled = {name for name, enabled in CONTROL.FEATURE_SWITCHES.items() if not enabled}
        self.assertEqual(disabled, {
            "dapi_mean_intensity", "dapi_std_intensity", "dapi_min_intensity",
            "dapi_max_intensity", "dapi_intensity_range",
            "nb_dapi_mean_intensity_mean", "nb_dapi_mean_intensity_std",
            "chromatin_cv_proxy", "chromatin_range_ratio",
        })
        enabled, reported_disabled = PIPELINE.CORE.resolve_model_features(
            "augmented", CONTROL.FEATURE_SWITCHES
        )
        self.assertTrue(enabled)
        self.assertEqual(set(reported_disabled), disabled)
        self.assertTrue(disabled.isdisjoint(enabled))

    def test_unknown_or_nonboolean_switch_rejected(self):
        with self.assertRaises(KeyError):
            PIPELINE.CORE.resolve_model_features("augmented", {"not_a_feature": False})
        with self.assertRaises(TypeError):
            PIPELINE.CORE.resolve_model_features("augmented", {"area_px": 1})

    def test_interpretable_profile_is_complete_and_has_neutral_current_labels(self):
        self.assertEqual(set(INTERPRETABLE.FEATURE_SWITCHES),
                         set(PIPELINE.CORE.ALL_SWITCHABLE_MODEL_FEATURES))
        enabled, _ = PIPELINE.CORE.resolve_model_features(
            "raw", INTERPRETABLE.FEATURE_SWITCHES)
        augmented, augmented_disabled = PIPELINE.CORE.resolve_model_features(
            "augmented", INTERPRETABLE.FEATURE_SWITCHES)
        self.assertEqual(enabled, augmented)
        self.assertTrue(all("intensity" not in name.lower() for name in enabled))
        self.assertTrue(set(PIPELINE.CORE.COMPOSITE_MODEL_FEATURES).issubset(augmented_disabled))
        self.assertEqual(len(INTERPRETABLE.PHENOTYPE_LABELS), 3)
        combined = " ".join(row["short_label"] for row in INTERPRETABLE.PHENOTYPE_LABELS).lower()
        self.assertNotIn("cell cycle", combined)
        self.assertNotIn("pluripot", combined)


if __name__ == "__main__":
    unittest.main()

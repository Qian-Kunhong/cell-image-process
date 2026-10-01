import unittest

from ekin_dapi_yap import model_a_pipeline as PIPELINE
import run_40x_analysis as CONTROL


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


if __name__ == "__main__":
    unittest.main()

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parent / "yap_40x_trial.py"
SPEC = importlib.util.spec_from_file_location("yap_40x_trial", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class YAPPipelineTests(unittest.TestCase):
    def test_folder_parser(self):
        match = MODULE.FOLDER_PATTERN.match("HA2-7_5-40-Image Export-33")
        self.assertIsNotNone(match)
        self.assertEqual(match.group("experimental_group"), "HA2")
        density_code = match.group("seeding_density_code").replace("_", ".")
        self.assertEqual(MODULE.SEEDING_DENSITY_CELLS_PER_CM2[density_code], 7_500.0)
        self.assertEqual(match.group("magnification"), "40")

    def test_experimental_group_metadata(self):
        self.assertEqual(MODULE.EXPERIMENTAL_GROUP_METADATA["Ctrl"]["ha_concentration_nM"], 0.0)
        self.assertIsNone(MODULE.EXPERIMENTAL_GROUP_METADATA["Ctrl"]["ha_exposure_h"])
        self.assertEqual(MODULE.EXPERIMENTAL_GROUP_METADATA["HA1"]["ha_exposure_h"], 72.0)
        self.assertEqual(MODULE.EXPERIMENTAL_GROUP_METADATA["HA1"]["ha_concentration_nM"], 2.5)
        self.assertEqual(MODULE.EXPERIMENTAL_GROUP_METADATA["HA2"]["ha_exposure_h"], 48.0)
        self.assertEqual(MODULE.EXPERIMENTAL_GROUP_METADATA["HA2"]["ha_concentration_nM"], 5.0)

    def test_model_feature_guard_rejects_yap(self):
        with self.assertRaises(AssertionError):
            MODULE.validate_morphology_feature_names(["area_px", "yap_nuclear_ratio"])
        with self.assertRaises(AssertionError):
            MODULE.validate_morphology_feature_names(["area_px", "AF488_mean"])
        with self.assertRaises(AssertionError):
            MODULE.validate_morphology_feature_names(["area_px", "seeding_density_cells_per_cm2"])
        MODULE.validate_morphology_feature_names(["area_px", "dapi_mean_intensity"])

    def test_ring_labels_do_not_overlap_nuclei(self):
        mask = np.zeros((40, 50), dtype=np.int32)
        mask[10:16, 10:16] = 1
        mask[10:16, 25:31] = 2
        rings, inner, outer, _ = MODULE.build_ring_labels(mask)
        self.assertGreater(outer, inner)
        self.assertFalse(np.any((rings > 0) & (mask > 0)))
        self.assertTrue(set(np.unique(rings)).issubset({0, 1, 2}))


if __name__ == "__main__":
    unittest.main()

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch
from io import BytesIO

import numpy as np
import pandas as pd
from skimage.segmentation import expand_labels

from yap_ratio_qc import YAPQCConfig, build_ring_labels, measure_array
from refresh_yap_posthoc import replace_posthoc


def synthetic(nuclear=110., cytoplasm=60.):
    rr, cc = np.indices((160, 160))
    mask = np.zeros((160, 160), np.int32)
    mask[(rr-80)**2 + (cc-80)**2 <= 15**2] = 1
    image = np.full(mask.shape, 10.)
    image[expand_labels(mask, distance=20) > 0] = cytoplasm
    image[mask > 0] = nuclear
    return image, mask


class YAPRatioQCTests(unittest.TestCase):
    def test_uniform_known_ratio(self):
        image, mask = synthetic()
        table, rings, info = measure_array(image, mask, "test")
        row = table.iloc[0]
        self.assertTrue(row.posthoc_yap_ratio_valid)
        self.assertAlmostEqual(row.posthoc_yap_nuclear_perinuclear_ratio, 2.)
        self.assertAlmostEqual(row.posthoc_yap_mean_nuclear_perinuclear_ratio, 2.)
        self.assertFalse(np.any((rings > 0) & (mask > 0)))
        self.assertEqual(info["background_noise_scale_dn"], 1.)

    def test_true_high_ratio_not_removed_for_being_high(self):
        image, mask = synthetic(nuclear=210, cytoplasm=20)
        row = measure_array(image, mask, "test")[0].iloc[0]
        self.assertTrue(row.posthoc_yap_ratio_valid)
        self.assertAlmostEqual(row.posthoc_yap_nuclear_perinuclear_ratio, 20.)

    def test_partial_ring_warns_without_bright_pixel_selection(self):
        image, mask = synthetic()
        rings, *_ = build_ring_labels(mask, .20, 0)
        _, cc = np.indices(mask.shape)
        image[(rings > 0) & (cc > 80)] = 10
        row = measure_array(image, mask, "test")[0].iloc[0]
        self.assertTrue(row.posthoc_yap_sampling_valid)
        self.assertTrue(row.posthoc_yap_ratio_valid)
        self.assertIn("ring_low_signal_fraction", row.posthoc_yap_qc_warnings)
        self.assertEqual(row.posthoc_yap_perinuclear_mean_raw, float(image[rings == 1].mean()))
        self.assertEqual(row.posthoc_yap_nuclear_perinuclear_ratio, row.posthoc_yap_unfiltered_narrow_ratio)

    def test_small_denominator_no_epsilon_ratio(self):
        image, mask = synthetic(cytoplasm=11)
        row = measure_array(image, mask, "test")[0].iloc[0]
        self.assertAlmostEqual(row.posthoc_yap_legacy_v1_ratio, 100.)
        self.assertTrue(row.posthoc_yap_ratio_valid)
        self.assertTrue(row.posthoc_yap_nuclear_detected_ring_unresolved)
        self.assertIn("denominator_below_noise_gate", row.posthoc_yap_qc_warnings)
        self.assertAlmostEqual(row.posthoc_yap_nuclear_perinuclear_ratio, 100.)

    def test_zero_and_negative_corrected_denominator(self):
        for level in (10, 5):
            image, mask = synthetic(cytoplasm=level)
            row = measure_array(image, mask, "test")[0].iloc[0]
            self.assertFalse(row.posthoc_yap_ratio_valid)
            self.assertTrue(np.isnan(row.posthoc_yap_unfiltered_narrow_ratio))

    def test_no_ring_pixels_in_fully_packed_mask(self):
        mask = np.ones((30, 30), dtype=np.int32)
        mask[:, 15:] = 2
        image = np.full(mask.shape, 50.)
        table, _, _ = measure_array(image, mask, "test")
        self.assertEqual(table.posthoc_yap_ring_area_px.sum(), 0)
        self.assertFalse(table.posthoc_yap_ratio_valid.any())
        self.assertFalse(table.posthoc_yap_sampling_valid.any())
        self.assertFalse(table.posthoc_yap_raw_ratio_valid.any())

    def test_hash_compatible_with_python310(self):
        from refresh_yap_posthoc import sha256
        with patch.object(Path, "open", return_value=BytesIO(b"")):
            self.assertEqual(sha256("empty"), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")

    def test_no_background_disables_ratio_but_preserves_raw(self):
        image, mask = synthetic()
        table, _, info = measure_array(image, mask, "test", replace(YAPQCConfig(), min_background_pixels=100000))
        self.assertFalse(info["background_adequate"])
        self.assertEqual(table.iloc[0].posthoc_yap_nuclear_median_raw, 110.)
        self.assertEqual(table.iloc[0].posthoc_yap_qc_reasons, "background_unavailable")
        self.assertTrue(np.isnan(table.iloc[0].posthoc_yap_nuclear_perinuclear_ratio))
        self.assertTrue(table.iloc[0].posthoc_yap_sampling_valid)
        self.assertTrue(table.iloc[0].posthoc_yap_raw_ratio_valid)
        self.assertAlmostEqual(table.iloc[0].posthoc_yap_raw_nuclear_perinuclear_ratio, 110/60)

    def test_user_background_roi(self):
        image, mask = synthetic()
        roi = np.zeros_like(mask, bool)
        roi[:20, :20] = True
        table, _, info = measure_array(image, mask, "test", background_roi=roi)
        self.assertTrue(table.iloc[0].posthoc_yap_ratio_valid)
        self.assertEqual(info["background_method"], "user_supplied_cell_free_background_ROI")
        roi[80, 80] = True
        with self.assertRaises(ValueError):
            measure_array(image, mask, "test", background_roi=roi)

    def test_near_frame_edge_only_warns_for_yap(self):
        image, mask = synthetic()
        image, mask = image[:, 61:], mask[:, 61:]
        row = measure_array(image, mask, "test")[0].iloc[0]
        self.assertIn("ring_near_image_boundary", row.posthoc_yap_qc_warnings)
        self.assertTrue(row.posthoc_yap_sampling_valid)
        self.assertTrue(row.posthoc_yap_ratio_valid)

    def test_saturated_export_flagged(self):
        image, mask = synthetic(nuclear=255)
        row = measure_array(image, mask, "test")[0].iloc[0]
        self.assertIn("export_intensity_clipping", row.posthoc_yap_qc_warnings)
        self.assertTrue(row.posthoc_yap_ratio_valid)

    def test_all_nuclei_excluded_from_all_rings(self):
        image, mask = synthetic()
        mask[70:80, 101:111] = 2
        before = mask.copy()
        _, rings, _ = measure_array(image, mask, "test")
        np.testing.assert_array_equal(mask, before)
        self.assertFalse(np.any((mask > 0) & (rings > 0)))
        self.assertEqual(set(np.unique(rings)), {0, 1, 2})

    def test_bright_neighbor_nucleus_cannot_change_own_ring_intensity(self):
        image, mask = synthetic()
        mask[70:90, 97:110] = 2
        image[mask == 2] = 30
        first, rings, info = measure_array(image, mask, "test")
        image[mask == 2] = 250
        second, _, _ = measure_array(image, mask, "test")
        self.assertEqual(info["sampling_pixels_overlapping_dapi_nuclei"], 0)
        self.assertEqual(first.loc[first.label.eq(1), "posthoc_yap_perinuclear_mean_raw"].iloc[0],
                         second.loc[second.label.eq(1), "posthoc_yap_perinuclear_mean_raw"].iloc[0])
        self.assertFalse(np.any((rings == 1) & (mask == 2)))

    def test_one_pixel_region_is_retained(self):
        # A fully enclosed synthetic seed with one unlabeled pixel tests the
        # minimum-pixel rule, not biological validity of a real segmentation.
        mask = np.zeros((200, 200), dtype=np.int32)
        mask[65:135, 65:135] = 2
        mask[85:115, 85:115] = 1
        mask[100, 100] = 0
        image = np.full(mask.shape, 10.)
        image[mask > 0] = 110
        image[100, 100] = 60
        table, _, _ = measure_array(image, mask, "test")
        row = table.loc[table.label.eq(1)].iloc[0]
        self.assertEqual(row.posthoc_yap_ring_area_px, 1)
        self.assertTrue(row.posthoc_yap_sampling_valid)
        self.assertAlmostEqual(row.posthoc_yap_raw_nuclear_perinuclear_ratio, 110/60)
        self.assertIn("insufficient_ring_area", row.posthoc_yap_qc_warnings)

    def test_default_has_no_buffer_and_keeps_every_non_nuclear_pixel_in_radius(self):
        image, mask = synthetic()
        _, rings, info = measure_array(image, mask, "test")
        expected = expand_labels(mask, distance=info["outer_distance_px"])
        expected[mask > 0] = 0
        np.testing.assert_array_equal(rings, expected)
        self.assertEqual(info["inner_gap_px"], 0)

    def test_soft_warning_thresholds_do_not_change_measurement_availability(self):
        image, mask = synthetic()
        config = replace(YAPQCConfig(), warning_ring_pixels=100000, min_ring_nucleus_area_ratio=100,
                         min_snr=1000, max_saturated_fraction=0.)
        row = measure_array(image, mask, "test", config)[0].iloc[0]
        self.assertTrue(row.posthoc_yap_ratio_valid)
        self.assertFalse(row.posthoc_yap_strict_gate_pass_diagnostic_only)
        self.assertAlmostEqual(row.posthoc_yap_nuclear_perinuclear_ratio, 2.)

    def test_stricter_snr_is_subset(self):
        image, mask = synthetic(cytoplasm=14)
        row = measure_array(image, mask, "test")[0].iloc[0]
        self.assertTrue(row.posthoc_yap_sensitivity_snr_2_pass)
        self.assertTrue(row.posthoc_yap_sensitivity_snr_3_pass)
        self.assertFalse(row.posthoc_yap_sensitivity_snr_5_pass)

    def test_full_model_rows_columns_and_posteriors_preserved(self):
        image, mask = synthetic()
        measurement = measure_array(image, mask, "test")[0]
        core = measurement[["cell_id", "image_id", "label"]].copy()
        core["dapi_mean_intensity"] = 123.
        core["umap_1"] = .8
        core["dominant_phenotype"] = "Phenotype 1"
        core["P_phenotype_1"] = .7
        core["P_phenotype_2"] = .3
        core["posthoc_yap_old"] = -1
        new = replace_posthoc(core, measurement)
        columns = [c for c in core if not c.startswith("posthoc_yap_")]
        pd.testing.assert_frame_equal(new[columns], core[columns], check_exact=True)
        self.assertNotIn("posthoc_yap_old", new)
        measurement.loc[0, "cell_id"] = "missing"
        with self.assertRaises(ValueError):
            replace_posthoc(core, measurement)

    def test_shapes_empty_and_config_fail_loudly(self):
        image, mask = synthetic()
        with self.assertRaises(ValueError):
            measure_array(image[:, :-1], mask, "test")
        with self.assertRaises(ValueError):
            measure_array(image, np.zeros_like(mask), "test")
        with self.assertRaises(ValueError):
            YAPQCConfig(noise_floor_dn=0)
        with self.assertRaises(ValueError):
            YAPQCConfig(min_signal_fraction=1.1)


if __name__ == "__main__":
    unittest.main()

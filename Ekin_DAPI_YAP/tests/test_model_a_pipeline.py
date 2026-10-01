from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


from ekin_dapi_yap import model_a_pipeline as MODULE


class ModelAPipelineTests(unittest.TestCase):
    def test_experimental_group_palette_distinct_from_current_phenotypes(self):
        from matplotlib.colors import to_hex
        group_colors = {to_hex(color) for color in MODULE.EXPERIMENTAL_GROUP_COLORS.values()}
        phenotype_colors = {to_hex(color) for color in
                            MODULE.phenotype_colors([f"Phenotype {i}" for i in range(1, 7)]).values()}
        self.assertEqual(len(group_colors), 3)
        self.assertTrue(group_colors.isdisjoint(phenotype_colors))

    def test_composition_counts_do_not_overlap_titles(self):
        cells = pd.DataFrame([
            dict(experimental_group_label=group, seeding_density_cells_per_cm2=density,
                 dominant_phenotype="Phenotype 1", magnification="40x")
            for group in ("Ctrl", "HA1", "HA2") for density in (2500., 5000., 7500., 10000.)])
        original = cells.copy(deep=True)
        def inspect_layout(fig, *args, **kwargs):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            main_title = fig._suptitle.get_window_extent(renderer)
            for ax, group_label in zip(fig.axes, ("Control (No HA)", "HA-1 (72h, 2.5nM)", "HA-2 (48h, 5nM)")):
                self.assertEqual(ax.get_title(), "")
                self.assertEqual(ax.get_xlabel(), group_label)
                group_box = ax.xaxis.label.get_window_extent(renderer)
                counts = [text for text in ax.texts if text.get_text().startswith("n=")]
                self.assertEqual(len(counts), 4)
                for text in counts:
                    count_box = text.get_window_extent(renderer)
                    self.assertGreater(count_box.y0, ax.get_window_extent(renderer).y1)
                    self.assertFalse(count_box.overlaps(main_title))
                for tick in ax.get_xticklabels():
                    self.assertNotIn("n=", tick.get_text())
                    self.assertFalse(tick.get_window_extent(renderer).overlaps(group_box))
        with patch("matplotlib.figure.Figure.savefig", autospec=True, side_effect=inspect_layout):
            MODULE.save_phenotype_composition_plot(cells, Path("unused"))
        pd.testing.assert_frame_equal(cells, original)

    def test_yap_condition_plot_separates_definitions_and_keeps_empty_conditions(self):
        cells = pd.DataFrame({"experimental_group_label": ["Ctrl", "HA1"],
                              "seeding_density_cells_per_cm2": [2500., 5000.],
                              "image_id": ["a", "b"],
                              "posthoc_yap_raw_log2_nuclear_perinuclear_ratio": [.25, .5],
                              "posthoc_yap_log2_nuclear_perinuclear_ratio": [1., np.nan]})
        original = cells.copy(deep=True)
        with patch("matplotlib.figure.Figure.savefig"), patch.object(Path, "mkdir"):
            raw = MODULE.save_yap_condition_boxplot(cells, Path("unused"))
            corrected = MODULE.save_yap_condition_boxplot(cells, Path("unused"), raw=False)
        self.assertEqual(len(raw), 12)
        self.assertEqual(len(corrected), 12)
        self.assertEqual(raw.n_available.sum(), 2)
        self.assertEqual(corrected.n_available.sum(), 1)
        self.assertEqual(raw.iloc[0].median_log2_ratio, .25)
        self.assertEqual(corrected.iloc[0].median_log2_ratio, 1.)
        self.assertTrue(np.isnan(corrected.iloc[5].median_log2_ratio))
        pd.testing.assert_frame_equal(cells, original)

    def test_run_does_not_shadow_yap_qc_summary(self):
        self.assertTrue(callable(MODULE.qc_summary))
        self.assertNotIn("qc_summary", MODULE.run.__code__.co_varnames)
        self.assertIn("dapi_qc_summary", MODULE.run.__code__.co_varnames)

    def test_square_umap_layout_and_data_unchanged(self):
        cells = pd.DataFrame({"umap_1": [1., 3., 8.], "umap_2": [2., 20., 5.],
                              "magnification": ["40x"]*3,
                              "dominant_phenotype": ["Phenotype 1", "Phenotype 2", "Phenotype 1"],
                              "experimental_group_label": ["Ctrl", "HA1", "HA2"]})
        original = cells.copy(deep=True)
        with patch("matplotlib.figure.Figure.savefig") as save:
            layouts = MODULE.save_square_umap_scatter_plots(cells, Path("unused"))
        self.assertEqual(save.call_count, 2)
        first, second = list(layouts.values())
        self.assertEqual(first["xlim"], second["xlim"])
        self.assertEqual(first["ylim"], second["ylim"])
        for item in layouts.values():
            self.assertEqual(item["canvas_pixels"], [2200, 2200])
            self.assertAlmostEqual(item["axes_box_aspect"], 1.)
            self.assertAlmostEqual(np.diff(item["xlim"])[0], np.diff(item["ylim"])[0])
            self.assertTrue(item["legend_outside_axes"])
        pd.testing.assert_frame_equal(cells, original)

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

from pathlib import Path
import unittest

import numpy as np
import pandas as pd


import make_ppt_figures as MODULE
MODULE_PATH = Path(MODULE.__file__).resolve()


class PPTFigureTests(unittest.TestCase):
    def test_group_palette_is_high_contrast_and_unique(self):
        self.assertEqual(len(set(MODULE.GROUP_COLORS.values())), 3)
        self.assertEqual(MODULE.GROUP_COLORS["Ctrl"], "#171717")
        self.assertGreaterEqual(len(set(MODULE.PHENOTYPE_PALETTE)), 12)

    def test_titles_and_labels_are_centralized(self):
        self.assertEqual(len(MODULE.TITLES), 16)
        self.assertEqual(tuple(MODULE.GROUP_LABELS), MODULE.GROUP_ORDER)
        self.assertEqual(set(MODULE.PHENOTYPE_NOTES), set(MODULE.HEATMAP_PHENOTYPE_NOTES))
        self.assertEqual(MODULE.TITLES["composition"], "Morphology phenotype counts")
        self.assertIn("14_umap_dominant_phenotypes", MODULE.SVG_NOTES)
        self.assertIn("excluded", MODULE.SVG_NOTES["14_umap_dominant_phenotypes"])

    def test_gmm_glossary_covers_displayed_terms(self):
        self.assertEqual(
            set(MODULE.GMM_TERM_NOTES),
            {"K", "Penalized BIC", "Delta penalized BIC", "Selected K"},
        )
        self.assertIn("not proof", MODULE.GMM_TERM_NOTES["Selected K"])

    def test_ha_effect_is_density_matched_and_control_referenced(self):
        rows = []
        for group, values in (("Ctrl", [1.0, 2.0, 3.0, 4.0]),
                              ("HA1", [4.0, 5.0, 6.0, 7.0]),
                              ("HA2", [1.0, 2.0, 3.0, 4.0])):
            for value in values:
                row = {
                    "experimental_group_label": group,
                    "seeding_density_cells_per_cm2": 2500.0,
                    MODULE.YAP_RAW_COLUMN: 0.2,
                }
                row.update({feature: value for feature in MODULE.HA_MORPHOLOGY_METRICS})
                rows.append(row)
        effects = MODULE.summarize_ha_effects(pd.DataFrame(rows))
        ctrl = effects[
            effects.experimental_group_label.eq("Ctrl")
            & effects.seeding_density_cells_per_cm2.eq(2500.0)
        ]
        ha1 = effects[
            effects.experimental_group_label.eq("HA1")
            & effects.seeding_density_cells_per_cm2.eq(2500.0)
        ]
        self.assertTrue(np.allclose(ctrl.control_referenced_effect, 0.0))
        self.assertTrue(np.allclose(ha1.control_referenced_effect, 2.0))
        conditions = MODULE.summarize_ha_conditions(pd.DataFrame(rows), effects)
        ha1_condition = conditions[
            conditions.experimental_group_label.eq("HA1")
            & conditions.seeding_density_cells_per_cm2.eq(2500.0)
        ].iloc[0]
        self.assertAlmostEqual(ha1_condition.morphology_deviation_from_control, 2.0)
        across_density = MODULE.summarize_ha_across_density_strata(effects)
        ha1_summary = across_density[
            across_density.experimental_group_label.eq("HA1")
        ]
        self.assertTrue(np.allclose(ha1_summary.median_effect, 2.0))
        yap_effects = MODULE.summarize_yap_effects(pd.DataFrame(rows))
        available = yap_effects[yap_effects.seeding_density_cells_per_cm2.eq(2500.0)]
        self.assertTrue(np.allclose(available.effect_vs_density_matched_control, 0.0))

    def test_phenotype_order_is_numeric(self):
        cells = pd.DataFrame({"dominant_phenotype": ["Phenotype 10", "Phenotype 2", "Phenotype 1"]})
        self.assertEqual(MODULE.phenotype_order(cells), ["Phenotype 1", "Phenotype 2", "Phenotype 10"])

    def test_umap_orientation_is_rigid(self):
        cells = pd.DataFrame({
            "umap_1": [0.0, 1.0, 2.0, 3.0],
            "umap_2": [0.0, 2.0, 1.0, 4.0],
        })
        original = cells[["umap_1", "umap_2"]].to_numpy(float)
        rotated = MODULE.oriented_umap(cells)
        original_distances = np.linalg.norm(original[:, None, :] - original[None, :, :], axis=2)
        rotated_distances = np.linalg.norm(rotated[:, None, :] - rotated[None, :, :], axis=2)
        self.assertTrue(np.allclose(original_distances, rotated_distances))
        self.assertGreaterEqual(np.var(rotated[:, 0]), np.var(rotated[:, 1]))

    def test_composition_uses_actual_counts_and_labels_totals(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn('f"n={int(total)}"', source)
        self.assertIn('ax.bar(x, values, bottom=bottom', source)


if __name__ == "__main__":
    unittest.main()

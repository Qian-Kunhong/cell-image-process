import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_ha_uniformity as ha


class HAUniformityTests(unittest.TestCase):
    def synthetic_cells(self):
        rng = np.random.default_rng(7)
        rows = []
        for density in ha.DENSITY_ORDER:
            for group, scale in (("Ctrl", 1.0), ("HA1", .45), ("HA2", 1.7)):
                for index in range(80):
                    row = {
                        "experimental_group_label": group,
                        "seeding_density_cells_per_cm2": density,
                        "image_id": f"{group}_{density:g}",
                        ha.YAP_RAW: rng.normal(.3, scale * .2),
                        ha.YAP_CORRECTED: rng.normal(.4, scale * .2),
                    }
                    for feature, spec in ha.MORPHOLOGY_FEATURES.items():
                        base = 100.0 if feature == "area_px" else 1.5
                        value = base + rng.normal(0, scale * (.08 * base))
                        if spec["transform"] in {"log", "log1p"}:
                            value = max(value, .001)
                        row[feature] = value
                    rows.append(row)
        return pd.DataFrame(rows)

    def test_known_lower_and_higher_dispersion_have_expected_signs(self):
        cells = self.synthetic_cells()
        multivariate, feature, _ = ha.calculate_uniformity(cells)
        ha1 = multivariate[multivariate.experimental_group_label.eq("HA1")]
        ha2 = multivariate[multivariate.experimental_group_label.eq("HA2")]
        self.assertTrue((ha1.uniformity_gain_vs_ctrl_log2 > 0).all())
        self.assertTrue((ha2.uniformity_gain_vs_ctrl_log2 < 0).all())
        self.assertTrue((feature[feature.experimental_group_label.eq("HA1")]
                         .uniformity_gain_vs_ctrl_log2.median() > 0))

    def test_yap_is_separate_and_gmm_or_intensity_are_not_required(self):
        cells = self.synthetic_cells()
        ha.validate_cells(cells)
        multivariate, _, _ = ha.calculate_uniformity(cells)
        self.assertNotIn("dominant_phenotype", cells.columns)
        self.assertNotIn("dapi_mean_intensity", cells.columns)
        self.assertTrue(np.isfinite(multivariate.multivariate_dispersion).all())

    def test_domain_and_leave_one_feature_out_sensitivity_are_complete(self):
        domains, sensitivity = ha.calculate_domain_uniformity(self.synthetic_cells())
        expected_domain_rows = len(ha.MORPHOLOGY_DOMAINS) * len(ha.DENSITY_ORDER) * len(ha.GROUP_ORDER)
        expected_sensitivity_rows = len(ha.MORPHOLOGY_FEATURES) * len(ha.DENSITY_ORDER) * len(ha.GROUP_ORDER)
        self.assertEqual(len(domains), expected_domain_rows)
        self.assertEqual(len(sensitivity), expected_sensitivity_rows)
        self.assertEqual(set(sensitivity.omitted_feature), set(ha.MORPHOLOGY_FEATURES))

    def test_end_to_end_writes_tables_figures_and_no_p_values(self):
        cells = self.synthetic_cells()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "result"
            (root / "tables").mkdir(parents=True)
            cells.to_csv(root / "tables" / "model_a_single_cell_results.csv", index=False)
            result = ha.run(root)
            analysis = Path(result["analysis_dir"])
            self.assertTrue((analysis / "figures" / "01_ha_morphology_uniformity.png").exists())
            self.assertTrue((analysis / "HA_UNIFORMITY_REPORT.md").exists())
            summary = pd.read_csv(analysis / "tables" / "ha_uniformity_summary.csv")
            self.assertFalse(any("p_value" in column.lower() for column in summary.columns))


if __name__ == "__main__":
    unittest.main()

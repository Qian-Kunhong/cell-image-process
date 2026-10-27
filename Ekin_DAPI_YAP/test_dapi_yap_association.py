import unittest
import numpy as np
import pandas as pd

from analyze_dapi_yap_association import calculate_associations


class DAPIYAPAssociationTests(unittest.TestCase):
    def test_within_field_summary_is_not_pooled_only(self):
        rows = []
        for image, offset in (("a", 0), ("b", 100)):
            for value in range(25):
                rows.append({
                    "image_id": image, "experimental_group_label": "Ctrl",
                    "seeding_density_cells_per_cm2": 2500,
                    "feature": value + offset,
                    "posthoc_yap_raw_log2_nuclear_perinuclear_ratio": value,
                    "posthoc_yap_log2_nuclear_perinuclear_ratio": value,
                })
        summary, fields = calculate_associations(pd.DataFrame(rows), ["feature"])
        raw = summary[summary.measurement.eq("raw")].iloc[0]
        self.assertEqual(raw.n_fields, 2)
        self.assertAlmostEqual(raw.within_field_median_rho, 1.0)
        self.assertEqual(len(fields[fields.measurement.eq("raw")]), 2)


if __name__ == "__main__":
    unittest.main()

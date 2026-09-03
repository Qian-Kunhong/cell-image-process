import unittest
import numpy as np
import pandas as pd
from phenotype_display import categorical_rgb, qc_keep_mask, low_confidence_rows


class DisplayTests(unittest.TestCase):
    def test_display_threshold_strictly_below_half(self):
        rows = pd.DataFrame({"gmm_max_posterior": [0.49, 0.5, 0.51, 0.79]})
        self.assertEqual(low_confidence_rows(rows).index.tolist(), [0])

    def setUp(self):
        self.mask = np.zeros((12, 14), dtype=np.int32)
        self.mask[0:3, 2:5] = 1
        self.mask[4:7, 4:7] = 2
        self.mask[7:10, 8:11] = 3
        self.rows = pd.DataFrame({"label": [2, 3], "dominant_phenotype": ["Phenotype 5", "Phenotype 6"]})

    def test_excluded_removed_without_mutating_or_renumbering_raw(self):
        original = self.mask.copy()
        clean = qc_keep_mask(self.mask, self.rows)
        np.testing.assert_array_equal(self.mask, original)
        self.assertEqual(set(np.unique(clean)), {0, 2, 3})

    def test_each_edge_is_rejected_if_in_result_table(self):
        for transform in (lambda x: x, np.flipud, lambda x: x.T, lambda x: np.fliplr(x.T)):
            with self.subTest(transform=transform), self.assertRaises(ValueError):
                qc_keep_mask(transform(self.mask), pd.DataFrame({"label": [1]}))

    def test_categorical_pixels_equal_legend_colors(self):
        colors = {"Phenotype 5": (.17, .63, .17, 1), "Phenotype 6": (.60, .87, .54, 1)}
        clean = qc_keep_mask(self.mask, self.rows)
        rgb = categorical_rgb(clean, self.rows, colors)
        for label, name in zip(self.rows.label, self.rows.dominant_phenotype):
            self.assertTrue(np.all(rgb[clean == label] == np.round(255 * np.array(colors[name][:3]))))
        self.assertTrue(np.all(rgb[clean == 0] == 0))

    def test_one_pixel_inset_is_rejected_by_edge_buffer(self):
        mask = np.zeros((12, 14), dtype=np.int32)
        mask[1:4, 5:8] = 4
        with self.assertRaises(ValueError):
            qc_keep_mask(mask, pd.DataFrame({"label": [4]}), edge_buffer_px=2)


if __name__ == "__main__":
    unittest.main()

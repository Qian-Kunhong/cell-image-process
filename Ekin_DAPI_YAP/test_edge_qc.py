from pathlib import Path
from unittest.mock import patch
import unittest
import numpy as np
import yap_40x_trial as pipeline


class EdgeQCTests(unittest.TestCase):
    def test_border_objects_neither_fit_nuclei_nor_shape_neighbors(self):
        mask = np.zeros((50, 50), dtype=np.int32)
        mask[:4, 1:5] = 1
        mask[47:49, 10:15] = 2  # one-pixel gap: old exact-border rule missed this
        mask[10:15, 10:15] = 3
        mask[25:30, 25:30] = 4
        item = pipeline.YAPImageSet("Ctrl", "5", 5000, "40x", 1, "test", Path("."), Path("dapi.png"), Path("yap.png"), Path("merge.png"))
        with patch.object(pipeline, "load_gray", return_value=np.ones((50, 50))*40), patch.object(pipeline.np, "load", return_value=mask):
            frame = pipeline.extract_dapi_features(item, Path("mask.npy"), edge_buffer_px=2).set_index("label")
        self.assertEqual(frame.loc[1, "qc_reason"], "touches_border")
        self.assertEqual(frame.loc[2, "qc_reason"], "within_2px_image_edge")
        self.assertEqual(frame.index[frame.qc_keep].tolist(), [3, 4])
        self.assertTrue(frame.loc[[1, 2], "nb_area_mean_px2"].isna().all())
        self.assertTrue((frame.loc[[3, 4], "adaptive_neighbor_count"] == 1).all())
        self.assertTrue((frame.loc[[3, 4], "nb_area_mean_px2"] == 25).all())


if __name__ == "__main__":
    unittest.main()

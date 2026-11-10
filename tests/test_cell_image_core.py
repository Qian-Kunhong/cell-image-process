"""Golden-array checks for the shared computation kernel."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "Ekin_DAPI_YAP"))

from cell_image_core.contracts import FeatureTable, SampleRecord, SegmentationResult
from cell_image_core.features import add_composite_features, add_spatial_features, extract_morphology_2d
from cell_image_core.io import normalize_percentile_uint8, to_three_channel
from cell_image_core.segmentation import segment_cellpose, separate_touching_labels


class CoreContractTests(unittest.TestCase):
    def test_contracts_reject_invalid_shapes_and_accept_paths(self):
        record = SampleRecord("sample", "image", Path("dapi.png"), {"YAP": Path("yap.png")})
        self.assertEqual(record.channels["YAP"], Path("yap.png"))
        result = SegmentationResult("image", np.zeros((3, 4), dtype=np.int32), "test")
        self.assertEqual(result.labels.shape, (3, 4))
        with self.assertRaises(ValueError):
            SegmentationResult("image", np.zeros((2, 3, 4), dtype=np.int32), "test")

    def test_normalization_exposes_legacy_rounding(self):
        image = np.arange(100, dtype=np.float32).reshape(10, 10)
        p1, p99 = np.percentile(image, [1, 99])
        scaled = np.clip((image - p1) / (p99 - p1 + 1e-6), 0, 1)
        expected = (scaled * 255).astype(np.uint8)
        actual = normalize_percentile_uint8(
            image,
            denominator_floor=0.0,
            denominator_offset=1e-6,
            round_values=False,
            finite_only=False,
        )
        np.testing.assert_array_equal(actual, expected)

        narrow = np.linspace(0.0, 1e-5, 100, dtype=np.float32).reshape(10, 10)
        p1, p99 = np.percentile(narrow, [1, 99])
        expected_narrow = (
            np.clip((narrow - p1) / (p99 - p1 + 1e-6), 0, 1) * 255
        ).astype(np.uint8)
        actual_narrow = normalize_percentile_uint8(
            narrow,
            denominator_floor=0.0,
            denominator_offset=1e-6,
            round_values=False,
            finite_only=False,
        )
        np.testing.assert_array_equal(actual_narrow, expected_narrow)

    def test_safe_division_keeps_legacy_missing_perimeter_behavior(self):
        from cell_image_core.features import circularity, safe_div

        self.assertTrue(np.isnan(safe_div(1.0, None)))
        self.assertTrue(np.isnan(circularity(10.0, None)))

    def test_three_channel_conversion(self):
        image = np.arange(12).reshape(3, 4)
        converted = to_three_channel(image)
        self.assertEqual(converted.shape, (3, 4, 3))
        np.testing.assert_array_equal(converted[..., 0], image)

    def test_cellpose_adapter_records_parameters_without_loading_a_model(self):
        class FakeModel:
            def eval(self, image, **kwargs):
                self.image = image
                self.kwargs = kwargs
                return np.zeros(image.shape[:2], dtype=np.int32), None, None

        model = FakeModel()
        result = segment_cellpose(
            np.ones((8, 9), dtype=np.float32),
            model=model,
            image_id="field",
            flow_threshold=0.25,
            cellprob_threshold=-0.1,
            diameter=17,
            extra_parameters={"normalize": False},
        )
        self.assertEqual(model.image.shape, (8, 9, 3))
        self.assertEqual(result.labels.shape, (8, 9))
        self.assertEqual(result.parameters["diameter"], 17)
        self.assertIs(model.kwargs["normalize"], False)
        with self.assertRaises(ValueError):
            segment_cellpose(
                np.ones((8, 9), dtype=np.float32),
                model=model,
                image_id="field",
                extra_parameters={"diameter": 10},
            )


class MorphologyGoldenArrayTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((50, 50), dtype=np.int32)
        self.mask[:4, 1:5] = 1
        self.mask[47:49, 10:15] = 2
        self.mask[10:15, 10:15] = 3
        self.mask[25:30, 25:30] = 4
        self.image = np.arange(2500, dtype=np.float32).reshape(50, 50)

    def test_edge_qc_and_feature_contract(self):
        frame = extract_morphology_2d(self.mask, self.image, image_id="field", edge_buffer_px=2)
        FeatureTable(frame)
        indexed = frame.set_index("label")
        self.assertEqual(indexed.loc[1, "qc_reason"], "touches_border")
        self.assertEqual(indexed.loc[2, "qc_reason"], "within_2px_image_edge")
        self.assertEqual(indexed.index[indexed.qc_keep].tolist(), [3, 4])

    def test_touching_label_separation_matches_historical_rule(self):
        mask = np.array(
            [[0, 0, 0, 0], [0, 1, 1, 2], [0, 1, 1, 2], [0, 0, 0, 0]],
            dtype=np.int32,
        )
        separated = separate_touching_labels(mask)
        self.assertTrue(np.all(separated[:, 2:] == 0))
        self.assertEqual(separated[1, 1], 1)

    def test_yap_adapter_matches_shared_kernel_and_keeps_project_metadata(self):
        from ekin_dapi_yap import model_a_pipeline as pipeline

        item = pipeline.YAPImageSet(
            "Ctrl", "5", 5000, "40x", 1, "field", Path("."),
            Path("dapi.png"), Path("yap.png"), Path("merge.png"),
        )
        with patch.object(pipeline, "load_gray", return_value=self.image), patch.object(
            pipeline.np, "load", return_value=self.mask
        ):
            active = pipeline.extract_dapi_features(item, Path("mask.npy"), edge_buffer_px=2)
        metadata = {
            "experimental_group_label": "Ctrl",
            **pipeline.EXPERIMENTAL_GROUP_METADATA["Ctrl"],
            "seeding_density_folder_label": "5",
            "seeding_density_cells_per_cm2": 5000,
            "replicate": "not_provided",
            "magnification": "40x",
            "imaging_condition": "magnification_40x",
            "export_index": 1,
            "image_set_folder": ".",
            "dapi_org_path": "dapi.png",
            "yap_af488_org_path": "yap.png",
            "merge_image_path": "merge.png",
        }
        expected = extract_morphology_2d(
            self.mask,
            self.image,
            image_id="field",
            edge_buffer_px=2,
            metadata=metadata,
        )
        kept = expected.loc[expected.qc_keep].reset_index(drop=True)
        kept = add_composite_features(add_spatial_features(kept))
        for column in kept.columns.difference(expected.columns):
            expected[column] = expected["label"].map(kept.set_index("label")[column])
        pd.testing.assert_frame_equal(
            expected.sort_values("label").reset_index(drop=True),
            active.sort_values("label").reset_index(drop=True),
            check_dtype=False,
            check_like=True,
        )

    def test_spatial_and_composite_features_are_deterministic_and_dimensionless(self):
        base = extract_morphology_2d(self.mask, self.image, image_id="field", edge_buffer_px=2)
        base = base.loc[base.qc_keep].reset_index(drop=True)
        spatial = add_spatial_features(base, k_neighbors=6)
        repeated = add_spatial_features(base, k_neighbors=6)
        pd.testing.assert_frame_equal(spatial, repeated)
        self.assertTrue((spatial["adaptive_neighbor_count"] == 1).all())
        self.assertTrue((spatial["nb_area_mean_px2"] == 25).all())
        composite = add_composite_features(spatial)
        np.testing.assert_allclose(composite["nuclear_size_log_ratio"], 0.0)
        self.assertTrue(np.isfinite(composite["nearest_spacing_nuclear_units"]).all())

        custom_index = base.copy()
        custom_index.index = [10, 20]
        indexed = add_spatial_features(custom_index, k_neighbors=6)
        self.assertEqual(indexed.index.tolist(), [10, 20])
        np.testing.assert_allclose(
            indexed["nn1_distance_px"].to_numpy(),
            spatial["nn1_distance_px"].to_numpy(),
        )
        with self.assertRaises(ValueError):
            add_spatial_features(base, k_neighbors=0)


if __name__ == "__main__":
    unittest.main()


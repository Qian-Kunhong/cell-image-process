import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Ekin_DAPI_YAP"))
from shared.entrypoint import configured_args
from shared import dapi_model_a as core
import yap_40x_trial as yap


class EntryTests(unittest.TestCase):
    def test_modes_and_magnifications_have_separate_outputs(self):
        for parser in (core.parse_args, yap.parse_args):
            paths = set()
            for mode in ("baseline", "composite"):
                for mag in ("20x", "40x"):
                    args = configured_args(parser, ROOT, mode, mag, argv=[])
                    self.assertEqual(args.feature_set, mode)
                    self.assertEqual(args.fit_magnification, mag)
                    paths.add(args.output_root)
            self.assertEqual(len(paths), 4)

    def test_explicit_paths_and_magnification(self):
        args = configured_args(yap.parse_args, ROOT, "baseline", argv=[
            "--output-root", "chosen", "--fit-magnification=20x"])
        self.assertEqual(args.output_root, Path("chosen"))
        self.assertEqual(args.fit_magnification, "20x")

    def test_oct4_samples_have_separate_outputs(self):
        a = configured_args(core.parse_args, ROOT, "baseline", argv=["--sample", "Sample 1"])
        b = configured_args(core.parse_args, ROOT, "baseline", argv=["--sample", "Sample 2"])
        self.assertNotEqual(a.output_root, b.output_root)

    def test_conflicting_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            configured_args(yap.parse_args, ROOT, "baseline", argv=["--feature-set", "composite"])

    def test_model_inputs_really_differ(self):
        import numpy as np
        import pandas as pd
        frame = pd.DataFrame({name: np.arange(20, dtype=float) + i
                              for i, name in enumerate(core.MODEL_FEATURES + core.COMPOSITE_MODEL_FEATURES)})
        _, baseline, _ = core.preprocess_features(frame, "raw")
        _, composite, _ = core.preprocess_features(frame, "augmented")
        self.assertFalse(set(baseline) & set(core.COMPOSITE_MODEL_FEATURES))
        self.assertTrue(set(core.COMPOSITE_MODEL_FEATURES) <= set(composite))


if __name__ == "__main__":
    unittest.main()

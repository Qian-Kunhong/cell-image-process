"""测试群体特征聚合：只测纯函数，不碰真实数据。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rp.population import summarize_population  # noqa: E402


def make_fake_cells() -> pd.DataFrame:
    """两个群体，各 100 个细胞；第二组分布更宽（用来验证离散度能区分）。"""
    rng = np.random.default_rng(0)
    a = rng.normal(10.0, 1.0, 100)
    b = rng.normal(10.0, 3.0, 100)
    return pd.DataFrame(
        {
            "well_id": ["A01"] * 100 + ["B02"] * 100,
            "area_um2": np.concatenate([a, b]),
            "eccentricity": np.concatenate([rng.uniform(0, 1, 100), rng.uniform(0, 1, 100)]),
            "cluster": [0] * 80 + [1] * 20 + [0] * 50 + [1] * 50,
            "file": ["x"] * 200,  # 非数值列应被自动排除
        }
    )


class TestSummarizePopulation(unittest.TestCase):
    def setUp(self) -> None:
        self.cells = make_fake_cells()
        self.pop = summarize_population(self.cells, group_col="well_id", cluster_col="cluster")

    def test_one_row_per_group(self) -> None:
        self.assertEqual(len(self.pop), 2)
        self.assertEqual(list(self.pop["group_id"]), ["A01", "B02"])
        self.assertEqual(list(self.pop["n_cells"]), [100, 100])

    def test_auto_feature_selection_skips_text_columns(self) -> None:
        self.assertNotIn("file__mean", self.pop.columns)
        self.assertIn("area_um2__mean", self.pop.columns)
        self.assertIn("eccentricity__mean", self.pop.columns)

    def test_distribution_shape_is_kept(self) -> None:
        # 均值相同但方差不同的两组，必须被 std / iqr 区分出来
        self.assertAlmostEqual(
            self.pop.loc[0, "area_um2__mean"], self.pop.loc[1, "area_um2__mean"], delta=1.0
        )
        self.assertGreater(self.pop.loc[1, "area_um2__std"], self.pop.loc[0, "area_um2__std"] * 2)
        self.assertGreater(self.pop.loc[1, "area_um2__iqr"], self.pop.loc[0, "area_um2__iqr"])

    def test_quantiles_are_ordered(self) -> None:
        row = self.pop.loc[0]
        keys = ["area_um2__q10", "area_um2__q25", "area_um2__q50", "area_um2__q75", "area_um2__q90"]
        values = [row[k] for k in keys]
        self.assertEqual(values, sorted(values))

    def test_cluster_share_sums_to_one(self) -> None:
        shares = [c for c in self.pop.columns if c.startswith("cluster__share_")]
        self.assertEqual(len(shares), 2)
        for _, row in self.pop.iterrows():
            self.assertAlmostEqual(sum(row[c] for c in shares), 1.0, places=6)

    def test_entropy_direction(self) -> None:
        # A01 是 80/20，B02 是 50/50，后者熵更高
        self.assertGreater(self.pop.loc[1, "cluster__entropy"], self.pop.loc[0, "cluster__entropy"])

    def test_each_feature_has_own_valid_count(self) -> None:
        # 缺失值会让有效样本数下降，这个数必须跟着特征走，不能全表共用一个 n
        cells = self.cells.copy()
        cells.loc[cells.index[:40], "area_um2"] = np.nan
        out = summarize_population(cells, group_col="well_id")
        self.assertEqual(out.loc[0, "area_um2__n"], 60)
        self.assertEqual(out.loc[0, "n_cells"], 100)
        self.assertEqual(out.loc[0, "eccentricity__n"], 100)

    def test_nan_feature_does_not_crash(self) -> None:
        cells = self.cells.copy()
        cells.loc[cells.index[:40], "area_um2"] = np.nan
        out = summarize_population(cells, group_col="well_id")
        self.assertTrue(np.isfinite(out.loc[0, "area_um2__mean"]))

    def test_missing_group_col_raises(self) -> None:
        with self.assertRaises(KeyError):
            summarize_population(self.cells, group_col="plate_id")


if __name__ == "__main__":
    unittest.main()

"""把细胞级特征表压成群体级（孔/视野）特征表。

用法：
    python scripts/01_build_population_features.py --input cells.csv --group-col well_id

输入：一行一个细胞的 CSV，至少含群体列（默认 well_id）与若干数值特征列。
输出：analysis_outputs/01_population/<时间戳>/population_features.csv + run_info.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rp.paths import make_run_dir, save_run_info  # noqa: E402
from rp.population import DEFAULT_QUANTILES, summarize_population  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="细胞级特征 -> 群体级特征")
    p.add_argument("--input", required=True, help="细胞级 CSV 路径")
    p.add_argument("--group-col", default="well_id", help="群体键列名，默认 well_id")
    p.add_argument("--feature", nargs="*", default=None, help="要汇总的特征列；省略则自动取全部数值列")
    p.add_argument("--cluster-col", default=None, help="无监督簇标签列；给出则输出各簇占比与熵")
    p.add_argument("--encoding", default="utf-8-sig", help="读 CSV 的编码；GBK 表用 gbk")
    p.add_argument("--tag", default="", help="输出目录后缀，便于区分不同数据集")
    p.add_argument("--out-dir", default=None, help="覆盖输出根目录（默认取 config/project.json）")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    src = Path(args.input)
    if not src.exists():
        print(f"[错误] 输入文件不存在：{src}", file=sys.stderr)
        return 2

    cells = pd.read_csv(src, encoding=args.encoding)
    if args.group_col not in cells.columns:
        print(
            f"[错误] 输入表里没有群体列 {args.group_col!r}；现有列：{list(cells.columns)[:12]}",
            file=sys.stderr,
        )
        return 2

    pop = summarize_population(
        cells,
        group_col=args.group_col,
        feature_cols=args.feature,
        cluster_col=args.cluster_col,
        quantiles=DEFAULT_QUANTILES,
    )

    out_dir = make_run_dir("01_population", tag=args.tag, base=args.out_dir)
    out_csv = out_dir / "population_features.csv"
    pop.to_csv(out_csv, index=False, encoding="utf-8-sig")
    save_run_info(
        out_dir,
        params={
            "input": str(src),
            "group_col": args.group_col,
            "cluster_col": args.cluster_col,
            "n_feature_cols": len([c for c in pop.columns if "__" in c]),
            "encoding": args.encoding,
        },
        extra={"n_groups": int(len(pop)), "n_cells": int(len(cells))},
    )

    print(f"群体数：{len(pop)}   细胞数：{len(cells)}   输出列：{pop.shape[1]}")
    print(f"已写出：{out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

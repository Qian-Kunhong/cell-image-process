"""把旧仓库的逐核指标表收拢成一张细胞级表，并核对核数口径。

这一步是新旧之间唯一的接口：旧仓库（只读）产出 per-sample 的 nucleus_3d_metrics.csv，
本脚本把它们拼成一行一个细胞的 cells.csv，供 01 聚合到群体级。

同时输出核数对照表：逐核表行数 vs 旧汇总表 n_nuclei，差多少、差在哪个样本都写清楚，
不允许用其中一套数字悄悄替掉另一套。

用法：
    python scripts/00_collect_legacy_nuclei.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rp.paths import legacy_path, make_run_dir, save_run_info  # noqa: E402

AUDIT_SUBDIR = ("analysis_outputs", "hepatocyte_confocal_audit", "dapi_3d", "yap")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="合并旧仓库逐核表 -> cells.csv")
    p.add_argument("--tag", default="legacy", help="输出目录后缀")
    p.add_argument("--out-dir", default=None, help="覆盖输出根")
    return p.parse_args(argv)


def collect(audit_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (细胞级表, 每个样本的核数对照表)。"""
    frames: list[pd.DataFrame] = []
    for sample_dir in sorted(p for p in audit_dir.iterdir() if p.is_dir()):
        csv = sample_dir / "nucleus_3d_metrics.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv, encoding="utf-8-sig")
        df.insert(0, "sample_code", sample_dir.name)
        df["source_file"] = str(csv.relative_to(audit_dir.parents[3]))
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"在 {audit_dir} 下没有找到任何 nucleus_3d_metrics.csv")

    cells = pd.concat(frames, ignore_index=True)

    # 与旧汇总表对齐培养日/方案（只 join 元数据，不动任何测量值）
    summary_csv = audit_dir / "analysis_summary" / "condition_3d_summary.csv"
    if summary_csv.exists():
        meta = pd.read_csv(summary_csv, encoding="utf-8-sig")[
            ["sample", "culture_day", "paper_day", "regimen"]
        ].rename(columns={"sample": "sample_code"})
        cells = cells.merge(meta, on="sample_code", how="left", validate="many_to_one")
        counts = (
            cells.groupby("sample_code")
            .size()
            .rename("n_rows_in_metrics")
            .reset_index()
            .merge(
                pd.read_csv(summary_csv, encoding="utf-8-sig")[
                    ["sample", "n_nuclei", "complete_objects"]
                ].rename(columns={"sample": "sample_code"}),
                on="sample_code",
                how="outer",
            )
        )
        counts["diff_rows_vs_n_nuclei"] = counts["n_rows_in_metrics"] - counts["n_nuclei"]
    else:
        counts = cells.groupby("sample_code").size().rename("n_rows_in_metrics").reset_index()
        counts["n_nuclei"] = pd.NA
        counts["diff_rows_vs_n_nuclei"] = pd.NA

    return cells, counts.sort_values("sample_code", ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audit_dir = legacy_path(*AUDIT_SUBDIR)
    print(f"读取旧仓：{audit_dir}")

    cells, counts = collect(audit_dir)

    out_dir = make_run_dir("00_cells", tag=args.tag, base=args.out_dir)
    cells_csv = out_dir / "cells.csv"
    counts_csv = out_dir / "nucleus_count_check.csv"
    cells.to_csv(cells_csv, index=False, encoding="utf-8-sig")
    counts.to_csv(counts_csv, index=False, encoding="utf-8-sig")

    total_rows = len(cells)
    total_summary = int(counts["n_nuclei"].fillna(0).sum()) if counts["n_nuclei"].notna().any() else None
    save_run_info(
        out_dir,
        params={"audit_dir": str(audit_dir), "samples": int(counts.shape[0])},
        extra={
            "n_cells": total_rows,
            "n_nuclei_in_legacy_summary": total_summary,
            "count_agreement": bool(total_summary == total_rows) if total_summary else None,
        },
    )

    print(f"细胞级表：{total_rows} 行 × {cells.shape[1]} 列 -> {cells_csv}")
    print(f"核数对照：{counts_csv}")
    if total_summary is not None and total_summary != total_rows:
        print(
            f"[口径差异] 逐核表 {total_rows} 行 vs 旧汇总 {total_summary} 核，"
            f"差 {total_summary - total_rows:+d}；逐样本差异见 nucleus_count_check.csv"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

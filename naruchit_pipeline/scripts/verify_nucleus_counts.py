"""核数一致性校验：掩膜标签数 vs 逐核表行数 vs 旧汇总表 n_nuclei。

三个数字本该相等。一旦不等，说明某个环节的产物被覆盖过或者用了不同版本的掩膜，
此时任何基于核数的结论都不能直接引用。

用法：
    python scripts/verify_nucleus_counts.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rp.paths import legacy_path, make_run_dir, save_run_info  # noqa: E402

AUDIT_SUBDIR = ("analysis_outputs", "hepatocyte_confocal_audit", "dapi_3d", "yap")


def count_masks_labels(npz_path: Path) -> int:
    with np.load(npz_path) as npz:
        labels = npz["labels"]
        n = int(np.unique(labels).size) - 1  # 0 是背景
    return n


def build_table(audit_dir: Path) -> pd.DataFrame:
    summary_csv = audit_dir / "analysis_summary" / "condition_3d_summary.csv"
    summary = pd.read_csv(summary_csv, encoding="utf-8-sig") if summary_csv.exists() else None

    rows = []
    for sample_dir in sorted(p for p in audit_dir.iterdir() if p.is_dir()):
        npz = sample_dir / "dapi_3d_masks.npz"
        metrics = sample_dir / "nucleus_3d_metrics.csv"
        if not npz.exists() and not metrics.exists():
            continue

        row = {
            "sample_code": sample_dir.name,
            "masks_labels": count_masks_labels(npz) if npz.exists() else pd.NA,
            "metrics_rows": sum(1 for _ in open(metrics, encoding="utf-8-sig")) - 1 if metrics.exists() else pd.NA,
        }
        if summary is not None and "sample" in summary.columns:
            matched = summary[summary["sample"].astype(str) == sample_dir.name]
            row["summary_n_nuclei"] = int(matched["n_nuclei"].iloc[0]) if len(matched) else pd.NA
            row["summary_complete_objects"] = int(matched["complete_objects"].iloc[0]) if len(matched) else pd.NA
        rows.append(row)

    table = pd.DataFrame(rows)
    ok = (
        table["masks_labels"].notna()
        & table["metrics_rows"].notna()
        & (table["masks_labels"] == table["metrics_rows"])
    )
    table["masks_eq_metrics"] = ok
    if "summary_n_nuclei" in table.columns:
        table["summary_matches_masks"] = table["summary_n_nuclei"].notna() & (
            table["summary_n_nuclei"] == table["masks_labels"]
        )
        table["diff_summary_minus_masks"] = table["summary_n_nuclei"] - table["masks_labels"]
    return table


def main(argv: list[str] | None = None) -> int:
    args = argparse.ArgumentParser(description="核数一致性校验").parse_args(argv)
    audit_dir = legacy_path(*AUDIT_SUBDIR)
    table = build_table(audit_dir)

    out_dir = make_run_dir("verify")
    out_csv = out_dir / "nucleus_consistency.csv"
    table.to_csv(out_csv, index=False, encoding="utf-8-sig")

    total_masks = int(table["masks_labels"].sum())
    total_metrics = int(table["metrics_rows"].sum())
    mismatch_summary = (
        int((~table["summary_matches_masks"]).sum()) if "summary_matches_masks" in table.columns else 0
    )
    save_run_info(
        out_dir,
        params={"audit_dir": str(audit_dir), "samples": int(len(table))},
        extra={
            "total_masks_labels": total_masks,
            "total_metrics_rows": total_metrics,
            "masks_vs_metrics_all_match": bool(table["masks_eq_metrics"].all()),
            "samples_where_summary_differs": mismatch_summary,
        },
    )

    pd.set_option("display.width", 200)
    print(table.to_string(index=False))
    print()
    print(f"掩膜标签数合计 = {total_masks}")
    print(f"逐核表行数合计 = {total_metrics}")
    print(f"掩膜 vs 逐核表：{'全部一致' if table['masks_eq_metrics'].all() else '存在不一致'}")
    if "summary_matches_masks" in table.columns:
        print(f"旧汇总表与掩膜不一致的样本数 = {mismatch_summary} / {len(table)}")
        print(
            f"旧汇总表合计 = {int(table['summary_n_nuclei'].sum())}，"
            f"与掩膜差 {int(table['summary_n_nuclei'].sum()) - total_masks:+d}"
        )
    print(f"已写出：{out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

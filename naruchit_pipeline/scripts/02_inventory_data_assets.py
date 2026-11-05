"""盘点数据根的可用资产：文件级清单 + 单位级计数。

这一步不读像素、不做测量，只回答三个问题：
1. 数据根下每一类文件分别有多少、多大、在哪；
2. 每类文件的统计单位是什么（条件/视野/孔/核），实际有几个；
3. 哪些目录只是论文产物（图、稿件），不属于可分析原始数据。

输出（analysis_outputs/02_inventory/<时间戳>/）：
- files.csv    一行一个文件：category, rel_path, suffix, size_bytes
- summary.csv  一行一个类别：n_files, size_mb, example_path
- units.csv    一行一个可统计单位组：dataset, level1..3, n_items, unit, note

用法：
    python scripts/02_inventory_data_assets.py
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rp.paths import data_path, make_run_dir, save_run_info  # noqa: E402

# 路径前缀（小写、正斜杠）-> 类别。长前缀优先匹配，避免被短前缀抢走。
CATEGORY_RULES: list[tuple[str, str]] = [
    ("03_raw data/22.07_analysis", "endpoint_summary_workbook"),
    ("03_raw data/immunostaining/confocal - yap/", "confocal_YAP_3D"),
    ("03_raw data/immunostaining/confocal - e-cad f-actin/", "confocal_EcadFactin_3D"),
    ("03_raw data/immunostaining/yap quantification/references/", "reference_literature"),
    ("03_raw data/immunostaining/yap quantification/", "yap_quantification"),
    ("03_raw data/immunostaining/in cell/", "incell_if"),
    ("03_raw data/biostudio/differentiation/", "biostudio_differentiation_timelapse"),
    ("03_raw data/biostudio/expansion/", "biostudio_expansion_timelapse"),
    ("03_raw data/phase contrast/differentiation/", "phase_contrast_differentiation"),
    ("03_raw data/phase contrast/expansion/", "phase_contrast_expansion"),
    ("03_raw data/flow cytometry/differentiation_afp-fitc, alb-pe/", "flow_afp_alb"),
    ("03_raw data/flow cytometry/expansion/", "flow_expansion"),
    ("03_raw data/plate reader/albumin secretion/", "plate_albumin"),
    ("03_raw data/plate reader/urea secretion/", "plate_urea"),
    ("03_raw data/pcr/hepatocyte differentiation/", "pcr_hepatocyte"),
    ("03_raw data/pcr/trilineage diff potential/", "pcr_trilineage"),
    ("03_raw data/western blot/", "western_blot"),
    ("03_raw data/cell growth/", "cell_growth"),
    ("01_summary slide/", "paper_summary_slide"),
    ("02_submission documents/", "paper_manuscript_and_figures"),
    ("04_images/", "paper_figure_exports"),
]

DOSE_RE = re.compile(r"HA(00|01|05|10)")


def classify(rel: str) -> str:
    low = rel.lower()
    for prefix, cat in CATEGORY_RULES:
        if low.startswith(prefix):
            return cat
    if "/" not in rel and low.endswith(".pdf"):
        return "paper_published_pdf"
    return "unclassified"


def walk(root: Path) -> pd.DataFrame:
    rows = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        rows.append(
            {
                "category": classify(rel),
                "rel_path": rel,
                "suffix": p.suffix.lower(),
                "size_bytes": p.stat().st_size,
            }
        )
    return pd.DataFrame(rows).sort_values(["category", "rel_path"], ignore_index=True)


def summarize_files(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("category")
        .agg(n_files=("rel_path", "size"), size_mb=("size_bytes", lambda s: round(s.sum() / 2**20, 1)))
        .reset_index()
        .sort_values("n_files", ascending=False, ignore_index=True)
    )
    first = df.groupby("category")["rel_path"].first().rename("example_path")
    return out.merge(first, on="category")


def units_confocal(root: Path) -> list[dict]:
    """共聚焦面板：一行一个样本（OIB = 一个视野）。"""
    rows: list[dict] = []
    for cat, sub in (
        ("confocal_YAP_3D", ("03_Raw data", "Immunostaining", "Confocal - YAP")),
        ("confocal_EcadFactin_3D", ("03_Raw data", "Immunostaining", "Confocal - E-cad F-actin")),
    ):
        d = root.joinpath(*sub)
        for f in sorted(d.glob("*.oib")):
            rows.append(
                dict(dataset=cat, level1=f.stem, level2="", level3="",
                     n_items=1, unit="oib(1 视野)", note="每条件 1 个视野，无孔号")
            )
    return rows


def units_yap_quant(root: Path) -> list[dict]:
    """CellProfiler YAP 定量：一行一个条件的核数。"""
    rows: list[dict] = []
    base = root.joinpath("03_Raw data", "Immunostaining", "YAP quantification")
    for sample_dir in sorted(p for p in base.iterdir() if p.is_dir() and p.name != "References"):
        n_slices = len(list(sample_dir.glob("*.tif")))
        ana_dirs = sorted(p for p in sample_dir.iterdir() if p.is_dir())
        for ana in ana_dirs:
            nuclei_csv = next(ana.glob("*_Nuclei.csv"), None)
            if nuclei_csv is None:
                rows.append(dict(dataset="yap_quantification", level1=sample_dir.name, level2=ana.name,
                                 level3="", n_items=0, unit="nuclei(缺失 Nuclei.csv)",
                                 note=f"{n_slices} 张切片"))
                continue
            df = pd.read_csv(nuclei_csv, encoding="utf-8-sig")
            n_nuc = len(df)
            n_img = int(df["ImageNumber"].nunique()) if "ImageNumber" in df else 0
            rows.append(
                dict(dataset="yap_quantification", level1=sample_dir.name, level2=ana.name, level3="",
                     n_items=n_nuc, unit="nuclei(CellProfiler 4.0.7)",
                     note=f"{n_slices} 张 2D 切片 / {n_img} 张图")
            )
    return rows


def units_incell(root: Path) -> list[dict]:
    """IN Cell 面板：一行一个 (面板, 倍率, 天数, 剂量) 组合，数通道文件。"""
    base = root.joinpath("03_Raw data", "Immunostaining", "IN Cell")
    acc: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for f in base.rglob("*.tif"):
        parts = f.relative_to(base).parts
        panel = parts[0]
        mag = parts[1] if len(parts) > 1 else ""
        day = ""
        if mag.lower() in {"4x", "10x"}:
            if len(parts) > 3:
                day = parts[2]
        else:
            mag = ""
            day = parts[1] if len(parts) > 2 else ""
        m = DOSE_RE.search(f.stem.upper())
        dose = f"HA{m.group(1)}" if m else "NA"
        acc[(panel, mag, day, dose)] += 1
    rows = [
        dict(dataset="incell_if", level1=panel, level2=f"{mag}/{day}".strip("/"), level3=dose,
             n_items=n, unit="tif(通道文件)", note="DAPI 通道可做核分割")
        for (panel, mag, day, dose), n in sorted(acc.items())
    ]
    return rows


def units_flow(root: Path) -> list[dict]:
    """流式：一行一个 (面板, 组) 的孔数。组名取文件名里 '.' 之前的标识。"""
    base = root.joinpath("03_Raw data", "Flow cytometry")
    rows: list[dict] = []
    for f in sorted(base.rglob("*.fcs")):
        parts = f.relative_to(base).parts
        panel = parts[0] + ("/" + parts[1] if len(parts) > 2 else "")
        stem = f.stem
        bits = stem.split(".")
        group = ".".join(bits[:-1]) if len(bits) > 1 and bits[-1].isdigit() else stem
        rows.append(dict(dataset="flow", level1=panel, level2=group, level3="", n_items=1,
                         unit="fcs(孔)", note=""))
    df = pd.DataFrame(rows)
    agg = (
        df.groupby(["dataset", "level1", "level2"], as_index=False)
        .size()
        .rename(columns={"size": "n_items"})
    )
    agg["unit"] = "fcs(孔)"
    agg["level3"] = ""
    agg["note"] = agg["level2"].str.upper().str.startswith("ISO").map({True: "同型对照", False: ""})
    return agg.to_dict("records")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="盘点数据资产")
    ap.add_argument("--tag", default="naruchit", help="输出目录后缀")
    ap.add_argument("--out-dir", default=None, help="覆盖输出根")
    args = ap.parse_args(argv)

    root = data_path()
    print(f"扫描数据根：{root}")

    files = walk(root)
    summary = summarize_files(files)
    units = units_confocal(root) + units_yap_quant(root) + units_incell(root) + units_flow(root)
    units_df = pd.DataFrame(units)

    out_dir = make_run_dir("02_inventory", tag=args.tag, base=args.out_dir)
    files.to_csv(out_dir / "files.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    units_df.to_csv(out_dir / "units.csv", index=False, encoding="utf-8-sig")

    save_run_info(
        out_dir,
        params={"data_root": str(root), "n_rules": len(CATEGORY_RULES)},
        extra={
            "n_files": int(len(files)),
            "n_categories": int(summary.shape[0]),
            "unclassified": int((files["category"] == "unclassified").sum()),
        },
    )

    print(f"\n文件总数 {len(files)}，类别 {summary.shape[0]} 个：")
    for _, r in summary.iterrows():
        print(f"  {r['category']:<34} {r['n_files']:>5} 个  {r['size_mb']:>10.1f} MB")
    if (files["category"] == "unclassified").any():
        print("\n[未分类]")
        for rel in files.loc[files["category"] == "unclassified", "rel_path"].head(20):
            print("  " + rel)
    print(f"\n产物目录：{out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

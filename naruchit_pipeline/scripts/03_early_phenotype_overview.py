"""03 早期核表型总览：973 核形态 + 2821 核 YAP 按条件分组画箱线图。

输入：
- 旧仓 3D 逐核表：analysis_outputs/hepatocyte_confocal_audit/dapi_3d/yap/<sample>/nucleus_3d_metrics.csv
- 旧仓条件汇总（含 culture_day/regimen 解码）：.../analysis_summary/condition_3d_summary.csv
- CellProfiler 2D YAP 表：F:\\...\\YAP quantification\\<sample>\\Ana_<sample>2\\<sample>_Nuclei.csv

输出：
- analysis_outputs/03_early_phenotype/<ts>/early_phenotype_overview.png
- early_phenotype_per_condition.csv（每条件的中位数/四分位）
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib


def main() -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # 中文字体
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # --- 路径 ---
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from rp.paths import data_path, legacy_path  # noqa: E402

    YAP_ROOT_3D = legacy_path("analysis_outputs", "hepatocyte_confocal_audit", "dapi_3d", "yap")
    COND_CSV = YAP_ROOT_3D / "analysis_summary" / "condition_3d_summary.csv"
    YAP_ROOT_2D = data_path("03_Raw data", "Immunostaining", "YAP quantification")

    OUT_DIR = PROJECT_ROOT / "analysis_outputs" / "03_early_phenotype"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. 读条件解码 ---
    cond = pd.read_csv(COND_CSV, encoding="utf-8-sig")
    cond = cond[["sample", "culture_day", "paper_day", "regimen", "n_nuclei"]].copy()
    regimen_cn = {
        "shared_baseline": "基线",
        "no_HA": "不暴露",
        "single_HA": "单次",
        "repeated_HA": "每日",
    }
    cond["regimen_cn"] = cond["regimen"].map(regimen_cn)
    print(cond.to_string(), file=sys.stderr)

    # --- 2. 读 973 核形态（逐核） ---
    morph_frames = []
    for sample in cond["sample"]:
        f = YAP_ROOT_3D / sample / "nucleus_3d_metrics.csv"
        if not f.exists():
            print(f"[skip] {f}", file=sys.stderr)
            continue
        df = pd.read_csv(f, encoding="utf-8-sig")
        df = df[["sample", "label", "equivalent_sphere_diameter_um", "volume_um3",
                "pca_elongation_1_over_3"]].copy()
        morph_frames.append(df)
    morph = pd.concat(morph_frames, ignore_index=True)
    morph = morph.merge(cond, on="sample", how="left")
    print(f"morph rows: {len(morph)}", file=sys.stderr)

    # --- 3. 读 2821 核 YAP（逐核） ---
    yap_frames = []
    for sample in cond["sample"]:
        sample_dir = YAP_ROOT_2D / sample
        if not sample_dir.exists():
            print(f"[skip] {sample_dir}", file=sys.stderr)
            continue
        # Ana_<sample>2 下的 <sample>_Nuclei.csv
        ana = list(sample_dir.glob(f"Ana_*/{sample}_Nuclei.csv"))
        if not ana:
            print(f"[skip] no Ana Nuclei.csv for {sample}", file=sys.stderr)
            continue
        df = pd.read_csv(ana[0], encoding="utf-8-sig")
        df["sample"] = sample
        df = df[["sample", "ImageNumber", "Math_NucByRingYAP"]].copy()
        yap_frames.append(df)
    yap = pd.concat(yap_frames, ignore_index=True)
    yap = yap.merge(cond, on="sample", how="left")
    print(f"yap rows: {len(yap)}", file=sys.stderr)

    # --- 4. 条件级汇总（写 CSV） ---
    def q(x, p):
        return x.quantile(p) if len(x) else np.nan

    rows = []
    for (paper_day, regimen_cn, sample), g in morph.groupby(["paper_day", "regimen_cn", "sample"]):
        rows.append({
            "sample": sample,
            "paper_day": paper_day,
            "regimen": regimen_cn,
            "n_morph": len(g),
            "diameter_median": g["equivalent_sphere_diameter_um"].median(),
            "diameter_q25": q(g["equivalent_sphere_diameter_um"], 0.25),
            "diameter_q75": q(g["equivalent_sphere_diameter_um"], 0.75),
        })
    for (paper_day, regimen_cn, sample), g in yap.groupby(["paper_day", "regimen_cn", "sample"]):
        # merge into existing row by sample
        for r in rows:
            if r["sample"] == sample:
                r["n_yap"] = len(g)
                r["yap_ratio_median"] = g["Math_NucByRingYAP"].median()
                r["yap_ratio_q25"] = q(g["Math_NucByRingYAP"], 0.25)
                r["yap_ratio_q75"] = q(g["Math_NucByRingYAP"], 0.75)
    summary = pd.DataFrame(rows).sort_values(["paper_day", "regimen"])
    summary_path = OUT_DIR / "early_phenotype_per_condition.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(summary.to_string(), file=sys.stderr)

    # --- 5. 画图：1 行 2 列 ---
    order_day = [-3, -2, -1, 0]
    order_regimen = ["基线", "不暴露", "单次", "每日"]
    colors = {"基线": "#999999", "不暴露": "#4C72B0", "单次": "#DD8452", "每日": "#C44E52"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

    def boxplot_panel(ax, df, value_col, ylabel, title):
        # x 位置：每个 day 一组，组内 3-4 个 box
        positions = []
        labels = []
        box_data = []
        box_colors = []
        pos = 0
        for day in order_day:
            day_df = df[df["paper_day"] == day]
            for reg in order_regimen:
                sub = day_df[day_df["regimen_cn"] == reg][value_col].dropna().values
                if len(sub) == 0:
                    pos += 1
                    continue
                positions.append(pos)
                labels.append(reg)
                box_data.append(sub)
                box_colors.append(colors[reg])
                pos += 1
            pos += 0.6  # day 之间间隔
        bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                        patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        for k in ("whiskers", "caps", "medians"):
            for line in bp[k]:
                line.set_color("black")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    boxplot_panel(axes[0], morph, "equivalent_sphere_diameter_um",
                 "等效球直径 (µm)", "核形态：直径")
    boxplot_panel(axes[1], yap, "Math_NucByRingYAP",
                  "YAP 核/环 强度比", "YAP 核质比")

    # day 分隔线 + 标注
    for ax in axes:
        # 在 day 之间加竖线（粗略）
        ax.axvline(x=3.8, color="gray", ls="--", alpha=0.4)
        ax.axvline(x=7.4, color="gray", ls="--", alpha=0.4)
        ax.axvline(x=11.0, color="gray", ls="--", alpha=0.4)
        # day 标签放在组上方
        ax.text(1.5, ax.get_ylim()[1], "day -3\n(基线)", ha="center", va="top", fontsize=8)
        ax.text(5.5, ax.get_ylim()[1], "day -2", ha="center", va="top", fontsize=8)
        ax.text(9.1, ax.get_ylim()[1], "day -1", ha="center", va="top", fontsize=8)
        ax.text(12.5, ax.get_ylim()[1], "day 0", ha="center", va="top", fontsize=8)

    fig.suptitle("扩增期 HA 处理后早期核表型（973 核形态 + 2821 核 YAP）", fontsize=11)
    fig.tight_layout()
    png_path = OUT_DIR / "early_phenotype_overview.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    print(f"saved: {png_path}", file=sys.stderr)
    print(f"saved: {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

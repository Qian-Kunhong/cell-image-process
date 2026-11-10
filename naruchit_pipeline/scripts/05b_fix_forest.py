"""05b 修正森林图：单视野条件之间的描述性效应量。"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib


def main() -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from rp.paths import legacy_path  # noqa: E402

    YAP_ROOT_3D = legacy_path("analysis_outputs", "hepatocyte_confocal_audit", "dapi_3d", "yap")
    COND_CSV = YAP_ROOT_3D / "analysis_summary" / "condition_3d_summary.csv"
    OUT_DIR = PROJECT_ROOT / "analysis_outputs" / "05_replicate_ekin"

    # --- 加载数据 ---
    cond = pd.read_csv(COND_CSV, encoding="utf-8-sig")
    cond = cond[["sample", "culture_day", "paper_day", "regimen"]].copy()
    regimen_cn = {"shared_baseline": "基线", "no_HA": "不暴露", "single_HA": "单次", "repeated_HA": "每日"}
    cond["regimen_cn"] = cond["regimen"].map(regimen_cn)

    FEATURES = [
        "volume_um3", "height_um",
        "bbox_y_um", "bbox_x_um",
        "pca_axis_1_um", "pca_axis_2_um", "pca_axis_3_um",
        "pca_elongation_1_over_3",
        "max_xy_area_um2",
        "equivalent_sphere_diameter_um",
    ]
    frames = []
    for sample in cond["sample"]:
        f = YAP_ROOT_3D / sample / "nucleus_3d_metrics.csv"
        if not f.exists():
            continue
        frames.append(pd.read_csv(f, encoding="utf-8-sig"))
    morph = pd.concat(frames, ignore_index=True).merge(cond, on="sample", how="left")

    # 缺测行丢弃，不把缺测值填成形态学上的 0。
    morph = morph.dropna(subset=FEATURES).reset_index(drop=True)

    # --- 图 3 修正：森林图（菱形中位数） ---
    features_to_plot = [
        ("equivalent_sphere_diameter_um", "Nuclear diameter"),
        ("volume_um3", "Nuclear volume"),
        ("pca_elongation_1_over_3", "Elongation"),
        ("height_um", "Height (z)"),
        ("max_xy_area_um2", "Max XY area"),
    ]

    day0 = morph[morph["paper_day"] == 0]
    ctrl = day0[day0["regimen_cn"] == "不暴露"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    y_pos = np.arange(len(features_to_plot))

    for i, (col, label) in enumerate(features_to_plot):
        ctrl_median = ctrl[col].median()
        ctrl_q25 = ctrl[col].quantile(0.25)
        ctrl_q75 = ctrl[col].quantile(0.75)
        ctrl_iqr = ctrl_q75 - ctrl_q25
        if ctrl_iqr == 0:
            continue

        for reg, color, offset in [("单次", "#C44E52", 0.15), ("每日", "#8172B3", -0.15)]:
            ha = day0[day0["regimen_cn"] == reg]
            ha_med = ha[col].median()
            ha_q25 = ha[col].quantile(0.25)
            ha_q75 = ha[col].quantile(0.75)

            # 效应量 = (ha_med - ctrl_med) / ctrl_iqr
            eff_med = (ha_med - ctrl_median) / ctrl_iqr
            eff_low = (ha_q25 - ctrl_median) / ctrl_iqr
            eff_high = (ha_q75 - ctrl_median) / ctrl_iqr

            y = i + offset
            # 水平线 = IQR 范围
            ax.plot([eff_low, eff_high], [y, y], color=color, lw=1.5, alpha=0.6)
            # 菱形 = 中位数
            ax.scatter(eff_med, y, marker="D", s=80, color=color, zorder=5, edgecolors="white", linewidths=0.5)

    ax.axvline(x=0, color="black", lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f[1] for f in features_to_plot])
    ax.set_xlabel("HA effect relative to no-HA control (IQR units, day 0)")
    ax.set_title("Descriptive HA effect on nuclear morphology (day 0; one field per condition)")
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#C44E52', markersize=9, label='Single HA (day -3)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#8172B3', markersize=9, label='Repeated HA (day -3→0)'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ha_effect_forest_fixed.png", dpi=180, bbox_inches="tight")
    plt.close()

    print(f"Done. Figure saved to {OUT_DIR}", file=sys.stderr)


if __name__ == "__main__":
    main()

"""06 HA 效应森林图：日期颜色和 YAP 箱线图统一（蓝/绿/橙）。"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib


def main() -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from rp.paths import legacy_path  # noqa: E402

    YAP_ROOT_3D = legacy_path("analysis_outputs", "hepatocyte_confocal_audit", "dapi_3d", "yap")
    matched = pd.read_csv(
        YAP_ROOT_3D / "best_focus_2d_vs_3d" / "matched_best_focus_2d_to_3d_nuclei.csv",
        encoding="utf-8-sig",
    )
    centroid_frames = []
    for sample in matched["sample"].unique():
        f = YAP_ROOT_3D / sample / "nucleus_3d_metrics.csv"
        if f.exists():
            df = pd.read_csv(f, encoding="utf-8-sig")
            df = df[["sample", "label", "centroid_x_um", "centroid_y_um"]]
            centroid_frames.append(df)
    centroids = pd.concat(centroid_frames, ignore_index=True)
    matched = matched.merge(centroids, on=["sample", "label"], how="left")

    regimen_en = {"shared_baseline": "Baseline", "no_HA": "No HA",
                  "single_HA": "Single", "repeated_HA": "Repeated"}
    matched["group"] = matched["regimen"].map(regimen_en)

    def compute_neighbors(df):
        results = []
        for sample, sub in df.groupby("sample"):
            coords = sub[["centroid_x_um", "centroid_y_um"]].values
            areas = sub["area_um2"].values
            circ = sub["circularity"].values
            n = len(sub)
            nn_dist = np.full(n, np.nan)
            crowding = np.full(n, np.nan)
            nb_disagree = np.full(n, np.nan)
            for i in range(n):
                d = np.sqrt(np.sum((coords - coords[i])**2, axis=1))
                d[i] = np.inf
                nn_idx = np.argsort(d)[:6]
                nn_dist[i] = d[nn_idx[0]]
                crowding[i] = areas[nn_idx].sum() / (np.pi * nn_dist[i]**2 + 1e-6)
                nb_disagree[i] = abs(circ[i] - circ[nn_idx].mean())
            sub = sub.copy()
            sub["nn_distance"] = nn_dist
            sub["local_crowding"] = crowding
            sub["neighbor_disagree"] = nb_disagree
            results.append(sub)
        return pd.concat(results, ignore_index=True)

    matched = compute_neighbors(matched)

    FEATURES = {
        "area_um2": "Nuclear area",
        "aspect_ratio": "Elongation",
        "circularity": "Circularity",
        "nn_distance": "Nuclear spacing",
        "local_crowding": "Local crowding",
        "neighbor_disagree": "Neighbor heterogeneity",
    }

    time_points = [-2, -1, 0]
    rows = []
    for day in time_points:
        day_data = matched[matched["paper_day"] == day]
        ctrl = day_data[day_data["group"] == "No HA"]
        for feat, label in FEATURES.items():
            ctrl_vals = ctrl[feat].dropna()
            ctrl_med = ctrl_vals.median()
            ctrl_iqr = ctrl_vals.quantile(0.75) - ctrl_vals.quantile(0.25)
            if ctrl_iqr == 0:
                continue
            for grp in ["Single", "Repeated"]:
                ha_vals = day_data[day_data["group"] == grp][feat].dropna()
                effect = (ha_vals.median() - ctrl_med) / ctrl_iqr
                rows.append({"day": day, "feature": feat, "metric": label,
                             "group": grp, "effect": effect})
    eff_df = pd.DataFrame(rows)

    # === 日期颜色（和 YAP 箱线图统一）===
    DAY_COLORS = {-2: "#2166AC", -1: "#4DAF4A", 0: "#E8985E"}
    DAY_LABELS = {-2: "Day −2", -1: "Day −1", 0: "Day 0"}

    # 面板色：用灰色横线，不抢日期颜色
    GROUP_LINE = {"Single": "#888888", "Repeated": "#AAAAAA"}

    metrics = [FEATURES[f] for f in FEATURES]
    y = np.arange(len(metrics), dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.subplots_adjust(bottom=0.18, wspace=0.08)

    for ax, grp in zip(axes, ["Single", "Repeated"]):
        for row_i, metric in enumerate(metrics):
            sub = eff_df[(eff_df["group"] == grp) & (eff_df["metric"] == metric)].sort_values("day")
            vals = sub["effect"].values
            # 横线 min-max（灰色）
            ax.hlines(row_i, np.min(vals), np.max(vals), color=GROUP_LINE[grp],
                      lw=3, alpha=0.5, zorder=1)
            # 三个时间点：蓝/绿/橙
            offsets = np.linspace(-0.1, 0.1, len(sub))
            for j, (_, r) in enumerate(sub.iterrows()):
                ax.scatter(r["effect"], row_i + offsets[j], s=90,
                           color=DAY_COLORS[r["day"]], edgecolors="white",
                           linewidth=1.2, zorder=3)
        ax.axvline(0, color="#555555", lw=1.2, ls="--", alpha=0.7)
        ax.set_yticks(y, metrics, fontsize=12)
        ax.invert_yaxis()
        ax.set_title(f"{grp} HA", fontsize=15, fontweight="bold", pad=10)
        ax.set_xlabel("Effect size (IQR units)", fontsize=11, labelpad=8)
        ax.grid(axis="x", color="#EEEEEE", lw=1)

    day_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=10,
               markerfacecolor=DAY_COLORS[d], markeredgecolor="white",
               label=DAY_LABELS[d])
        for d in time_points
    ]
    fig.legend(handles=day_handles, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 0.01), fontsize=11)

    fig.suptitle("HA effect on nuclear morphology (868 matched nuclei)",
                 fontsize=15, fontweight="bold", y=0.98)
    out = PROJECT_ROOT / "analysis_outputs" / "05_replicate_ekin" / "ha_effect_forest.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"Saved: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()

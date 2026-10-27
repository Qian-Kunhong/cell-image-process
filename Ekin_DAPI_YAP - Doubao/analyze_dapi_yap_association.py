"""Relate continuous DAPI-derived features to post-hoc YAP localization.

The primary result is the median of within-image Spearman correlations.  This
keeps treatment, density and field-level acquisition differences from creating
a purely between-image association.  Pooled correlations are retained as a
descriptive contrast.  Cells are observations, not biological replicates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


CANVAS = (13.333, 7.5)
DPI = 220
MIN_CELLS_PER_FIELD = 20
YAP_COLUMNS = {
    "raw": "posthoc_yap_raw_log2_nuclear_perinuclear_ratio",
    "background_corrected": "posthoc_yap_log2_nuclear_perinuclear_ratio",
}

# Non-redundant axes used on the PPT overview.  The full model feature set is
# still analyzed and written to CSV.
DISPLAY_FEATURES = {
    "area_px": "Nuclear area",
    "aspect_ratio": "Elongation",
    "circularity": "Circularity",
    "solidity": "Solidity",
    "nn1_distance_px": "Nearest-neighbor distance",
    "knn6_distance_mean_px": "Mean 6-neighbor distance",
    "fixed_neighbor_count": "Fixed-radius neighbors",
    "local_density_per_px2": "Local nuclear density",
    "nb_area_std_px2": "Neighbor size variability",
    "nb_aspect_ratio_std": "Neighbor shape variability",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Completed analysis root containing run_info.json and tables/")
    parser.add_argument("--analysis-dir", type=Path,
                        help="Default: <output-root>/dapi_yap_association")
    return parser.parse_args(argv)


def _rho(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    pair = pd.concat([pd.to_numeric(x, errors="coerce"),
                      pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan, len(pair)
    return float(spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic), len(pair)


def load_feature_columns(root: Path, cells: pd.DataFrame) -> list[str]:
    info = json.loads((root / "run_info.json").read_text(encoding="utf-8"))
    columns = [c for c in info["model_feature_columns"] if c in cells.columns]
    forbidden = ("dapi_", "yap", "pca_", "umap_", "phenotype", "gmm")
    columns = [c for c in columns if not any(token in c.lower() for token in forbidden)]
    if not columns:
        raise ValueError("No eligible DAPI morphology features found")
    return columns


def calculate_associations(cells: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    field_rows: list[dict] = []
    for measurement, yap_column in YAP_COLUMNS.items():
        for feature in features:
            pooled_rho, pooled_n = _rho(cells[feature], cells[yap_column])
            rhos = []
            for image_id, field in cells.groupby("image_id", sort=True):
                rho, n = _rho(field[feature], field[yap_column])
                if n < MIN_CELLS_PER_FIELD or not np.isfinite(rho):
                    continue
                rhos.append(rho)
                field_rows.append({
                    "measurement": measurement, "feature": feature,
                    "image_id": image_id,
                    "experimental_group_label": field["experimental_group_label"].iloc[0],
                    "seeding_density_cells_per_cm2": field["seeding_density_cells_per_cm2"].iloc[0],
                    "n_cells": n, "spearman_rho": rho,
                })
            values = np.asarray(rhos, dtype=float)
            median = float(np.median(values)) if len(values) else np.nan
            summary_rows.append({
                "measurement": measurement, "feature": feature,
                "pooled_spearman_rho": pooled_rho, "pooled_n_cells": pooled_n,
                "within_field_median_rho": median,
                "within_field_q25_rho": float(np.quantile(values, .25)) if len(values) else np.nan,
                "within_field_q75_rho": float(np.quantile(values, .75)) if len(values) else np.nan,
                "n_fields": len(values),
                "direction_consistency_fraction": (
                    float(np.mean(np.sign(values) == np.sign(median)))
                    if len(values) and median != 0 else np.nan
                ),
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(field_rows)


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "figure.facecolor": "white",
        "axes.facecolor": "white", "axes.titlesize": 16,
        "axes.labelsize": 13, "xtick.labelsize": 10, "ytick.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def plot_overview(summary: pd.DataFrame, fields: pd.DataFrame, output: Path) -> None:
    _style()
    raw = summary[summary.measurement.eq("raw")].set_index("feature")
    names = [feature for feature in DISPLAY_FEATURES if feature in raw.index]
    ordered = sorted(names, key=lambda f: raw.loc[f, "within_field_median_rho"])
    labels = [DISPLAY_FEATURES[f] for f in ordered]
    y = np.arange(len(ordered))

    fig = plt.figure(figsize=CANVAS)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.55), left=.16, right=.96,
                           bottom=.29, top=.79, wspace=.30)
    ax = fig.add_subplot(grid[0, 0])
    med = raw.loc[ordered, "within_field_median_rho"].to_numpy(float)
    q25 = raw.loc[ordered, "within_field_q25_rho"].to_numpy(float)
    q75 = raw.loc[ordered, "within_field_q75_rho"].to_numpy(float)
    ax.hlines(y, q25, q75, color="#B3B3B3", linewidth=4, alpha=.9)
    ax.scatter(med, y, s=65, color="#4D4D4D", zorder=4)
    ax.axvline(0, color="#555555", linestyle=":", linewidth=1.2)
    ax.set_yticks(y, labels)
    ax.set_xlim(-.65, .65)
    ax.set_xlabel("Median within-field Spearman correlation with YAP\n"
                  "Line: interquartile range across 12 fields")
    ax.set_title("Median correlation across fields", fontweight="bold")
    ax.grid(axis="x", color="#E6E6E6", linewidth=.8)

    heat = fields[(fields.measurement.eq("raw")) & fields.feature.isin(ordered)].copy()
    matrix = heat.pivot(index="image_id", columns="feature", values="spearman_rho")
    matrix = matrix.reindex(columns=ordered)
    # Experimental order, not lexical image-id order (which puts 10k before 2.5k).
    group_order = {"Ctrl": 0, "HA1": 1, "HA2": 2}
    field_order = (heat[["image_id", "experimental_group_label",
                         "seeding_density_cells_per_cm2"]]
                   .drop_duplicates("image_id")
                   .assign(_group=lambda x: x.experimental_group_label.map(group_order),
                           _density=lambda x: pd.to_numeric(
                               x.seeding_density_cells_per_cm2, errors="coerce"))
                   .sort_values(["_group", "_density", "image_id"])
                   .image_id.tolist())
    matrix = matrix.reindex(field_order)
    field_metadata = (heat[["image_id", "experimental_group_label",
                            "seeding_density_cells_per_cm2"]]
                      .drop_duplicates("image_id").set_index("image_id"))
    field_labels = []
    for image_id in matrix.index:
        row = field_metadata.loc[image_id]
        density = float(row.seeding_density_cells_per_cm2) / 1000
        density_label = f"{density:g}k"
        field_labels.append(f"{row.experimental_group_label} · {density_label}")
    ax2 = fig.add_subplot(grid[0, 1])
    im = ax2.imshow(matrix, aspect="auto", cmap="PuOr_r",
                    norm=TwoSlopeNorm(vmin=-.7, vcenter=0, vmax=.7))
    ax2.set_xticks(np.arange(len(ordered)), [DISPLAY_FEATURES[f] for f in ordered],
                   rotation=35, ha="right", fontsize=8.5)
    ax2.set_yticks(np.arange(len(matrix)), field_labels)
    ax2.set_title("Correlation within each image field", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax2, fraction=.035, pad=.025)
    cbar.set_label("Spearman ρ")

    fig.suptitle("DAPI features associated with YAP localization",
                 fontsize=24, fontweight="bold", y=.94)
    fig.text(.5, .855,
             "Within-field analysis reduces confounding by HA group, seeding density and field-level acquisition",
             ha="center", fontsize=13, color="#444444")
    fig.text(.5, .035,
             "Raw log₂(nuclear/perinuclear YAP); descriptive single-cell association, not biological-replicate inference",
             ha="center", fontsize=10.5, color="#555555")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=DPI, facecolor="white")
    fig.savefig(output.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def plot_top_relationships(cells: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    raw = summary[summary.measurement.eq("raw")].copy()
    raw["magnitude"] = raw.within_field_median_rho.abs()
    top = raw.sort_values("magnitude", ascending=False).head(3).feature.tolist()
    fig, axes = plt.subplots(1, len(top), figsize=CANVAS, sharey=True)
    yap = YAP_COLUMNS["raw"]
    for ax, feature in zip(np.atleast_1d(axes), top):
        pieces = []
        for _, field in cells.groupby("image_id"):
            pair = field[[feature, yap, "experimental_group_label"]].dropna().copy()
            if len(pair) < MIN_CELLS_PER_FIELD:
                continue
            pair["feature_rank"] = rankdata(pair[feature]) / (len(pair) + 1)
            pair["yap_rank"] = rankdata(pair[yap]) / (len(pair) + 1)
            pieces.append(pair)
        data = pd.concat(pieces, ignore_index=True)
        ax.hexbin(data.feature_rank, data.yap_rank, gridsize=25, mincnt=1,
                  cmap="Purples", linewidths=0)
        coefficient = raw.set_index("feature").loc[feature, "within_field_median_rho"]
        ax.set_title(f"{DISPLAY_FEATURES.get(feature, feature)}\nmedian within-field ρ = {coefficient:+.2f}",
                     fontweight="bold")
        ax.set_xlabel("Within-field DAPI feature rank")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes[0].set_ylabel("Within-field YAP localization rank")
    fig.suptitle("Strongest DAPI–YAP relationships", fontsize=24, fontweight="bold", y=.96)
    fig.text(.5, .055, "Ranks are calculated separately within each image field before pooling for display",
             ha="center", fontsize=11, color="#555555")
    fig.tight_layout(rect=(.05, .1, .98, .86))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=DPI, facecolor="white")
    fig.savefig(output.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def run(root: Path, analysis_dir: Path | None = None) -> Path:
    analysis = analysis_dir or root / "dapi_yap_association"
    table_dir, figure_dir = analysis / "tables", analysis / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(root / "tables" / "model_a_single_cell_results.csv")
    features = load_feature_columns(root, cells)
    summary, fields = calculate_associations(cells, features)
    summary.to_csv(table_dir / "dapi_feature_yap_associations.csv", index=False, encoding="utf-8-sig")
    fields.to_csv(table_dir / "dapi_feature_yap_correlations_by_field.csv", index=False, encoding="utf-8-sig")
    plot_overview(summary, fields, figure_dir / "01_dapi_yap_association_overview.png")
    plot_top_relationships(cells, summary, figure_dir / "02_top_dapi_yap_relationships.png")
    metadata = {
        "primary_measurement": YAP_COLUMNS["raw"],
        "sensitivity_measurement": YAP_COLUMNS["background_corrected"],
        "association": "Spearman correlation",
        "primary_estimand": "median within-image correlation across image fields",
        "minimum_cells_per_field": MIN_CELLS_PER_FIELD,
        "n_cells": len(cells), "n_fields": int(cells.image_id.nunique()),
        "features": features, "dapi_intensity_excluded": True,
        "cells_are_not_biological_replicates": True,
    }
    (analysis / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return analysis


if __name__ == "__main__":
    args = parse_args()
    result = run(args.output_root, args.analysis_dir)
    print(f"DAPI–YAP association analysis written to: {result}")

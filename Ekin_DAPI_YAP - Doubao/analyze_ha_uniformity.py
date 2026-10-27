"""Quantify whether HA treatment is associated with greater single-cell uniformity.

This is a descriptive, density-matched analysis.  It does not treat cells as
biological replicates and it does not use GMM phenotype labels to define
uniformity.  Morphology is derived from DAPI geometry/neighborhood features;
YAP remains a separate post-hoc readout.

Run this file after ``run_40x_analysis.py``. Edit SETTINGS below
for presentation labels/colors, or pass ``--output-root`` to analyze another
completed result directory.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


# =========================== SETTINGS: EDIT HERE ============================
CANVAS = (13.333, 7.5)
DPI = 220
SAVE_FORMATS = ("png", "svg")

GROUP_ORDER = ("Ctrl", "HA1", "HA2")
TREATMENT_ORDER = ("HA1", "HA2")
GROUP_LABELS = {
    "Ctrl": "Control (No HA)",
    "HA1": "HA-1 (72 h, 2.5 nM)",
    "HA2": "HA-2 (48 h, 5 nM)",
}
GROUP_COLORS = {"Ctrl": "#202020", "HA1": "#8B1E9C", "HA2": "#008C95"}
DENSITY_ORDER = (2500.0, 5000.0, 7500.0, 10000.0)

# Deliberately de-redundant, interpretable morphology axes.  DAPI intensity and
# all YAP measurements are excluded from the multivariate morphology score.
MORPHOLOGY_FEATURES = {
    "area_px": {"label": "Nuclear area", "transform": "log1p"},
    "aspect_ratio": {"label": "Elongation", "transform": "log"},
    "circularity": {"label": "Circularity", "transform": "identity"},
    "solidity": {"label": "Solidity", "transform": "identity"},
    "nearest_spacing_nuclear_units": {"label": "Nuclear spacing", "transform": "log"},
    "local_crowding_area_fraction_proxy": {"label": "Local crowding", "transform": "log1p"},
    "neighbor_size_log_disagreement": {"label": "Neighbor size variability", "transform": "log1p"},
    "neighbor_shape_disagreement": {"label": "Neighbor shape variability", "transform": "log1p"},
    "neighborhood_angular_asymmetry": {"label": "Neighborhood asymmetry", "transform": "log1p"},
}

MORPHOLOGY_DOMAINS = {
    "Overall morphology": tuple(MORPHOLOGY_FEATURES),
    "Nuclear morphology": ("area_px", "aspect_ratio", "circularity", "solidity"),
    "Spatial organization": (
        "nearest_spacing_nuclear_units", "local_crowding_area_fraction_proxy",
        "neighbor_size_log_disagreement", "neighbor_shape_disagreement",
        "neighborhood_angular_asymmetry",
    ),
}

TREND_FEATURES = (
    "area_px", "circularity", "aspect_ratio",
    "nearest_spacing_nuclear_units", "local_crowding_area_fraction_proxy",
    "neighbor_shape_disagreement",
)

YAP_RAW = "posthoc_yap_raw_log2_nuclear_perinuclear_ratio"
YAP_CORRECTED = "posthoc_yap_log2_nuclear_perinuclear_ratio"
# ========================= END SETTINGS =====================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_root = (Path(__file__).resolve().parent / "outputs" /
                    "composite_no_dapi_intensity" / "40x" / "all_fields")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=default_root,
                        help="Completed pipeline output containing tables/model_a_single_cell_results.csv")
    parser.add_argument("--analysis-dir", type=Path, default=None,
                        help="Default: <output-root>/ha_uniformity")
    return parser.parse_args(argv)


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.titlesize": 17, "axes.labelsize": 14,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "legend.fontsize": 11, "figure.facecolor": "white",
        "axes.facecolor": "white", "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 1.1,
        "savefig.facecolor": "white",
    })


def _save(fig: plt.Figure, figure_dir: Path, stem: str) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    for extension in SAVE_FORMATS:
        fig.savefig(figure_dir / f"{stem}.{extension}", dpi=DPI, bbox_inches=None)
    plt.close(fig)


def _transform(series: pd.Series, method: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    if method == "identity":
        return values
    if method == "log1p":
        return np.log1p(values.where(values >= 0))
    if method == "log":
        return np.log(values.where(values > 0))
    raise ValueError(f"Unknown transform: {method}")


def _mad(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return math.nan
    median = float(np.median(array))
    return float(np.median(np.abs(array - median)))


def _safe_log2_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or numerator <= 0 or denominator <= 0:
        return math.nan
    return float(np.log2(numerator / denominator))


def validate_cells(cells: pd.DataFrame) -> None:
    required = {
        "experimental_group_label", "seeding_density_cells_per_cm2", "image_id",
        *MORPHOLOGY_FEATURES.keys(), YAP_RAW, YAP_CORRECTED,
    }
    missing = sorted(required - set(cells.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    groups = set(cells["experimental_group_label"].dropna().unique())
    if not set(GROUP_ORDER).issubset(groups):
        raise ValueError(f"Expected groups {GROUP_ORDER}; found {sorted(groups)}")


def transformed_morphology(cells: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=cells.index)
    for feature, spec in MORPHOLOGY_FEATURES.items():
        result[feature] = _transform(cells[feature], spec["transform"])
    return result


def calculate_uniformity(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return multivariate dispersion, feature dispersion, and location shifts.

    At each density, every feature is scaled by the matching Control IQR.
    Multivariate dispersion is the median per-cell RMS distance from that
    group's own median profile.  A positive uniformity gain means lower
    dispersion than the density-matched Control.
    """
    transformed = transformed_morphology(cells)
    meta = cells[["experimental_group_label", "seeding_density_cells_per_cm2", "image_id"]]
    data = pd.concat([meta, transformed], axis=1)
    multivariate_rows: list[dict] = []
    feature_rows: list[dict] = []
    location_rows: list[dict] = []

    for density in DENSITY_ORDER:
        density_data = data[data["seeding_density_cells_per_cm2"].eq(density)].copy()
        ctrl = density_data[density_data["experimental_group_label"].eq("Ctrl")]
        ctrl_median = ctrl[list(MORPHOLOGY_FEATURES)].median()
        ctrl_iqr = (ctrl[list(MORPHOLOGY_FEATURES)].quantile(.75) -
                    ctrl[list(MORPHOLOGY_FEATURES)].quantile(.25))
        usable = [f for f in MORPHOLOGY_FEATURES if np.isfinite(ctrl_iqr[f]) and ctrl_iqr[f] > 1e-12]
        if len(usable) < 2:
            raise ValueError(f"Too few variable Control features at density {density:g}")
        scaled = density_data[usable].sub(ctrl_median[usable], axis=1).div(ctrl_iqr[usable], axis=1)

        group_dispersion: dict[str, float] = {}
        for group in GROUP_ORDER:
            mask = density_data["experimental_group_label"].eq(group)
            z = scaled.loc[mask]
            center = z.median(axis=0)
            cell_distance = np.sqrt(np.nanmean(np.square(z.sub(center, axis=1).to_numpy(float)), axis=1))
            group_dispersion[group] = float(np.nanmedian(cell_distance))
            multivariate_rows.append({
                "seeding_density_cells_per_cm2": density,
                "experimental_group_label": group,
                "n_cells": int(mask.sum()),
                "n_features": len(usable),
                "multivariate_dispersion": group_dispersion[group],
                "uniformity_gain_vs_ctrl_log2": math.nan,
            })

        ctrl_dispersion = group_dispersion["Ctrl"]
        for row in multivariate_rows[-len(GROUP_ORDER):]:
            row["uniformity_gain_vs_ctrl_log2"] = -_safe_log2_ratio(
                row["multivariate_dispersion"], ctrl_dispersion)

        for feature in usable:
            ctrl_values = density_data.loc[
                density_data["experimental_group_label"].eq("Ctrl"), feature
            ].dropna()
            ctrl_feature_mad = _mad(ctrl_values)
            ctrl_feature_median = float(ctrl_values.median())
            scale = float(ctrl_iqr[feature])
            for group in GROUP_ORDER:
                values = density_data.loc[
                    density_data["experimental_group_label"].eq(group), feature
                ].dropna()
                group_mad = _mad(values)
                group_median = float(values.median()) if len(values) else math.nan
                feature_rows.append({
                    "seeding_density_cells_per_cm2": density,
                    "experimental_group_label": group,
                    "feature": feature,
                    "feature_label": MORPHOLOGY_FEATURES[feature]["label"],
                    "n_cells_valid": int(len(values)),
                    "mad_transformed": group_mad,
                    "ctrl_mad_transformed": ctrl_feature_mad,
                    "uniformity_gain_vs_ctrl_log2": -_safe_log2_ratio(group_mad, ctrl_feature_mad),
                })
                location_rows.append({
                    "seeding_density_cells_per_cm2": density,
                    "experimental_group_label": group,
                    "feature": feature,
                    "feature_label": MORPHOLOGY_FEATURES[feature]["label"],
                    "n_cells_valid": int(len(values)),
                    "median_transformed": group_median,
                    "ctrl_median_transformed": ctrl_feature_median,
                    "median_shift_in_ctrl_iqr": ((group_median - ctrl_feature_median) / scale
                                                 if np.isfinite(group_median) else math.nan),
                })

    return (pd.DataFrame(multivariate_rows), pd.DataFrame(feature_rows),
            pd.DataFrame(location_rows))


def calculate_yap(cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for density in DENSITY_ORDER:
        density_cells = cells[cells["seeding_density_cells_per_cm2"].eq(density)]
        for measurement, measurement_label in (
            (YAP_RAW, "Uncorrected log2 nuclear/perinuclear"),
            (YAP_CORRECTED, "Background-corrected log2 nuclear/perinuclear"),
        ):
            ctrl_values = pd.to_numeric(
                density_cells.loc[density_cells["experimental_group_label"].eq("Ctrl"), measurement],
                errors="coerce",
            ).dropna()
            ctrl_mad = _mad(ctrl_values)
            for group in GROUP_ORDER:
                group_all = density_cells[density_cells["experimental_group_label"].eq(group)]
                values = pd.to_numeric(group_all[measurement], errors="coerce").dropna()
                mad = _mad(values)
                rows.append({
                    "seeding_density_cells_per_cm2": density,
                    "experimental_group_label": group,
                    "measurement": measurement,
                    "measurement_label": measurement_label,
                    "n_cells_total": int(len(group_all)),
                    "n_cells_valid": int(len(values)),
                    "valid_fraction": float(len(values) / len(group_all)) if len(group_all) else math.nan,
                    "median": float(values.median()) if len(values) else math.nan,
                    "q25": float(values.quantile(.25)) if len(values) else math.nan,
                    "q75": float(values.quantile(.75)) if len(values) else math.nan,
                    "mad": mad,
                    "ctrl_mad": ctrl_mad,
                    "uniformity_gain_vs_ctrl_log2": -_safe_log2_ratio(mad, ctrl_mad),
                })
    return pd.DataFrame(rows)


def calculate_domain_uniformity(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate predefined-domain scores and leave-one-feature-out sensitivity."""
    transformed = transformed_morphology(cells)
    meta = cells[["experimental_group_label", "seeding_density_cells_per_cm2"]]
    data = pd.concat([meta, transformed], axis=1)

    def subset_rows(domain: str, features: tuple[str, ...], omitted: str | None = None) -> list[dict]:
        rows: list[dict] = []
        for density in DENSITY_ORDER:
            density_data = data[data["seeding_density_cells_per_cm2"].eq(density)]
            ctrl = density_data[density_data["experimental_group_label"].eq("Ctrl")]
            ctrl_iqr = ctrl[list(features)].quantile(.75) - ctrl[list(features)].quantile(.25)
            usable = [f for f in features if np.isfinite(ctrl_iqr[f]) and ctrl_iqr[f] > 1e-12]
            ctrl_median = ctrl[usable].median()
            scaled = density_data[usable].sub(ctrl_median, axis=1).div(ctrl_iqr[usable], axis=1)
            dispersions: dict[str, float] = {}
            counts: dict[str, int] = {}
            for group in GROUP_ORDER:
                mask = density_data["experimental_group_label"].eq(group)
                values = scaled.loc[mask]
                center = values.median(axis=0)
                distance = np.sqrt(np.nanmean(np.square(values.sub(center, axis=1).to_numpy(float)), axis=1))
                dispersions[group] = float(np.nanmedian(distance))
                counts[group] = int(mask.sum())
            for group in GROUP_ORDER:
                rows.append({
                    "domain": domain,
                    "omitted_feature": omitted,
                    "seeding_density_cells_per_cm2": density,
                    "experimental_group_label": group,
                    "n_cells": counts[group],
                    "n_features": len(usable),
                    "multivariate_dispersion": dispersions[group],
                    "uniformity_gain_vs_ctrl_log2": -_safe_log2_ratio(
                        dispersions[group], dispersions["Ctrl"]),
                })
        return rows

    domain_rows: list[dict] = []
    for domain, features in MORPHOLOGY_DOMAINS.items():
        domain_rows.extend(subset_rows(domain, features))
    sensitivity_rows: list[dict] = []
    all_features = tuple(MORPHOLOGY_FEATURES)
    for omitted in all_features:
        features = tuple(feature for feature in all_features if feature != omitted)
        sensitivity_rows.extend(subset_rows("Overall morphology", features, omitted=omitted))
    return pd.DataFrame(domain_rows), pd.DataFrame(sensitivity_rows)


def summarize_uniformity(multivariate: pd.DataFrame, feature: pd.DataFrame,
                         yap: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for group in TREATMENT_ORDER:
        morphology = multivariate[multivariate["experimental_group_label"].eq(group)]
        raw_yap = yap[(yap["experimental_group_label"].eq(group)) & (yap["measurement"].eq(YAP_RAW))]
        for domain, frame in (("Multivariate DAPI morphology", morphology),
                              ("YAP localization (uncorrected)", raw_yap)):
            gains = frame["uniformity_gain_vs_ctrl_log2"].dropna()
            rows.append({
                "experimental_group_label": group,
                "domain": domain,
                "n_densities_evaluable": int(len(gains)),
                "n_densities_more_uniform_than_ctrl": int((gains > 0).sum()),
                "median_uniformity_gain_log2": float(gains.median()) if len(gains) else math.nan,
                "minimum_uniformity_gain_log2": float(gains.min()) if len(gains) else math.nan,
                "maximum_uniformity_gain_log2": float(gains.max()) if len(gains) else math.nan,
            })
        per_feature = feature[feature["experimental_group_label"].eq(group)]
        for feature_name, sub in per_feature.groupby("feature", sort=False):
            gains = sub["uniformity_gain_vs_ctrl_log2"].dropna()
            rows.append({
                "experimental_group_label": group,
                "domain": f"Feature: {MORPHOLOGY_FEATURES[feature_name]['label']}",
                "n_densities_evaluable": int(len(gains)),
                "n_densities_more_uniform_than_ctrl": int((gains > 0).sum()),
                "median_uniformity_gain_log2": float(gains.median()) if len(gains) else math.nan,
                "minimum_uniformity_gain_log2": float(gains.min()) if len(gains) else math.nan,
                "maximum_uniformity_gain_log2": float(gains.max()) if len(gains) else math.nan,
            })
    return pd.DataFrame(rows)


def _density_labels() -> list[str]:
    return [f"{density / 1000:g}k" for density in DENSITY_ORDER]


def plot_uniformity_overview(multivariate: pd.DataFrame, feature: pd.DataFrame,
                             figure_dir: Path) -> None:
    fig, (ax, heat_ax) = plt.subplots(1, 2, figsize=CANVAS, gridspec_kw={"width_ratios": [1, 2.2]})
    x = np.arange(len(DENSITY_ORDER))
    width = .34
    for index, group in enumerate(TREATMENT_ORDER):
        sub = (multivariate[multivariate["experimental_group_label"].eq(group)]
               .set_index("seeding_density_cells_per_cm2").reindex(DENSITY_ORDER))
        ax.bar(x + (index - .5) * width, sub["uniformity_gain_vs_ctrl_log2"], width,
               color=GROUP_COLORS[group], label=GROUP_LABELS[group])
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xticks(x, _density_labels())
    ax.set_xlabel("Seeding density (cells/cm²)")
    ax.set_ylabel("Uniformity gain vs Control\n−log₂(dispersion ratio)")
    ax.set_title("Overall morphology uniformity", fontweight="bold")
    ax.legend(frameon=False, loc="best")

    treatment = feature[feature["experimental_group_label"].isin(TREATMENT_ORDER)].copy()
    treatment["row"] = treatment.apply(
        lambda row: f"{row['experimental_group_label']} · {row['seeding_density_cells_per_cm2']/1000:g}k", axis=1)
    row_order = [f"{group} · {density/1000:g}k" for group in TREATMENT_ORDER for density in DENSITY_ORDER]
    col_order = list(MORPHOLOGY_FEATURES)
    matrix = treatment.pivot(index="row", columns="feature", values="uniformity_gain_vs_ctrl_log2")
    matrix = matrix.reindex(index=row_order, columns=col_order)
    finite = np.abs(matrix.to_numpy(float)[np.isfinite(matrix.to_numpy(float))])
    limit = max(float(np.quantile(finite, .95)) if len(finite) else 1.0, .25)
    image = heat_ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                           norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
    heat_ax.set_xticks(np.arange(len(col_order)),
                       [MORPHOLOGY_FEATURES[f]["label"] for f in col_order],
                       rotation=38, ha="right")
    heat_ax.set_yticks(np.arange(len(row_order)), row_order)
    heat_ax.set_title("Feature-level uniformity", fontweight="bold")
    colorbar = fig.colorbar(image, ax=heat_ax, fraction=.035, pad=.02)
    colorbar.set_label("Uniformity gain vs Control")
    fig.suptitle("Does HA increase morphological uniformity?", fontsize=24, fontweight="bold", y=.97)
    fig.text(.5, .91, "Density-matched, DAPI morphology only · positive values indicate lower within-field dispersion",
             ha="center", fontsize=13, color="#444444")
    fig.subplots_adjust(left=.08, right=.95, bottom=.24, top=.82, wspace=.34)
    _save(fig, figure_dir, "01_ha_morphology_uniformity")


def plot_effect_profile(location: pd.DataFrame, figure_dir: Path) -> None:
    treatment = location[location["experimental_group_label"].isin(TREATMENT_ORDER)].copy()
    treatment["row"] = treatment.apply(
        lambda row: f"{row['experimental_group_label']} · {row['seeding_density_cells_per_cm2']/1000:g}k", axis=1)
    row_order = [f"{group} · {density/1000:g}k" for group in TREATMENT_ORDER for density in DENSITY_ORDER]
    col_order = list(MORPHOLOGY_FEATURES)
    matrix = treatment.pivot(index="row", columns="feature", values="median_shift_in_ctrl_iqr")
    matrix = matrix.reindex(index=row_order, columns=col_order)
    finite = np.abs(matrix.to_numpy(float)[np.isfinite(matrix.to_numpy(float))])
    limit = max(float(np.quantile(finite, .95)) if len(finite) else 1.0, .5)
    fig, ax = plt.subplots(figsize=CANVAS)
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm",
                      norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
    ax.set_xticks(np.arange(len(col_order)), [MORPHOLOGY_FEATURES[f]["label"] for f in col_order],
                  rotation=32, ha="right")
    ax.set_yticks(np.arange(len(row_order)), row_order)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix.iloc[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:+.2f}", ha="center", va="center",
                        fontsize=9, color="white" if abs(value) > limit * .58 else "#222222")
    colorbar = fig.colorbar(image, ax=ax, fraction=.025, pad=.02)
    colorbar.set_label("Median shift relative to Control (Control IQR)")
    fig.suptitle("HA-associated morphology shifts", fontsize=24, fontweight="bold", y=.97)
    fig.text(.5, .91, "Each HA condition is compared with Control at the same seeding density",
             ha="center", fontsize=13, color="#444444")
    fig.subplots_adjust(left=.14, right=.93, bottom=.24, top=.82)
    _save(fig, figure_dir, "02_ha_morphology_effect_profile")


def plot_feature_trends(cells: pd.DataFrame, figure_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=CANVAS)
    x = np.arange(len(DENSITY_ORDER))
    for ax, feature in zip(axes.flat, TREND_FEATURES):
        for group in GROUP_ORDER:
            medians, lower, upper = [], [], []
            for density in DENSITY_ORDER:
                values = pd.to_numeric(cells.loc[
                    cells["experimental_group_label"].eq(group) &
                    cells["seeding_density_cells_per_cm2"].eq(density), feature
                ], errors="coerce").dropna()
                median = float(values.median())
                medians.append(median)
                lower.append(median - float(values.quantile(.25)))
                upper.append(float(values.quantile(.75)) - median)
            ax.errorbar(x, medians, yerr=np.array([lower, upper]), color=GROUP_COLORS[group],
                        marker="o", markersize=5, linewidth=2, capsize=3, label=GROUP_LABELS[group])
        ax.set_title(MORPHOLOGY_FEATURES[feature]["label"], fontweight="bold")
        ax.set_xticks(x, _density_labels())
        ax.grid(axis="y", alpha=.18)
    axes[1, 0].set_xlabel("Seeding density (cells/cm²)")
    axes[1, 1].set_xlabel("Seeding density (cells/cm²)")
    axes[1, 2].set_xlabel("Seeding density (cells/cm²)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(.5, .015))
    fig.suptitle("Interpretable morphology trends", fontsize=24, fontweight="bold", y=.97)
    fig.text(.5, .91, "Median and interquartile range · cells describe one image field per condition",
             ha="center", fontsize=13, color="#444444")
    fig.subplots_adjust(left=.08, right=.98, bottom=.14, top=.82, hspace=.42, wspace=.28)
    _save(fig, figure_dir, "03_ha_morphology_trends")


def plot_yap(yap: pd.DataFrame, figure_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=CANVAS)
    x = np.arange(len(DENSITY_ORDER))
    raw = yap[yap["measurement"].eq(YAP_RAW)]
    for group in GROUP_ORDER:
        sub = raw[raw["experimental_group_label"].eq(group)].set_index(
            "seeding_density_cells_per_cm2").reindex(DENSITY_ORDER)
        median = sub["median"].to_numpy(float)
        yerr = np.vstack([median - sub["q25"].to_numpy(float), sub["q75"].to_numpy(float) - median])
        axes[0].errorbar(x, median, yerr=yerr, marker="o", linewidth=2, capsize=3,
                         color=GROUP_COLORS[group], label=GROUP_LABELS[group])
    axes[0].axhline(0, color="#777777", linewidth=1, linestyle="--")
    axes[0].set_xticks(x, _density_labels())
    axes[0].set_title("YAP localization", fontweight="bold")
    axes[0].set_ylabel("Uncorrected log₂(nuclear/perinuclear)")
    axes[0].set_xlabel("Seeding density (cells/cm²)")
    axes[0].legend(frameon=False, fontsize=9)

    width = .34
    for index, group in enumerate(TREATMENT_ORDER):
        sub = raw[raw["experimental_group_label"].eq(group)].set_index(
            "seeding_density_cells_per_cm2").reindex(DENSITY_ORDER)
        axes[1].bar(x + (index - .5) * width, sub["uniformity_gain_vs_ctrl_log2"], width,
                    color=GROUP_COLORS[group], label=GROUP_LABELS[group])
    axes[1].axhline(0, color="#333333", linewidth=1)
    axes[1].set_xticks(x, _density_labels())
    axes[1].set_title("YAP localization uniformity", fontweight="bold")
    axes[1].set_ylabel("Uniformity gain vs Control")
    axes[1].set_xlabel("Seeding density (cells/cm²)")

    corrected = yap[yap["measurement"].eq(YAP_CORRECTED)]
    for index, group in enumerate(GROUP_ORDER):
        sub = corrected[corrected["experimental_group_label"].eq(group)].set_index(
            "seeding_density_cells_per_cm2").reindex(DENSITY_ORDER)
        axes[2].bar(x + (index - 1) * .24, 100 * sub["valid_fraction"], .23,
                    color=GROUP_COLORS[group], label=GROUP_LABELS[group])
    axes[2].set_xticks(x, _density_labels())
    axes[2].set_ylim(0, 105)
    axes[2].set_title("Corrected-ratio availability", fontweight="bold")
    axes[2].set_ylabel("Cells with valid corrected ratio (%)")
    axes[2].set_xlabel("Seeding density (cells/cm²)")
    fig.suptitle("YAP as an independent post-hoc readout", fontsize=24, fontweight="bold", y=.97)
    fig.text(.5, .91, "YAP is excluded from morphology preprocessing, UMAP and GMM",
             ha="center", fontsize=13, color="#444444")
    fig.subplots_adjust(left=.07, right=.98, bottom=.15, top=.80, wspace=.32)
    _save(fig, figure_dir, "04_yap_localization_and_uniformity")


def plot_evidence_summary(domains: pd.DataFrame, yap: pd.DataFrame, figure_dir: Path) -> None:
    records: list[dict] = []
    for group in TREATMENT_ORDER:
        for domain in MORPHOLOGY_DOMAINS:
            values = domains[(domains["experimental_group_label"].eq(group)) &
                             (domains["domain"].eq(domain))]["uniformity_gain_vs_ctrl_log2"].dropna()
            records.append({"group": group, "domain": domain,
                            "positive": int((values > 0).sum()), "median": float(values.median())})
        values = yap[(yap["experimental_group_label"].eq(group)) &
                     (yap["measurement"].eq(YAP_RAW))]["uniformity_gain_vs_ctrl_log2"].dropna()
        records.append({"group": group, "domain": "YAP localization",
                        "positive": int((values > 0).sum()), "median": float(values.median())})
    summary = pd.DataFrame(records)
    domain_order = [*MORPHOLOGY_DOMAINS, "YAP localization"]
    count_matrix = summary.pivot(index="domain", columns="group", values="positive").reindex(
        index=domain_order, columns=TREATMENT_ORDER)
    gain_matrix = summary.pivot(index="domain", columns="group", values="median").reindex(
        index=domain_order, columns=TREATMENT_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=CANVAS, gridspec_kw={"width_ratios": [1, 1.2]})
    count_image = axes[0].imshow(count_matrix, vmin=0, vmax=4, cmap="YlGn", aspect="auto")
    axes[0].set_xticks(range(2), [GROUP_LABELS[g] for g in TREATMENT_ORDER])
    axes[0].set_yticks(range(len(domain_order)), domain_order)
    axes[0].set_title("Consistency across densities", fontweight="bold")
    for row in range(count_matrix.shape[0]):
        for col in range(count_matrix.shape[1]):
            axes[0].text(col, row, f"{int(count_matrix.iloc[row, col])}/4",
                         ha="center", va="center", fontsize=15, fontweight="bold")
    colorbar = fig.colorbar(count_image, ax=axes[0], fraction=.045, pad=.03, ticks=range(5))
    colorbar.set_label("Densities more uniform than Control")

    values = gain_matrix.to_numpy(float)
    limit = max(float(np.nanmax(np.abs(values))), .25)
    gain_image = axes[1].imshow(gain_matrix, cmap="RdBu_r", aspect="auto",
                                norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
    axes[1].set_xticks(range(2), [GROUP_LABELS[g] for g in TREATMENT_ORDER])
    axes[1].set_yticks(range(len(domain_order)), domain_order)
    axes[1].set_title("Median effect across densities", fontweight="bold")
    for row in range(gain_matrix.shape[0]):
        for col in range(gain_matrix.shape[1]):
            value = gain_matrix.iloc[row, col]
            axes[1].text(col, row, f"{value:+.2f}", ha="center", va="center",
                         fontsize=14, fontweight="bold",
                         color="white" if abs(value) > limit * .55 else "#222222")
    colorbar = fig.colorbar(gain_image, ax=axes[1], fraction=.038, pad=.03)
    colorbar.set_label("Median uniformity gain vs Control")
    fig.suptitle("Evidence for HA-associated uniformity", fontsize=24, fontweight="bold", y=.97)
    fig.text(.5, .91, "Positive gain means lower dispersion · each density is matched to its own Control",
             ha="center", fontsize=13, color="#444444")
    fig.subplots_adjust(left=.18, right=.95, bottom=.16, top=.80, wspace=.48)
    _save(fig, figure_dir, "05_ha_uniformity_evidence_summary")


def write_report(path: Path, multivariate: pd.DataFrame, summary: pd.DataFrame,
                 yap: pd.DataFrame, domains: pd.DataFrame,
                 sensitivity: pd.DataFrame) -> None:
    lines = [
        "# HA uniformity analysis", "",
        "## Scope", "",
        "This analysis asks whether HA treatment is associated with lower within-field single-cell variability "
        "at the same seeding density. It is descriptive: each condition has one 40× image field, so cells are "
        "not treated as biological replicates and no p-values are reported.", "",
        "Positive uniformity gain means lower dispersion than density-matched Control. A value of +1 means "
        "half the Control dispersion; −1 means twice the Control dispersion.", "",
        "## Multivariate morphology result", "",
        "| Treatment | Densities more uniform | Median gain | Gains at 2.5k / 5k / 7.5k / 10k |", "|---|---:|---:|---|",
    ]
    for group in TREATMENT_ORDER:
        sub = (multivariate[multivariate["experimental_group_label"].eq(group)]
               .set_index("seeding_density_cells_per_cm2").reindex(DENSITY_ORDER))
        gains = sub["uniformity_gain_vs_ctrl_log2"]
        formatted = " / ".join("NA" if not np.isfinite(value) else f"{value:+.2f}" for value in gains)
        lines.append(f"| {GROUP_LABELS[group]} | {int((gains > 0).sum())}/4 | {gains.median():+.2f} | {formatted} |")
    lines += ["", "## YAP result", "",
              "Uncorrected YAP ratios are used for coverage; corrected ratios are reported separately as a sensitivity readout.", "",
              "| Treatment | Densities with lower YAP dispersion | Median gain |", "|---|---:|---:|"]
    for group in TREATMENT_ORDER:
        row = summary[(summary["experimental_group_label"].eq(group)) &
                      (summary["domain"].eq("YAP localization (uncorrected)"))].iloc[0]
        lines.append(f"| {GROUP_LABELS[group]} | {int(row['n_densities_more_uniform_than_ctrl'])}/"
                     f"{int(row['n_densities_evaluable'])} | {row['median_uniformity_gain_log2']:+.2f} |")
    lines += ["", "## Domain and sensitivity checks", "",
              "| Treatment | Domain | Densities more uniform | Median gain |", "|---|---|---:|---:|"]
    for group in TREATMENT_ORDER:
        for domain in MORPHOLOGY_DOMAINS:
            values = domains[(domains["experimental_group_label"].eq(group)) &
                             (domains["domain"].eq(domain))]["uniformity_gain_vs_ctrl_log2"].dropna()
            lines.append(f"| {GROUP_LABELS[group]} | {domain} | {int((values > 0).sum())}/"
                         f"{len(values)} | {values.median():+.2f} |")
        checks = []
        for omitted, sub in sensitivity[
            sensitivity["experimental_group_label"].eq(group)
        ].groupby("omitted_feature", sort=False):
            gains = sub["uniformity_gain_vs_ctrl_log2"].dropna()
            checks.append((omitted, float(gains.median()), int((gains > 0).sum())))
        best = max(checks, key=lambda item: item[1])
        lines.append(f"| {GROUP_LABELS[group]} | Leave-one-feature-out best case "
                     f"(omit {MORPHOLOGY_FEATURES[best[0]]['label']}) | {best[2]}/4 | {best[1]:+.2f} |")
    corrected = yap[yap["measurement"].eq(YAP_CORRECTED)]
    lines += ["", "## Interpretation limits", "",
              "- Uniform morphology is a quality-consistency proxy, not direct proof of pluripotency or functional quality.",
              "- HA-1 and HA-2 differ in both exposure time and concentration; their effects cannot be separated.",
              "- Seeding density is matched rather than pooled.",
              "- Independent wells/cultures are required for inferential statistics.",
              f"- Background-corrected YAP availability ranges from {100*corrected['valid_fraction'].min():.1f}% "
              f"to {100*corrected['valid_fraction'].max():.1f}% across conditions.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(output_root: Path, analysis_dir: Path | None = None) -> dict:
    output_root = output_root.resolve()
    analysis_dir = (analysis_dir or output_root / "ha_uniformity").resolve()
    table_dir = analysis_dir / "tables"
    figure_dir = analysis_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    source = output_root / "tables" / "model_a_single_cell_results.csv"
    if not source.exists():
        raise FileNotFoundError(f"Missing completed single-cell table: {source}")
    cells = pd.read_csv(source)
    validate_cells(cells)
    multivariate, feature, location = calculate_uniformity(cells)
    domains, sensitivity = calculate_domain_uniformity(cells)
    yap = calculate_yap(cells)
    summary = summarize_uniformity(multivariate, feature, yap)

    multivariate.to_csv(table_dir / "ha_morphology_uniformity_by_density.csv", index=False, encoding="utf-8-sig")
    feature.to_csv(table_dir / "ha_feature_uniformity_by_density.csv", index=False, encoding="utf-8-sig")
    location.to_csv(table_dir / "ha_morphology_location_shift_by_density.csv", index=False, encoding="utf-8-sig")
    yap.to_csv(table_dir / "ha_yap_uniformity_by_density.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(table_dir / "ha_uniformity_summary.csv", index=False, encoding="utf-8-sig")
    domains.to_csv(table_dir / "ha_domain_uniformity_by_density.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(table_dir / "ha_leave_one_feature_out_sensitivity.csv", index=False, encoding="utf-8-sig")

    _style()
    plot_uniformity_overview(multivariate, feature, figure_dir)
    plot_effect_profile(location, figure_dir)
    plot_feature_trends(cells, figure_dir)
    plot_yap(yap, figure_dir)
    plot_evidence_summary(domains, yap, figure_dir)
    write_report(analysis_dir / "HA_UNIFORMITY_REPORT.md", multivariate, summary, yap,
                 domains, sensitivity)

    info = {
        "objective": "Quantify whether HA is associated with greater cell-state uniformity",
        "source_table": str(source),
        "n_cells": int(len(cells)),
        "groups": list(GROUP_ORDER),
        "seeding_densities_cells_per_cm2": list(DENSITY_ORDER),
        "morphology_features": list(MORPHOLOGY_FEATURES),
        "morphology_domains": {name: list(features) for name, features in MORPHOLOGY_DOMAINS.items()},
        "sensitivity_analysis": "leave one morphology feature out and recompute all density-matched scores",
        "dapi_intensity_used": False,
        "gmm_phenotypes_used": False,
        "yap_used_in_morphology_score": False,
        "uniformity_definition": "negative log2 ratio of within-group dispersion to density-matched Control dispersion",
        "multivariate_dispersion": "median per-cell RMS distance to group median after scaling each feature by density-matched Control IQR",
        "feature_dispersion": "median absolute deviation after documented feature transform",
        "inference": "descriptive only; one 40x image field per condition; no cell-level p-values",
        "limitations": [
            "Uniformity is a quality-consistency proxy, not proof of pluripotency or function",
            "HA1 and HA2 differ in both exposure time and concentration",
            "Independent culture/well replicates are not provided",
            "Uncorrected YAP has higher coverage but remains sensitive to image background",
        ],
    }
    (analysis_dir / "analysis_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return {"analysis_dir": str(analysis_dir), "summary": summary.to_dict(orient="records")}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run(args.output_root, args.analysis_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Regenerate all presentation figures from existing result tables.

This script never segments cells, measures YAP, fits UMAP or refits the GMM.
Edit only the SETTINGS section below to change titles, fonts, colors or features.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


# =========================== SETTINGS: EDIT HERE ============================
CANVAS = (13.333, 7.5)       # 16:9 PowerPoint slide, inches
SCATTER_CANVAS = (8.5, 8.5)  # Square export for single-panel UMAP scatter plots.
DPI = 220
TITLE_SIZE = 24
SUBTITLE_SIZE = 13
AXIS_SIZE = 16
TICK_SIZE = 12
LEGEND_SIZE = 12
POINT_SIZE = 10
POINT_ALPHA = 0.62
SAVE_FORMATS = ("png", "svg")  # SVG stays sharp after scaling in PowerPoint.

GMM_TERM_NOTES = {
    "K": "Number of GMM components (statistical phenotypes)",
    "Penalized BIC": "Model fit plus penalties for complexity and tiny/fragmented clusters; lower is better",
    "Delta penalized BIC": "Difference from the best penalized BIC; zero is the selected solution",
    "Selected K": "K with the minimum penalized BIC; not proof of biological cell types",
}

GROUP_ORDER = ("Ctrl", "HA1", "HA2")
GROUP_LABELS = {
    "Ctrl": "Control (No HA)",
    "HA1": "HA-1 (72h, 2.5nM)",
    "HA2": "HA-2 (48h, 5nM)",
}
GROUP_COLORS = {"Ctrl": "#171717", "HA1": "#D000B5", "HA2": "#00B8C4"}
DENSITY_ORDER = (2500.0, 5000.0, 7500.0, 10000.0)
DENSITY_COLORS = {2500.0: "#482878", 5000.0: "#31688E", 7500.0: "#35B779", 10000.0: "#E6AB02"}

# Representative, interpretable morphology axes for HA-centred reporting.
# These are direct DAPI-derived measurements; DAPI intensity and YAP are not
# included in the morphology effect score.
HA_MORPHOLOGY_METRICS = {
    "area_px": ("Nuclear area", "px²"),
    "aspect_ratio": ("Elongation", "aspect ratio"),
    "circularity": ("Circularity", "0–1"),
    "nearest_spacing_nuclear_units": ("Nuclear spacing", "nuclear units"),
    "local_crowding_area_fraction_proxy": ("Local crowding", "area fraction"),
    "neighbor_shape_disagreement": ("Neighbor heterogeneity", "shape difference"),
}
YAP_RAW_COLUMN = "posthoc_yap_raw_log2_nuclear_perinuclear_ratio"

PHENOTYPE_NOTES = {
    "Phenotype 1": "Near-average profile",
    "Phenotype 2": "Higher DAPI intensity",
    "Phenotype 3": "Larger nuclei",
    "Phenotype 4": "Smaller nuclei; sparse neighborhood",
    "Phenotype 5": "Elongated, irregular nuclei",
    "Phenotype 6": "Variable neighboring nuclear shapes",
}

PHENOTYPE_PALETTE = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # rose
    "#D55E00",  # vermillion
    "#56B4E9",  # sky blue
    "#6A3D9A",  # purple
    "#8C8C00",  # olive (only used if K > 7)
    "#7A7A7A",  # grey
    "#8B4513",  # brown
    "#00A6A6",  # teal
    "#E7298A",  # magenta
)

ACTIVE_MAGNIFICATION = ""
HEATMAP_PHENOTYPE_NOTES = {
    "Phenotype 1": "Near-average profile",
    "Phenotype 2": "Higher DAPI intensity",
    "Phenotype 3": "Larger nuclei",
    "Phenotype 4": "Small nuclei; sparse context",
    "Phenotype 5": "Elongated, irregular nuclei",
    "Phenotype 6": "Variable neighbor shapes",
}

HEATMAP_FEATURES = {
    "area_px": "Nuclear area",
    "equivalent_diameter_px": "Equivalent diameter",
    "aspect_ratio": "Aspect ratio",
    "circularity": "Circularity",
    "solidity": "Solidity",
    "dapi_mean_intensity": "Mean DAPI intensity",
    "chromatin_cv_proxy": "DAPI variation",
    "chromatin_range_ratio": "DAPI intensity range",
    "nn1_distance_px": "Nearest-neighbor distance",
    "knn6_distance_mean_px": "Mean 6-neighbor distance",
    "local_crowding_area_fraction_proxy": "Local crowding",
    "nb_area_mean_px2": "Neighbor nuclear area",
    "nb_area_std_px2": "Neighbor size variability",
    "nb_aspect_ratio_mean": "Neighbor aspect ratio",
    "neighbor_shape_disagreement": "Neighbor shape variability",
    "neighborhood_angular_asymmetry": "Neighborhood asymmetry",
}

TITLES = {
    "phenotype_umap": "Morphology phenotypes",
    "group_umap": "HA groups on the morphology map",
    "density_umap": "Seeding density on the morphology map",
    "composition": "Morphology phenotype counts",
    "composition_fraction": "Phenotype composition",
    "heatmap": "DAPI morphology profiles",
    "membership": "Phenotype membership confidence",
    "yap": "YAP nuclear enrichment by condition",
    "yap_phenotype": "YAP localization across morphology phenotypes",
    "gmm": "GMM model selection",
    "ha_effect_heatmap": "HA-associated morphology shifts",
    "ha_trends": "Morphology across seeding densities",
    "ha_yap_relation": "Morphology shift and YAP localization",
    "ha_main_morphology": "Overall morphology effect of HA treatment",
    "ha_main_yap": "Overall YAP localization effect of HA treatment",
    "ha_density_effects": "Density-specific morphology effects of HA treatment",
}

# Long explanations are stored in SVG metadata rather than drawn on the slide.
# They remain searchable in the SVG source as <dc:description>.
SVG_NOTES = {
    "01_umap_phenotypes": "DAPI-derived features only. Colors show the dominant GMM component. YAP and experimental labels were excluded from preprocessing, UMAP and GMM. Coordinates were rigidly rotated to place the main UMAP axis horizontally; distances and cluster membership are unchanged.",
    "02_umap_experimental_groups": "Experimental groups are overlaid post hoc on the DAPI-only UMAP. Group labels were excluded from preprocessing, UMAP and GMM. The rigid display rotation does not change distances.",
    "03_umap_seeding_density": "Seeding densities are overlaid post hoc on the DAPI-only UMAP and were excluded from preprocessing, UMAP and GMM.",
    "04_phenotype_composition": "Stacked bars show actual single-cell counts assigned to each dominant GMM phenotype. Total bar height is the observed cell count, not a normalized percentage. One image field is available per condition, so the comparison is descriptive.",
    "05_phenotype_feature_profiles": "Each tile is the phenotype mean relative to the pooled mean in overall standard-deviation units. Labels P1, P2, and so on are statistical GMM components, not validated cell types.",
    "06_phenotype_membership": "Each panel shows the full GMM posterior probability for one phenotype. High values indicate stronger membership; transitional cells can have distributed probability across phenotypes.",
    "07_yap_localization": "Raw uncorrected log2 nuclear/perinuclear YAP ratios shown post hoc. YAP was excluded from preprocessing, UMAP and GMM. Perinuclear signal is a DAPI-excluded ring proxy, not a membrane-segmented whole-cell cytoplasm.",
    "08_gmm_model_selection": "K is the number of GMM components. Penalized BIC combines model fit with penalties for complexity and small or fragmented clusters; lower is better. Selected K is the minimum penalized BIC and is not proof of biological cell types.",
    "09_yap_by_phenotype": "Raw uncorrected log2 nuclear/perinuclear YAP ratios grouped by DAPI-defined dominant phenotype. YAP was measured only after clustering and did not define the phenotypes.",
    "10_ha_morphology_effect_summary": "Each HA condition is compared with the control at the same seeding density. Circles are density-specific effects in control-IQR units, diamonds are medians across the four density strata, and horizontal segments span the observed density-specific range.",
    "11_ha_density_specific_morphology_effects": "Every value compares an HA group with control at the same seeding density. Red indicates higher and blue lower than the matched control in control-IQR units. Density is a matching stratum, not the primary outcome.",
    "12_ha_yap_effect_summary": "Change in raw log2 nuclear/perinuclear YAP relative to the density-matched control. YAP is post-hoc biological characterization and was excluded from the morphology model.",
    "13_umap_experimental_groups": "Experimental groups are overlaid post hoc on the DAPI-only UMAP. Group labels were excluded from preprocessing, UMAP and GMM. Coordinates were rigidly rotated for display only.",
    "14_umap_dominant_phenotypes": "Colors show dominant GMM phenotype using DAPI-derived morphology and neighborhood features with DAPI intensity disabled. YAP and treatment labels were excluded from preprocessing, UMAP and GMM. Centroid labels aid reading. Rotation is rigid and does not alter distances, posterior probabilities or assignments.",
    "15_umap_phenotype_probabilities": "Full posterior probability for every GMM component. These panels reveal uncertainty and overlap that a hard dominant-phenotype map hides. YAP and treatment labels were excluded from the model.",
    "16_phenotype_composition_counts": "Stacked bars show actual cell counts for each dominant phenotype. Bar height therefore varies with the observed number of cells. One image field is available per condition; results are descriptive rather than replicate-level inference.",
    "17_yap_by_condition_boxplots": "Raw uncorrected log2 nuclear/perinuclear YAP ratio by HA group and seeding density. YAP was measured post hoc and excluded from UMAP and GMM. The perinuclear ring is a sampling proxy.",
    "18_yap_by_phenotype_boxplots": "Raw uncorrected log2 nuclear/perinuclear YAP ratio by dominant DAPI-defined phenotype. This characterizes phenotypes after clustering and does not use YAP to create the phenotype labels.",
}
# ========================= END SETTINGS =====================================


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phenotype_order(cells: pd.DataFrame) -> list[str]:
    return sorted(cells["dominant_phenotype"].dropna().unique(), key=lambda x: int(str(x).split()[-1]))


def phenotype_colors(names: list[str]) -> dict[str, tuple]:
    if len(names) > len(PHENOTYPE_PALETTE):
        cmap = plt.get_cmap("tab20")
        return {name: cmap(index % cmap.N) for index, name in enumerate(names)}
    return {name: PHENOTYPE_PALETTE[index] for index, name in enumerate(names)}


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": TICK_SIZE,
        "axes.titlesize": TITLE_SIZE, "axes.labelsize": AXIS_SIZE,
        "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE, "figure.facecolor": "white",
        "axes.facecolor": "white", "axes.spines.top": True,
        "axes.spines.right": True, "axes.linewidth": 1.2,
        "savefig.facecolor": "white",
    })


def title(fig, key: str) -> None:
    prefix = f"{ACTIVE_MAGNIFICATION} · " if ACTIVE_MAGNIFICATION else ""
    fig.suptitle(prefix + TITLES[key], fontsize=TITLE_SIZE, fontweight="bold", y=.96)


def save(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in SAVE_FORMATS:
        metadata = None
        if extension == "svg":
            metadata = {
                "Title": fig._suptitle.get_text() if fig._suptitle else stem,
                "Description": SVG_NOTES.get(
                    stem,
                    "Presentation figure generated from existing single-cell result tables. "
                    "See the analysis README and audit JSON for methods and limitations.",
                ),
                "Creator": "Ekin_DAPI_YAP/make_ppt_figures.py",
            }
        fig.savefig(output_dir / f"{stem}.{extension}", dpi=DPI, bbox_inches=None,
                    metadata=metadata)
    plt.close(fig)


def oriented_umap(cells: pd.DataFrame) -> np.ndarray:
    """Rigidly rotate UMAP so its longest global axis is horizontal.

    Rotation/reflection changes presentation only: all pairwise distances and
    all GMM results remain identical. Signs are anchored to the original axes
    so repeated runs of this plotting script do not randomly flip the map.
    """
    xy = cells[["umap_1", "umap_2"]].to_numpy(float)
    centered = xy - np.nanmean(xy, axis=0)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    basis = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    rotated = centered @ basis
    for index in range(2):
        reference = centered[:, index]
        correlation = np.corrcoef(rotated[:, index], reference)[0, 1]
        if np.isfinite(correlation) and correlation < 0:
            rotated[:, index] *= -1
    return rotated


def square_limits(xy: np.ndarray):
    midpoint = (xy.min(axis=0) + xy.max(axis=0)) / 2
    span = max(float(np.ptp(xy, axis=0).max()), 1e-6) * 1.08
    return [(center-span/2, center+span/2) for center in midpoint]


def plot_umap(cells, output_dir, color_by="phenotype", stem_override=None,
              include_phenotype_notes=True):
    key = "phenotype_umap" if color_by == "phenotype" else "group_umap"
    stem = stem_override or ("01_umap_phenotypes" if color_by == "phenotype"
                             else "02_umap_experimental_groups")
    fig, ax = plt.subplots(figsize=SCATTER_CANVAS)
    fig.subplots_adjust(left=.14, right=.97, bottom=.27, top=.86)
    xy = oriented_umap(cells)
    limits = square_limits(xy)
    if color_by == "phenotype":
        names = phenotype_order(cells)
        colors = phenotype_colors(names)
        for name in names:
            mask = cells.dominant_phenotype.eq(name).to_numpy()
            ax.scatter(xy[mask, 0], xy[mask, 1], s=POINT_SIZE, alpha=POINT_ALPHA,
                       color=colors[name], edgecolors="none")
            center = np.nanmedian(xy[mask], axis=0)
            short_name = name.replace("Phenotype ", "P")
            ax.scatter(center[0], center[1], s=185, color=colors[name],
                       edgecolor="white", linewidth=1.8, zorder=5)
            ax.text(center[0], center[1], short_name, ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white", zorder=6,
                    path_effects=[pe.withStroke(linewidth=1.3, foreground="#222222")])
        handles = []
        for name in names:
            label = name.replace("Phenotype ", "P")
            if include_phenotype_notes and PHENOTYPE_NOTES.get(name):
                label += f" · {PHENOTYPE_NOTES[name]}"
            handles.append(Line2D([], [], marker="o", linestyle="none", color=colors[name],
                                  markersize=8, label=label))
        legend_title = None
    else:
        for group in GROUP_ORDER:
            mask = cells.experimental_group_label.eq(group).to_numpy()
            ax.scatter(xy[mask, 0], xy[mask, 1], s=POINT_SIZE, alpha=.68,
                       color=GROUP_COLORS[group], edgecolors="none")
        handles = [Line2D([], [], marker="o", linestyle="none", color=GROUP_COLORS[g],
                          markersize=8, label=GROUP_LABELS[g]) for g in GROUP_ORDER]
        legend_title = None
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2", xlim=limits[0], ylim=limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.legend(handles=handles, title=legend_title, loc="lower center",
              bbox_to_anchor=(.5, -.29), ncol=min(len(handles), 6), frameon=False,
              title_fontsize=LEGEND_SIZE + 1)
    title(fig, key)
    save(fig, output_dir, stem)


def plot_density_umap(cells, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=CANVAS, sharex=True, sharey=True)
    xy = oriented_umap(cells)
    limits = square_limits(xy)
    for ax, group in zip(axes, GROUP_ORDER):
        for density in DENSITY_ORDER:
            mask = (cells.experimental_group_label.eq(group)
                    & cells.seeding_density_cells_per_cm2.eq(density)).to_numpy()
            ax.scatter(xy[mask, 0], xy[mask, 1], s=10, alpha=.70,
                       color=DENSITY_COLORS[density], edgecolors="none")
        ax.set(xlim=limits[0], ylim=limits[1], xlabel="UMAP 1")
        ax.set_box_aspect(1)
        ax.set_title(GROUP_LABELS[group], fontsize=15, fontweight="bold", pad=8)
    axes[0].set_ylabel("UMAP 2")
    handles = [Line2D([], [], marker="o", linestyle="none", color=DENSITY_COLORS[d], markersize=7,
                      label=f"{d/1000:g}k") for d in DENSITY_ORDER]
    fig.legend(handles=handles, title="Cells/cm²", ncol=4, loc="lower center",
               bbox_to_anchor=(.5, .02), frameon=False)
    fig.subplots_adjust(left=.07, right=.98, bottom=.15, top=.80, wspace=.15)
    title(fig, "density_umap")
    save(fig, output_dir, "03_umap_seeding_density")


def plot_composition(cells, output_dir, stem="04_phenotype_composition"):
    names = phenotype_order(cells)
    colors = phenotype_colors(names)
    fig, axes = plt.subplots(1, 3, figsize=CANVAS, sharey=True)
    max_count = cells.groupby(
        ["experimental_group_label", "seeding_density_cells_per_cm2"]
    ).size().max()
    for ax, group in zip(axes, GROUP_ORDER):
        sub = cells[cells.experimental_group_label.eq(group)]
        counts = (sub.groupby(["seeding_density_cells_per_cm2", "dominant_phenotype"]).size()
                  .unstack(fill_value=0).reindex(index=DENSITY_ORDER, columns=names, fill_value=0))
        bottom = np.zeros(4)
        x = np.arange(4)
        for name in names:
            values = counts[name].to_numpy(dtype=float)
            ax.bar(x, values, bottom=bottom, width=.78, color=colors[name])
            bottom += values
        ax.set_xticks(x, [f"{d/1000:g}k" for d in DENSITY_ORDER])
        ax.set_xlabel(GROUP_LABELS[group], fontsize=11, fontweight="bold", labelpad=8)
        for xpos, total in zip(x, counts.sum(axis=1)):
            ax.text(xpos, float(total) + max_count * .018, f"n={int(total)}",
                    ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    axes[0].set_ylim(0, max_count * 1.16)
    axes[0].set_ylabel("Cell count")
    handles = [Patch(facecolor=colors[name], label=f"P{name.split()[-1]}") for name in names]
    fig.legend(handles=handles, ncol=len(names), loc="lower center",
               bbox_to_anchor=(.5, -.005), frameon=False)
    fig.text(.5, .075, "Seeding density (cells/cm²)", ha="center", fontsize=12)
    fig.subplots_adjust(left=.07, right=.98, bottom=.23, top=.84, wspace=.13)
    title(fig, "composition")
    save(fig, output_dir, stem)


def plot_composition_fraction(cells, output_dir, stem="16_phenotype_composition_fraction"):
    """Backward-compatible wrapper; bars now use actual cell counts."""
    plot_composition(cells, output_dir, stem=stem)


def plot_heatmap(cells, output_dir):
    names = phenotype_order(cells)
    columns = [column for column in HEATMAP_FEATURES if column in cells]
    means = cells.groupby("dominant_phenotype")[columns].mean().reindex(names)
    z = (means - cells[columns].mean()) / cells[columns].std(ddof=0).replace(0, np.nan)
    fig, ax = plt.subplots(figsize=CANVAS)
    image = ax.imshow(z.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    ax.set_yticks(range(len(names)), [f"P{name.split()[-1]}" for name in names])
    ax.set_xticks(range(len(columns)), [HEATMAP_FEATURES[c] for c in columns], rotation=42, ha="right")
    ax.tick_params(axis="x", labelsize=10)
    bar = fig.colorbar(image, ax=ax, fraction=.025, pad=.02)
    bar.set_label("Difference from pooled mean (SD)", fontsize=13)
    ax.set_xlabel("Morphology and neighborhood features", labelpad=9)
    fig.subplots_adjust(left=.10, right=.93, bottom=.28, top=.84)
    title(fig, "heatmap")
    save(fig, output_dir, "05_phenotype_feature_profiles")


def plot_membership(cells, output_dir, stem="06_phenotype_membership"):
    columns = sorted([c for c in cells if c.startswith("P_phenotype_")], key=lambda c: int(c.split("_")[-1]))
    ncols, nrows = 4, math.ceil(len(columns)/4)
    fig, axes = plt.subplots(nrows, ncols, figsize=CANVAS, sharex=True, sharey=True, squeeze=False)
    xy = oriented_umap(cells)
    limits = square_limits(xy)
    color = (0/255, 98/255, 155/255)
    cmap = LinearSegmentedColormap.from_list("probability", [(*color, .04), (*color, 1.0)])
    for ax, column in zip(axes.flat, columns):
        ax.scatter(xy[:, 0], xy[:, 1], c=cells[column], s=7, cmap=cmap,
                   norm=Normalize(0, 1), edgecolors="none")
        ax.set_title(f"P{column.split('_')[-1]}", fontsize=14, fontweight="bold")
        ax.set(xlim=limits[0], ylim=limits[1])
        ax.set_box_aspect(1)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.flat[len(columns):]: ax.axis("off")
    cax = fig.add_axes([.92, .20, .015, .52])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap), cax=cax)
    bar.set_label("Posterior probability", fontsize=13)
    fig.subplots_adjust(left=.04, right=.90, bottom=.07, top=.84, wspace=.12, hspace=.22)
    title(fig, "membership")
    save(fig, output_dir, stem)


def plot_yap(cells, output_dir, stem="07_yap_localization"):
    column = "posthoc_yap_raw_log2_nuclear_perinuclear_ratio"
    fig, ax = plt.subplots(figsize=CANVAS)
    positions, arrays, colors, labels = [], [], [], []
    pos = 1
    for group in GROUP_ORDER:
        for density in DENSITY_ORDER:
            sub = cells[cells.experimental_group_label.eq(group) & cells.seeding_density_cells_per_cm2.eq(density)]
            values = pd.to_numeric(sub[column], errors="coerce").dropna().to_numpy()
            positions.append(pos); arrays.append(values); colors.append(GROUP_COLORS[group])
            labels.append(f"{density/1000:g}k")
            pos += 1
        pos += .7
    box = ax.boxplot(arrays, positions=positions, widths=.65, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(.72)
    ax.set_xticks(positions, labels, fontsize=10)
    ax.axhline(0, color="#555555", ls=":", lw=1.2)
    ax.set_ylabel("log₂(nuclear/perinuclear YAP)")
    handles = [Patch(facecolor=GROUP_COLORS[g], alpha=.72, label=GROUP_LABELS[g]) for g in GROUP_ORDER]
    ax.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(.5, -.24), frameon=False)
    ax.set_xlabel("Seeding density (cells/cm²)", labelpad=8)
    fig.subplots_adjust(left=.10, right=.98, bottom=.23, top=.84)
    title(fig, "yap")
    save(fig, output_dir, stem)


def plot_yap_by_phenotype(cells, output_dir, stem="09_yap_by_phenotype",
                          include_phenotype_notes=True):
    """Post-hoc YAP distribution for each DAPI-defined morphology phenotype."""
    column = "posthoc_yap_raw_log2_nuclear_perinuclear_ratio"
    names = phenotype_order(cells)
    colors = phenotype_colors(names)
    arrays, labels = [], []
    for name in names:
        sub = cells[cells.dominant_phenotype.eq(name)]
        values = pd.to_numeric(sub[column], errors="coerce").dropna().to_numpy()
        arrays.append(values)
        labels.append(f"P{name.split()[-1]}")
    fig, ax = plt.subplots(figsize=CANVAS)
    positions = np.arange(1, len(names) + 1)
    box = ax.boxplot(arrays, positions=positions, widths=.62, patch_artist=True,
                     showfliers=False, medianprops={"color": "#111111", "linewidth": 2.2})
    for patch, name in zip(box["boxes"], names):
        patch.set_facecolor(colors[name]); patch.set_alpha(.82)
    ax.axhline(0, color="#555555", ls=":", lw=1.2)
    ax.set_xticks(positions, labels, fontsize=11)
    ax.set_xlabel("Morphology phenotype", labelpad=10)
    ax.set_ylabel("log₂(nuclear/perinuclear YAP)")
    fig.subplots_adjust(left=.10, right=.97, bottom=.17, top=.84)
    title(fig, "yap_phenotype")
    save(fig, output_dir, stem)


def summarize_ha_effects(cells: pd.DataFrame) -> pd.DataFrame:
    """Return density-matched descriptive summaries for HA-centred plots.

    The standardized effect is a robust descriptive quantity:
    (condition median - matched-control median) / matched-control IQR.
    It is deliberately not labelled as a test statistic or uncertainty.
    """
    required = {
        "experimental_group_label",
        "seeding_density_cells_per_cm2",
        YAP_RAW_COLUMN,
        *HA_MORPHOLOGY_METRICS,
    }
    missing = sorted(required - set(cells))
    if missing:
        raise ValueError(f"Missing HA-effect columns: {missing}")

    rows = []
    for density in DENSITY_ORDER:
        density_cells = cells[cells["seeding_density_cells_per_cm2"].eq(density)]
        control = density_cells[density_cells["experimental_group_label"].eq("Ctrl")]
        for feature, (label, unit) in HA_MORPHOLOGY_METRICS.items():
            control_values = control[feature].dropna().astype(float)
            control_median = float(control_values.median()) if len(control_values) else np.nan
            control_iqr = (
                float(control_values.quantile(.75) - control_values.quantile(.25))
                if len(control_values)
                else np.nan
            )
            for group in GROUP_ORDER:
                values = density_cells.loc[
                    density_cells["experimental_group_label"].eq(group), feature
                ].dropna().astype(float)
                median = float(values.median()) if len(values) else np.nan
                q25 = float(values.quantile(.25)) if len(values) else np.nan
                q75 = float(values.quantile(.75)) if len(values) else np.nan
                effect = (
                    (median - control_median) / control_iqr
                    if np.isfinite(median) and np.isfinite(control_iqr) and control_iqr > 0
                    else np.nan
                )
                rows.append({
                    "experimental_group_label": group,
                    "seeding_density_cells_per_cm2": density,
                    "feature": feature,
                    "metric": label,
                    "unit": unit,
                    "n_cells": int(len(values)),
                    "median": median,
                    "q25": q25,
                    "q75": q75,
                    "control_median": control_median,
                    "control_iqr": control_iqr,
                    "control_referenced_effect": effect,
                })
    return pd.DataFrame(rows)


def summarize_ha_conditions(cells: pd.DataFrame, effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for density in DENSITY_ORDER:
        for group in GROUP_ORDER:
            effect_values = effects.loc[
                effects["seeding_density_cells_per_cm2"].eq(density)
                & effects["experimental_group_label"].eq(group),
                "control_referenced_effect",
            ].dropna().astype(float)
            deviation = float(np.sqrt(np.mean(np.square(effect_values)))) if len(effect_values) else np.nan
            subset = cells[
                cells["seeding_density_cells_per_cm2"].eq(density)
                & cells["experimental_group_label"].eq(group)
            ]
            yap = subset[YAP_RAW_COLUMN].dropna().astype(float)
            rows.append({
                "experimental_group_label": group,
                "seeding_density_cells_per_cm2": density,
                "n_morphology_cells": int(len(subset)),
                "morphology_deviation_from_control": deviation,
                "n_yap_available": int(len(yap)),
                "yap_raw_log2_ratio_median": float(yap.median()) if len(yap) else np.nan,
                "yap_raw_log2_ratio_q25": float(yap.quantile(.25)) if len(yap) else np.nan,
                "yap_raw_log2_ratio_q75": float(yap.quantile(.75)) if len(yap) else np.nan,
            })
    return pd.DataFrame(rows)


def plot_ha_effect_heatmap(effects: pd.DataFrame, output_dir: Path) -> None:
    groups = ("HA1", "HA2")
    metric_labels = [label for label, _ in HA_MORPHOLOGY_METRICS.values()]
    fig, axes = plt.subplots(1, 2, figsize=CANVAS, sharey=True)
    matrices = []
    for group in groups:
        matrix = np.full((len(metric_labels), len(DENSITY_ORDER)), np.nan)
        for row_index, label in enumerate(metric_labels):
            for col_index, density in enumerate(DENSITY_ORDER):
                values = effects.loc[
                    effects["experimental_group_label"].eq(group)
                    & effects["seeding_density_cells_per_cm2"].eq(density)
                    & effects["metric"].eq(label),
                    "control_referenced_effect",
                ]
                if len(values):
                    matrix[row_index, col_index] = float(values.iloc[0])
        matrices.append(matrix)
    finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices])
    limit = max(1.0, min(1.75, float(np.ceil(np.max(np.abs(finite)) * 4) / 4)))
    image = None
    for ax, group, matrix in zip(axes, groups, matrices):
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_title(GROUP_LABELS[group], fontsize=17, fontweight="bold", pad=12)
        ax.set_xticks(range(len(DENSITY_ORDER)), ["2.5k", "5k", "7.5k", "10k"])
        ax.set_yticks(range(len(metric_labels)), metric_labels)
        ax.set_xlabel("Seeding density (cells/cm²)")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                if np.isfinite(value):
                    color = "white" if abs(value) > limit * .58 else "#222222"
                    ax.text(col, row, f"{value:+.2f}", ha="center", va="center",
                            fontsize=12, fontweight="bold", color=color)
    axes[0].set_ylabel("DAPI morphology metric")
    colorbar = fig.colorbar(image, ax=axes, fraction=.025, pad=.035)
    colorbar.set_label("Shift from matched Ctrl (Ctrl IQR units)", fontsize=13)
    fig.subplots_adjust(left=.16, right=.90, bottom=.16, top=.84, wspace=.16)
    title(fig, "ha_density_effects")
    save(fig, output_dir, "11_ha_density_specific_morphology_effects")


def plot_ha_morphology_trends(effects: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=CANVAS)
    x = np.asarray(DENSITY_ORDER, dtype=float) / 1000.0
    for ax, (feature, (label, unit)) in zip(axes.flat, HA_MORPHOLOGY_METRICS.items()):
        for group in GROUP_ORDER:
            subset = effects[
                effects["feature"].eq(feature)
                & effects["experimental_group_label"].eq(group)
            ].set_index("seeding_density_cells_per_cm2").reindex(DENSITY_ORDER)
            median = subset["median"].to_numpy(float)
            q25 = subset["q25"].to_numpy(float)
            q75 = subset["q75"].to_numpy(float)
            yerr = np.vstack([median - q25, q75 - median])
            ax.errorbar(x, median, yerr=yerr, color=GROUP_COLORS[group], marker="o",
                        ms=6, lw=2, capsize=3, label=GROUP_LABELS[group])
        ax.set_title(label, fontsize=16, fontweight="bold")
        ax.set_ylabel(unit)
        ax.set_xticks(x, ["2.5k", "5k", "7.5k", "10k"])
        ax.grid(axis="y", color="#E3E3E3", lw=.8)
    for ax in axes[-1, :]:
        ax.set_xlabel("Seeding density (cells/cm²)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, .025), fontsize=12)
    fig.subplots_adjust(left=.08, right=.98, bottom=.22, top=.80, hspace=.42, wspace=.30)
    title(fig, "ha_trends")
    save(fig, output_dir, "11_ha_morphology_trends")


def plot_ha_morphology_yap_relation(condition_summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=CANVAS)
    for group in GROUP_ORDER:
        subset = condition_summary[
            condition_summary["experimental_group_label"].eq(group)
        ].sort_values("seeding_density_cells_per_cm2")
        ax.plot(subset["morphology_deviation_from_control"], subset["yap_raw_log2_ratio_median"],
                color=GROUP_COLORS[group], lw=2, alpha=.8)
        ax.scatter(subset["morphology_deviation_from_control"], subset["yap_raw_log2_ratio_median"],
                   s=115, color=GROUP_COLORS[group], edgecolor="white", linewidth=1.2,
                   label=GROUP_LABELS[group], zorder=3)
        for _, row in subset.iterrows():
            density = row["seeding_density_cells_per_cm2"] / 1000
            label = f"{density:g}k"
            offset = (7, 7)
            if group == "Ctrl":
                offset = {
                    2.5: (12, 10),
                    5.0: (12, -14),
                    7.5: (12, 10),
                    10.0: (12, -18),
                }[density]
            ax.annotate(label,
                        (row["morphology_deviation_from_control"], row["yap_raw_log2_ratio_median"]),
                        xytext=offset, textcoords="offset points", fontsize=10,
                        color=GROUP_COLORS[group], fontweight="bold")
    ax.axhline(0, color="#777777", ls=":", lw=1.2)
    ax.set_xlabel("Control-referenced morphology deviation (RMS effect)")
    ax.set_ylabel("Median raw log₂(YAP nuclear/perinuclear)")
    ax.grid(color="#E3E3E3", lw=.8)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, .5), frameon=False)
    fig.subplots_adjust(left=.10, right=.76, bottom=.16, top=.84)
    title(fig, "ha_yap_relation")
    save(fig, output_dir, "12_ha_morphology_yap_relation")


def summarize_ha_across_density_strata(effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in ("HA1", "HA2"):
        for feature, (label, _) in HA_MORPHOLOGY_METRICS.items():
            subset = effects[
                effects["experimental_group_label"].eq(group)
                & effects["feature"].eq(feature)
            ].sort_values("seeding_density_cells_per_cm2")
            values = subset["control_referenced_effect"].dropna().to_numpy(float)
            rows.append({
                "experimental_group_label": group,
                "feature": feature,
                "metric": label,
                "n_density_strata": int(len(values)),
                "median_effect": float(np.median(values)) if len(values) else np.nan,
                "minimum_effect": float(np.min(values)) if len(values) else np.nan,
                "maximum_effect": float(np.max(values)) if len(values) else np.nan,
                "n_lower_than_control": int(np.sum(values < 0)),
                "n_higher_than_control": int(np.sum(values > 0)),
            })
    return pd.DataFrame(rows)


def summarize_yap_effects(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for density in DENSITY_ORDER:
        density_cells = cells[cells["seeding_density_cells_per_cm2"].eq(density)]
        control = density_cells.loc[
            density_cells["experimental_group_label"].eq("Ctrl"), YAP_RAW_COLUMN
        ].dropna().astype(float)
        control_median = float(control.median()) if len(control) else np.nan
        for group in ("HA1", "HA2"):
            values = density_cells.loc[
                density_cells["experimental_group_label"].eq(group), YAP_RAW_COLUMN
            ].dropna().astype(float)
            group_median = float(values.median()) if len(values) else np.nan
            rows.append({
                "experimental_group_label": group,
                "seeding_density_cells_per_cm2": density,
                "n_yap_available": int(len(values)),
                "group_median_raw_log2_ratio": group_median,
                "control_median_raw_log2_ratio": control_median,
                "effect_vs_density_matched_control": group_median - control_median,
            })
    return pd.DataFrame(rows)


def plot_ha_main_morphology(effects: pd.DataFrame, output_dir: Path) -> None:
    groups = ("HA1", "HA2")
    metrics = [label for label, _ in HA_MORPHOLOGY_METRICS.values()]
    y = np.arange(len(metrics), dtype=float)
    offsets = np.linspace(-.15, .15, len(DENSITY_ORDER))
    finite = effects.loc[
        effects["experimental_group_label"].isin(groups), "control_referenced_effect"
    ].dropna().to_numpy(float)
    limit = max(1.0, float(np.ceil(np.max(np.abs(finite)) * 4) / 4))
    fig, axes = plt.subplots(1, 2, figsize=CANVAS, sharey=True)
    for ax, group in zip(axes, groups):
        for row, metric in enumerate(metrics):
            subset = effects[
                effects["experimental_group_label"].eq(group)
                & effects["metric"].eq(metric)
            ].sort_values("seeding_density_cells_per_cm2")
            values = subset["control_referenced_effect"].to_numpy(float)
            valid = values[np.isfinite(values)]
            if not len(valid):
                continue
            ax.hlines(row, np.min(valid), np.max(valid), color=GROUP_COLORS[group], lw=3, alpha=.45)
            densities = subset["seeding_density_cells_per_cm2"].to_numpy(float)
            for index, (density, value) in enumerate(zip(densities, values)):
                if np.isfinite(value):
                    ax.scatter([value], [row + offsets[index]], s=58,
                               color=DENSITY_COLORS[density], alpha=.82,
                               edgecolor="white", linewidth=.7, zorder=2)
            median = float(np.median(valid))
            ax.scatter([median], [row], s=165, marker="D", color=GROUP_COLORS[group],
                       edgecolor="white", linewidth=1.2, zorder=3)
            ax.text(median + (.045 if median >= 0 else -.045), row, f"{median:+.2f}",
                    ha="left" if median >= 0 else "right", va="center", fontsize=11,
                    color=GROUP_COLORS[group], fontweight="bold")
        ax.axvline(0, color="#222222", lw=1.5, ls=":")
        ax.set_xlim(-limit, limit)
        ax.set_yticks(y, metrics)
        ax.invert_yaxis()
        ax.set_title(GROUP_LABELS[group], fontsize=18, fontweight="bold", pad=12)
        ax.set_xlabel("HA effect relative to matched Ctrl (Ctrl IQR units)")
        ax.grid(axis="x", color="#E5E5E5", lw=.8)
    axes[0].set_ylabel("DAPI morphology metric")
    density_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=8,
               markerfacecolor=DENSITY_COLORS[density], markeredgecolor="white",
               label=f"{density / 1000:g}k")
        for density in DENSITY_ORDER
    ]
    fig.legend(handles=density_handles, title="Seeding density (cells/cm²)",
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, .025),
               fontsize=11, title_fontsize=11.5)
    fig.subplots_adjust(left=.16, right=.97, bottom=.18, top=.84, wspace=.16)
    title(fig, "ha_main_morphology")
    save(fig, output_dir, "10_ha_morphology_effect_summary")


def plot_ha_main_yap(yap_effects: pd.DataFrame, output_dir: Path) -> None:
    groups = ("HA1", "HA2")
    y = np.arange(len(groups), dtype=float)
    offsets = np.linspace(-.14, .14, len(DENSITY_ORDER))
    finite = yap_effects["effect_vs_density_matched_control"].dropna().to_numpy(float)
    limit = max(.30, float(np.ceil(np.max(np.abs(finite)) * 20) / 20 + .05))
    fig, ax = plt.subplots(figsize=CANVAS)
    for row, group in enumerate(groups):
        subset = yap_effects[
            yap_effects["experimental_group_label"].eq(group)
        ].sort_values("seeding_density_cells_per_cm2")
        values = subset["effect_vs_density_matched_control"].to_numpy(float)
        densities = subset["seeding_density_cells_per_cm2"].to_numpy(float)
        valid = values[np.isfinite(values)]
        ax.hlines(row, np.min(valid), np.max(valid), color=GROUP_COLORS[group], lw=5, alpha=.40)
        for index, (density, value) in enumerate(zip(densities, values)):
            if np.isfinite(value):
                ax.scatter([value], [row + offsets[index]], s=92,
                           color=DENSITY_COLORS[density], alpha=.88,
                           edgecolor="white", linewidth=.9, zorder=2)
        median = float(np.median(valid))
        ax.scatter([median], [row], s=230, marker="D", color=GROUP_COLORS[group],
                   edgecolor="white", linewidth=1.4, zorder=3)
        ax.text(median + (.015 if median >= 0 else -.015), row, f"{median:+.2f}",
                ha="left" if median >= 0 else "right", va="center", fontsize=13,
                color=GROUP_COLORS[group], fontweight="bold")
    ax.axvline(0, color="#222222", lw=1.6, ls=":")
    ax.set_xlim(-limit, limit)
    ax.set_yticks(y, [GROUP_LABELS[group] for group in groups])
    ax.invert_yaxis()
    ax.set_xlabel("HA effect on raw log₂(YAP nuclear/perinuclear)\n(relative to density-matched Ctrl)")
    ax.grid(axis="x", color="#E5E5E5", lw=.8)
    density_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=8,
               markerfacecolor=DENSITY_COLORS[density], markeredgecolor="white",
               label=f"{density / 1000:g}k")
        for density in DENSITY_ORDER
    ]
    fig.legend(handles=density_handles, title="Seeding density (cells/cm²)",
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, .03),
               fontsize=11, title_fontsize=11.5)
    fig.subplots_adjust(left=.23, right=.95, bottom=.20, top=.84)
    title(fig, "ha_main_yap")
    save(fig, output_dir, "12_ha_yap_effect_summary")


def make_ha_effect_figures(cells: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    effects = summarize_ha_effects(cells)
    conditions = summarize_ha_conditions(cells, effects)
    across_density = summarize_ha_across_density_strata(effects)
    yap_effects = summarize_yap_effects(cells)
    plot_ha_main_morphology(effects, output_dir)
    plot_ha_effect_heatmap(effects, output_dir)
    plot_ha_main_yap(yap_effects, output_dir)
    effects.to_csv(output_dir / "ha_control_referenced_effects.csv", index=False, encoding="utf-8-sig")
    conditions.to_csv(output_dir / "ha_condition_summary.csv", index=False, encoding="utf-8-sig")
    across_density.to_csv(output_dir / "ha_effect_summary_across_density_strata.csv", index=False,
                          encoding="utf-8-sig")
    yap_effects.to_csv(output_dir / "ha_yap_effects_vs_matched_control.csv", index=False,
                       encoding="utf-8-sig")
    return effects, conditions


def make_model_supplement_figures(cells: pd.DataFrame, output_dir: Path) -> None:
    """Add the current GMM/UMAP results without assigning biological names."""
    plot_umap(cells, output_dir, color_by="group",
              stem_override="13_umap_experimental_groups")
    plot_umap(cells, output_dir, color_by="phenotype",
              stem_override="14_umap_dominant_phenotypes",
              include_phenotype_notes=False)
    plot_membership(cells, output_dir, stem="15_umap_phenotype_probabilities")
    plot_composition(cells, output_dir,
                     stem="16_phenotype_composition_counts")
    plot_yap(cells, output_dir, stem="17_yap_by_condition_boxplots")
    plot_yap_by_phenotype(cells, output_dir,
                          stem="18_yap_by_phenotype_boxplots",
                          include_phenotype_notes=False)


def plot_gmm(selection, output_dir):
    fig, ax = plt.subplots(figsize=CANVAS)
    metric = selection["bic_penalized"]
    delta = metric - metric.min()
    selected = int(selection.loc[metric.idxmin(), "n_clusters"])
    ax.plot(selection.n_clusters, delta, color="#444444", lw=2.5, marker="o", ms=7)
    row = selection[selection.n_clusters.eq(selected)].iloc[0]
    ax.scatter([selected], [row.bic_penalized-metric.min()], s=180, color="#D000B5", zorder=3)
    ax.annotate(f"Selected K = {selected}", (selected, 0), xytext=(16, 24), textcoords="offset points",
                fontsize=15, fontweight="bold", arrowprops=dict(arrowstyle="->", lw=1.4))
    ax.set_xlabel("Number of GMM components (K)")
    ax.set_ylabel("Δ penalized BIC")
    ax.set_xticks(selection.n_clusters.astype(int))
    ax.grid(axis="y", color="#dddddd", lw=.8)
    fig.subplots_adjust(left=.10, right=.97, bottom=.16, top=.84)
    title(fig, "gmm")
    save(fig, output_dir, "08_gmm_model_selection")


def main(argv=None):
    global ACTIVE_MAGNIFICATION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path,
                        default=Path(__file__).resolve().parent / "outputs" / "composite_no_dapi_intensity" / "40x" / "all_fields")
    parser.add_argument("--figure-dir", type=Path, help="Default: <output-root>/figures_ppt")
    parser.add_argument(
        "--backup-existing",
        action="store_true",
        help="Keep the previous PNG/SVG/JSON files in a timestamped subfolder.",
    )
    parser.add_argument(
        "--ha-effects-only",
        action="store_true",
        help="Draw only the density-matched HA morphology/YAP summary figures.",
    )
    args = parser.parse_args(argv)
    root = args.output_root.resolve()
    ACTIVE_MAGNIFICATION = next(
        (part.lower() for part in root.parts if part.lower() == "40x"),
        "",
    )
    output_dir = (args.figure_dir or root / "figures_ppt").resolve()
    source = root / "tables" / "model_a_single_cell_results.csv"
    selection_source = root / "tables" / "gmm_model_selection.csv"
    before = {str(source): sha256(source), str(selection_source): sha256(selection_source)}
    if output_dir.exists() and any(output_dir.glob("*.png")):
        backup = None
        if args.backup_existing:
            backup = output_dir / ("previous_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
            backup.mkdir(parents=True)
        for path in output_dir.iterdir():
            if path.is_file() and path.suffix.lower() in {".png", ".svg", ".json"}:
                if backup is not None:
                    shutil.copy2(path, backup / path.name)
                path.unlink()
    cells = pd.read_csv(source)
    selection = pd.read_csv(selection_source)
    required = {
        "experimental_group_label",
        "seeding_density_cells_per_cm2",
        YAP_RAW_COLUMN,
        "umap_1",
        "umap_2",
        "dominant_phenotype",
        *HA_MORPHOLOGY_METRICS,
    }
    missing = sorted(required - set(cells))
    if missing:
        raise ValueError(f"Missing required result columns: {missing}")
    style()
    expected = []
    if not args.ha_effects_only:
        plot_umap(cells, output_dir, "phenotype")
        plot_umap(cells, output_dir, "group")
        plot_density_umap(cells, output_dir)
        plot_composition(cells, output_dir)
        plot_heatmap(cells, output_dir)
        plot_membership(cells, output_dir)
        plot_yap(cells, output_dir)
        plot_gmm(selection, output_dir)
        plot_yap_by_phenotype(cells, output_dir)
        expected.extend([f"{index:02d}_{name}.png" for index, name in enumerate(
            ("umap_phenotypes", "umap_experimental_groups", "umap_seeding_density",
             "phenotype_composition", "phenotype_feature_profiles", "phenotype_membership",
             "yap_localization", "gmm_model_selection", "yap_by_phenotype"), start=1)])
    make_ha_effect_figures(cells, output_dir)
    expected.extend([
        "10_ha_morphology_effect_summary.png",
        "11_ha_density_specific_morphology_effects.png",
        "12_ha_yap_effect_summary.png",
    ])
    if args.ha_effects_only:
        make_model_supplement_figures(cells, output_dir)
        expected.extend([
            "13_umap_experimental_groups.png",
            "14_umap_dominant_phenotypes.png",
            "15_umap_phenotype_probabilities.png",
            "16_phenotype_composition_counts.png",
            "17_yap_by_condition_boxplots.png",
            "18_yap_by_phenotype_boxplots.png",
        ])
    after = {path: sha256(Path(path)) for path in before}
    if before != after:
        raise AssertionError("A source table changed while drawing figures")
    missing_outputs = [name for name in expected if not (output_dir / name).exists()]
    if missing_outputs:
        raise AssertionError(f"Missing presentation figures: {missing_outputs}")
    audit = {"generated_at": datetime.now().isoformat(), "source_hashes": before,
             "source_tables_unchanged": True, "no_refitting": True, "no_remeasurement": True,
             "ha_effects_only": args.ha_effects_only,
             "canvas_inches": CANVAS, "dpi": DPI, "formats": SAVE_FORMATS,
             "figures": expected, "settings_file": str(Path(__file__).resolve())}
    (output_dir / "ppt_figure_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Generated {len(expected)} PPT-ready figures in {output_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parent
ANALYSIS_DIR = (
    WORKSPACE
    / "analysis_outputs"
    / "hepatocyte_confocal_audit"
    / "dapi_3d"
    / "yap"
    / "best_focus_2d_vs_3d"
)
OUTPUT_DIR = ANALYSIS_DIR / "presentation_figures"

CELLS_CSV = ANALYSIS_DIR / "matched_best_focus_2d_to_3d_nuclei.csv"
OVERALL_CSV = ANALYSIS_DIR / "overall_best_focus_correlations.csv"
CONDITION_CSV = ANALYSIS_DIR / "condition_best_focus_correlations.csv"

SAMPLE_LAYOUT = [
    ["D20D", "D30D", "D40D", "D50D"],
    [None, "D31D", "D41D", "D51D"],
    [None, "D33D", "D43D", "D53D"],
]
HEATMAP_SAMPLE_ORDER = [
    "D20D", "D30D", "D40D", "D50D",
    "D31D", "D41D", "D51D",
    "D33D", "D43D", "D53D",
]
DAY_LABELS = ["Day −3", "Day −2", "Day −1", "Day 0"]
ROW_LABELS = ["No HA", "Single HA", "Repeated HA"]

# Shared baseline is neutral gray. Within each regimen, later days use darker shades.
SAMPLE_COLORS = {
    "D20D": "#6B7280",
    "D30D": "#93C5FD",
    "D40D": "#3B82F6",
    "D50D": "#1D4ED8",
    "D31D": "#86EFAC",
    "D41D": "#22C55E",
    "D51D": "#15803D",
    "D33D": "#FCA5A5",
    "D43D": "#EF4444",
    "D53D": "#B91C1C",
}

SAMPLE_LEGEND_LABELS = {
    "D20D": "D20D · baseline",
    "D30D": "D30D · no HA · −2",
    "D40D": "D40D · no HA · −1",
    "D50D": "D50D · no HA · 0",
    "D31D": "D31D · single HA · −2",
    "D41D": "D41D · single HA · −1",
    "D51D": "D51D · single HA · 0",
    "D33D": "D33D · repeated HA · −2",
    "D43D": "D43D · repeated HA · −1",
    "D53D": "D53D · repeated HA · 0",
}

FEATURE_ORDER = [
    "Area",
    "Equivalent diameter",
    "Perimeter",
    "Minor axis",
    "Major axis",
    "Circularity",
    "Eccentricity",
    "Aspect ratio",
    "Solidity",
    "Extent",
    "DAPI mean (normalized)",
    "DAPI SD (normalized)",
    "DAPI CV",
    "DAPI IQR (normalized)",
    "DAPI intensity entropy",
]

PPT_FEATURE_ORDER = [
    "Area",
    "Perimeter",
    "Minor axis",
    "Circularity",
    "Eccentricity",
    "Solidity",
    "DAPI mean (normalized)",
    "DAPI intensity entropy",
]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Microsoft YaHei", "DejaVu Sans"],
        "font.size": 13,
        "axes.titlesize": 17,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.linewidth": 1.4,
    }
)


def save_both(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", bbox_inches="tight", facecolor="white")


def padded_limits(values: pd.Series, fraction: float = 0.06) -> tuple[float, float]:
    low = float(values.min())
    high = float(values.max())
    padding = max((high - low) * fraction, 1e-9)
    return low - padding, high + padding


def style_square_scatter_axis(ax: plt.Axes, x_limits: tuple[float, float], y_limits: tuple[float, float]) -> None:
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_box_aspect(1)
    ax.grid(True, color="#D1D5DB", linewidth=0.8, alpha=0.65)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
        spine.set_color("#111827")
    ax.tick_params(width=1.2, length=5)


def figure_a(cells: pd.DataFrame, overall: pd.DataFrame) -> None:
    stats = overall.set_index("feature").loc["area_um2"]
    x_limits = padded_limits(cells["area_um2"])
    y_limits = padded_limits(cells["volume_um3"])
    fig, ax = plt.subplots(figsize=(10.5, 9.5), constrained_layout=True)
    for sample in HEATMAP_SAMPLE_ORDER:
        subset = cells[cells["sample"] == sample]
        ax.scatter(
            subset["area_um2"],
            subset["volume_um3"],
            s=34,
            alpha=0.72,
            color=SAMPLE_COLORS[sample],
            edgecolors="white",
            linewidths=0.25,
            rasterized=True,
        )
    style_square_scatter_axis(ax, x_limits, y_limits)
    ax.set_xlabel("Best-focus 2D nuclear area (µm²)", labelpad=10)
    ax.set_ylabel("Matched 3D nuclear volume (µm³)", labelpad=10)
    ax.set_title("2D area reflects 3D nuclear size", fontweight="bold", pad=14)
    ax.text(
        0.03,
        0.97,
        f"Spearman ρ = {stats['spearman_rho']:.3f}\nPearson r = {stats['pearson_r']:.3f}\nn = {int(stats['n'])}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=14,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#9CA3AF", "alpha": 0.92},
    )
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=8, markerfacecolor="#6B7280", markeredgecolor="none", label="Baseline"),
        Line2D([0], [0], marker="o", linestyle="", markersize=8, markerfacecolor="#2563EB", markeredgecolor="none", label="No HA"),
        Line2D([0], [0], marker="o", linestyle="", markersize=8, markerfacecolor="#16A34A", markeredgecolor="none", label="Single HA"),
        Line2D([0], [0], marker="o", linestyle="", markersize=8, markerfacecolor="#DC2626", markeredgecolor="none", label="Repeated HA"),
        Line2D([0], [0], linestyle="", label="Light → dark: Day −2 → 0"),
    ]
    ax.legend(
        handles=handles,
        title="Condition",
        title_fontsize=12,
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        ncol=1,
        borderpad=0.7,
    )
    ax.text(-0.13, 1.08, "A", transform=ax.transAxes, fontsize=25, fontweight="bold", va="top")
    save_both(fig, "A_overall_best_focus_area_volume")
    plt.close(fig)


def figure_b(cells: pd.DataFrame, condition: pd.DataFrame) -> None:
    area_stats = condition[condition["feature"] == "area_um2"].set_index("sample")
    x_limits = padded_limits(cells["area_um2"])
    y_limits = padded_limits(cells["volume_um3"])
    fig, axes = plt.subplots(3, 4, figsize=(17.5, 13.2), constrained_layout=True)

    for row_index, row in enumerate(SAMPLE_LAYOUT):
        for column_index, sample in enumerate(row):
            ax = axes[row_index, column_index]
            if sample is None:
                ax.axis("off")
                continue
            subset = cells[cells["sample"] == sample]
            stats = area_stats.loc[sample]
            ax.scatter(
                subset["area_um2"],
                subset["volume_um3"],
                s=34,
                alpha=0.78,
                color=SAMPLE_COLORS[sample],
                edgecolors="white",
                linewidths=0.25,
                rasterized=True,
            )
            style_square_scatter_axis(ax, x_limits, y_limits)
            ax.set_title(sample, color=SAMPLE_COLORS[sample], fontweight="bold", pad=8)
            ax.text(
                0.04,
                0.96,
                f"n={int(stats['n'])}   ρ/r={stats['spearman_rho']:.2f}/{stats['pearson_r']:.2f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=11.5,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
            )
            if row_index == 2:
                ax.set_xlabel("2D nuclear area (µm²)")
            else:
                ax.set_xlabel("")
            if column_index == 0 or (column_index == 1 and row_index > 0):
                ax.set_ylabel("3D nuclear volume (µm³)")
            else:
                ax.set_ylabel("")

    for column_index, day in enumerate(DAY_LABELS):
        axes[0, column_index].text(
            0.5,
            1.22,
            day,
            transform=axes[0, column_index].transAxes,
            ha="center",
            va="bottom",
            fontsize=17,
            fontweight="bold",
            color="#111827",
        )

    row_colors = ["#1D4ED8", "#15803D", "#B91C1C"]
    for row_index, (label, color) in enumerate(zip(ROW_LABELS, row_colors)):
        anchor_ax = axes[row_index, 0] if row_index == 0 else axes[row_index, 1]
        anchor_ax.text(
            -0.34 if row_index == 0 else -0.42,
            0.5,
            label,
            transform=anchor_ax.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=17,
            fontweight="bold",
            color=color,
        )

    axes[0, 0].text(
        0.5,
        -0.14,
        "Shared baseline",
        transform=axes[0, 0].transAxes,
        ha="center",
        va="top",
        fontsize=12,
        color=SAMPLE_COLORS["D20D"],
        fontweight="bold",
    )
    fig.suptitle("Correlation within each condition", fontsize=21, fontweight="bold", y=1.015)
    fig.text(0.003, 1.01, "B", fontsize=25, fontweight="bold", va="top")
    save_both(fig, "B_grouped_best_focus_area_volume")
    plt.close(fig)


def heatmap_panel(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    title: str,
    sample_colors: dict[str, str],
) -> None:
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                ax.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9.5,
                    fontweight="semibold",
                    color="white" if abs(value) >= 0.55 else "#111827",
                )
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
    ax.tick_params(axis="both", labelsize=12)
    for tick in ax.get_xticklabels():
        tick.set_color(sample_colors[tick.get_text()])
        tick.set_fontweight("bold")
    ax.set_title(title, fontsize=19, fontweight="bold", pad=16)
    ax.set_xlabel("Sample / condition", labelpad=10)
    ax.set_ylabel("2D nuclear feature", labelpad=10)
    ax.grid(False)
    ax.set_aspect("equal")
    for boundary in [0.5, 3.5, 6.5]:
        ax.axvline(boundary, color="#111827", linewidth=2.0)
    return image


def figure_c(condition: pd.DataFrame) -> None:
    spearman = condition.pivot(index="feature_label", columns="sample", values="spearman_rho")
    spearman = spearman.reindex(index=PPT_FEATURE_ORDER, columns=HEATMAP_SAMPLE_ORDER)

    fig, ax = plt.subplots(figsize=(13.5, 10.5), constrained_layout=True)
    image = heatmap_panel(ax, spearman, "Feature–volume correlation (Spearman ρ)", SAMPLE_COLORS)
    colorbar = fig.colorbar(image, ax=ax, orientation="vertical", fraction=0.035, pad=0.03)
    colorbar.set_label("Correlation coefficient", fontsize=15)
    colorbar.ax.tick_params(labelsize=12)
    fig.text(0.003, 1.01, "C", fontsize=25, fontweight="bold", va="top")
    save_both(fig, "C_feature_volume_correlation_heatmaps")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(CELLS_CSV)
    overall = pd.read_csv(OVERALL_CSV)
    condition = pd.read_csv(CONDITION_CSV)

    figure_a(cells, overall)
    figure_b(cells, condition)
    figure_c(condition)

    readme = """# Slide-ready figures

- `A_overall_best_focus_area_volume`: all matched nuclei.
- `B_grouped_best_focus_area_volume`: 3 × 4 layout matching the requested reference image.
- `C_feature_volume_correlation_heatmaps`: Spearman and Pearson heatmaps with square cells.

Color mapping:

- No HA: blue, darker with later day.
- Single HA: green, darker with later day.
- Repeated HA: red, darker with later day.
- Shared D20D baseline: gray.

Both high-resolution PNG and editable SVG versions are provided.
"""
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(f"Saved presentation figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

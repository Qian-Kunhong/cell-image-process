"""Exact categorical colors and explicit QC masks; never changes model labels."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from skimage.segmentation import find_boundaries

DISPLAY_PMAX_THRESHOLD = 0.5


def low_confidence_rows(rows):
    """Display flag only: equality at the threshold is not flagged."""
    return rows[rows["gmm_max_posterior"] < DISPLAY_PMAX_THRESHOLD]


def qc_keep_mask(mask, rows, edge_buffer_px=0):
    """Return a copy containing result-table labels, rejecting any touch-edge label.

    Label IDs are preserved; an excluded raw segmentation is never erased on disk.
    This mask is for display/QC export, not a replacement for YAP occupancy masks.
    """
    labels = rows["label"].to_numpy(dtype=int)
    if len(labels) != len(np.unique(labels)):
        raise ValueError("Repeated label in single-image results")
    if np.any(labels <= 0) or not np.isin(labels, np.unique(mask)).all():
        raise ValueError("Result-table label absent from segmentation")
    if edge_buffer_px < 0:
        raise ValueError("edge_buffer_px must be nonnegative")
    width = edge_buffer_px + 1
    edge = np.unique(np.concatenate([
        mask[:width].ravel(), mask[-width:].ravel(),
        mask[:, :width].ravel(), mask[:, -width:].ravel(),
    ]))
    if np.isin(labels, edge[edge > 0]).any():
        raise ValueError("Touch-border nucleus appears in phenotype results; inspect QC before plotting")
    return np.where(np.isin(mask, labels), mask, 0).astype(mask.dtype)


def categorical_rgb(mask, rows, colors):
    """Opaque RGB labels: exactly the same colors as the model's global legend."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for row in rows.itertuples(index=False):
        rgb[mask == int(row.label)] = np.round(
            255 * np.asarray(colors[str(row.dominant_phenotype)][:3])
        ).astype(np.uint8)
    return rgb


def save_clear_phenotypes(dapi, raw_mask, rows, colors, title, path, qc_mask_path, edge_buffer_px=0):
    clean = qc_keep_mask(raw_mask, rows, edge_buffer_px=edge_buffer_px)
    rgb = categorical_rgb(clean, rows, colors)
    low = low_confidence_rows(rows)
    excluded = len(np.setdiff1d(np.unique(raw_mask[raw_mask > 0]), rows["label"]))
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    axes[0].imshow(dapi, cmap="gray", interpolation="nearest")
    boundary = find_boundaries(clean, mode="inner")
    outlines = np.zeros((*clean.shape, 4), dtype=float)
    outlines[boundary, :3] = rgb[boundary] / 255.0
    outlines[boundary, 3] = 1.0
    axes[0].imshow(outlines, interpolation="nearest")
    axes[0].set_title("DAPI only + QC-kept outlines\nWhite cross with black outline = uncertain assignment")
    cross_effects = [pe.Stroke(linewidth=3.2, foreground="black"), pe.Normal()]
    if len(low):
        crosses = axes[0].scatter(
            low["centroid_col_px"], low["centroid_row_px"],
            marker="x", s=26, color="white", linewidths=1.4,
        )
        crosses.set_path_effects(cross_effects)
    axes[1].imshow(rgb, interpolation="nearest")
    axes[1].set_title("Opaque phenotype mask: exact legend colors\nNumbers are phenotype IDs, not cell IDs")
    for row in rows.itertuples(index=False):
        axes[1].text(
            row.centroid_col_px, row.centroid_row_px, str(row.dominant_phenotype_rank),
            ha="center", va="center", fontsize=5.5, color="white",
            path_effects=[pe.Stroke(linewidth=1.6, foreground="black"), pe.Normal()],
        )
    names = sorted(colors, key=lambda name: int(name.split()[-1]))
    handles = [Patch(
        facecolor=colors[name], edgecolor="black", linewidth=.4,
        label=f"{name} (n={int(rows.dominant_phenotype.eq(name).sum())})",
    ) for name in names]
    cross_key = Line2D(
        [], [], marker="x", color="white", markeredgewidth=1.4,
        markersize=7, linestyle="None",
        label=f"Uncertain assignment\nPmax < {DISPLAY_PMAX_THRESHOLD:.2f} (n={len(low)})\n(crosses on left panel)",
        path_effects=cross_effects,
    )
    handles.append(cross_key)
    axes[1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title + f"\nQC-kept nuclei: {len(rows)}; excluded raw labels: {excluded} (not colored)", fontsize=13)
    fig.subplots_adjust(left=.01, right=.83, bottom=.02, top=.88, wspace=.03)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    qc_mask_path = Path(qc_mask_path)
    qc_mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(qc_mask_path, clean)

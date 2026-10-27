from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, wilcoxon
from skimage import exposure, feature, filters, measure, morphology, segmentation


WORKSPACE = Path(__file__).resolve().parent
ROOT = Path(
    r"F:\Naruchit data\1_Data of published manuscripts\3_Hepatocyte differentiation"
    r"\03_Raw data\Immunostaining\YAP quantification"
)
OUTPUT_DIR = WORKSPACE / "analysis_outputs" / "hepatocyte_confocal_audit" / "yap_crowding"
PIXEL_UM = 140.624 / 800
LOCAL_RADIUS_PX = 80
KNOWN_DUPLICATE = ("D43D", 4)


def stage_diameter(sample: str) -> tuple[int, int]:
    stage = int(sample[1])
    if stage <= 3:
        return 47, 100
    if stage == 4:
        return 40, 100
    return 40, 75


def segment_nuclei(image: np.ndarray, sample: str) -> np.ndarray:
    image = image.astype(np.float32)
    if image.max() > 1:
        image /= np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else image.max()
    smooth = filters.gaussian(image, sigma=1.2, preserve_range=True)
    threshold = min(float(filters.threshold_li(smooth)), 0.5)
    mask = smooth > threshold
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.remove_small_objects(mask, min_size=120)

    distance = ndi.distance_transform_edt(mask)
    peaks = feature.peak_local_max(
        filters.gaussian(distance, sigma=1.0, preserve_range=True),
        min_distance=26,
        labels=mask,
        exclude_border=False,
    )
    markers = np.zeros(mask.shape, dtype=np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    labels = segmentation.watershed(-distance, markers, mask=mask)

    min_d, max_d = stage_diameter(sample)
    min_area = np.pi * (min_d / 2) ** 2
    max_area = np.pi * (max_d / 2) ** 2
    keep = []
    height, width = labels.shape
    for region in measure.regionprops(labels):
        y0, x0, y1, x1 = region.bbox
        touches_border = y0 == 0 or x0 == 0 or y1 == height or x1 == width
        if min_area <= region.area <= max_area and not touches_border:
            keep.append(region.label)
    labels = np.where(np.isin(labels, keep), labels, 0)
    labels = measure.label(labels > 0)
    return labels


def cell_foreground(yap: np.ndarray) -> np.ndarray:
    image = yap.astype(np.float32)
    if image.max() > 1:
        image /= image.max()
    smooth = filters.gaussian(image, sigma=5, preserve_range=True)
    threshold = float(filters.threshold_li(smooth)) * 0.8
    mask = smooth > threshold
    return morphology.remove_small_objects(mask, min_size=200)


def measure_field(sample: str, field_number: int, dapi_path: Path, yap_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    dapi = tifffile.imread(dapi_path)
    yap = tifffile.imread(yap_path)
    labels = segment_nuclei(dapi, sample)
    foreground = cell_foreground(yap)

    expanded_1 = segmentation.expand_labels(labels, distance=1)
    expanded_6 = segmentation.expand_labels(labels, distance=6)
    ring_labels = np.where((expanded_6 > 0) & (expanded_1 == 0) & foreground, expanded_6, 0)

    props = measure.regionprops(labels, intensity_image=yap.astype(np.float32))
    centroids = np.asarray([region.centroid for region in props], dtype=float)
    if len(centroids) > 1:
        tree = cKDTree(centroids)
        distances, _ = tree.query(centroids, k=2)
        nearest_px = distances[:, 1]
        neighbor_count = np.asarray([len(v) - 1 for v in tree.query_ball_point(centroids, LOCAL_RADIUS_PX)])
    else:
        nearest_px = np.full(len(props), np.nan)
        neighbor_count = np.zeros(len(props), dtype=int)

    nuclear_mask = labels > 0
    kernel = morphology.disk(LOCAL_RADIUS_PX).astype(np.float32)
    local_fraction_map = fftconvolve(
        nuclear_mask.astype(np.float32), kernel, mode="same"
    ) / kernel.sum()

    rows = []
    for index, region in enumerate(props):
        label = region.label
        ring_values = yap[ring_labels == label].astype(np.float32)
        if ring_values.size < 20 or float(ring_values.mean()) <= 0:
            continue
        nucleus_mean = float(region.mean_intensity)
        ring_mean = float(ring_values.mean())
        y, x = region.centroid
        edge_distance_px = min(y, x, labels.shape[0] - 1 - y, labels.shape[1] - 1 - x)
        rows.append(
            {
                "sample": sample,
                "field": field_number,
                "field_id": f"{sample}_{field_number:02d}",
                "object_label": label,
                "centroid_y_px": y,
                "centroid_x_px": x,
                "nucleus_area_px2": region.area,
                "nucleus_area_um2": region.area * PIXEL_UM**2,
                "nucleus_eccentricity": region.eccentricity,
                "nucleus_solidity": region.solidity,
                "nearest_neighbor_um": nearest_px[index] * PIXEL_UM,
                "neighbors_within_14um": int(neighbor_count[index]),
                "local_nuclear_fraction_14um": float(local_fraction_map[int(round(y)), int(round(x))]),
                "edge_distance_um": edge_distance_px * PIXEL_UM,
                "yap_nucleus_mean": nucleus_mean,
                "yap_ring_mean": ring_mean,
                "yap_nucleus_ring_ratio": nucleus_mean / ring_mean,
            }
        )
    return pd.DataFrame(rows), labels


def existing_field_summary(sample: str) -> pd.DataFrame:
    path = next((ROOT / sample).rglob(f"{sample}_Nuclei.csv"))
    frame = pd.read_csv(path)
    return (
        frame.groupby("ImageNumber", as_index=False)
        .agg(
            existing_count=("ObjectNumber", "size"),
            existing_ratio_median=("Math_NucByRingYAP", "median"),
        )
        .rename(columns={"ImageNumber": "field"})
    )


def within_field_residual(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column] - frame.groupby("field_id")[column].transform("median")


def permutation_pvalue(frame: pd.DataFrame, x_col: str, y_col: str, observed: float, n_iter: int = 1000) -> float:
    rng = np.random.default_rng(20261027)
    x = frame[x_col].to_numpy()
    y = frame[y_col].to_numpy()
    groups = frame["field_id"].to_numpy()
    indices = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    extreme = 0
    for _ in range(n_iter):
        shuffled = y.copy()
        for idx in indices:
            shuffled[idx] = rng.permutation(shuffled[idx])
        rho = spearmanr(x, shuffled).statistic
        extreme += abs(rho) >= abs(observed)
    return (extreme + 1) / (n_iter + 1)


def save_qc_figure(qc: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]) -> Path:
    fig, axes = plt.subplots(len(qc), 3, figsize=(10, 3 * len(qc)), constrained_layout=True)
    if len(qc) == 1:
        axes = axes[None, :]
    for row, (name, dapi, yap, labels) in enumerate(qc):
        dapi_show = exposure.rescale_intensity(dapi, in_range=tuple(np.percentile(dapi, (1, 99.7))))
        yap_show = exposure.rescale_intensity(yap, in_range=tuple(np.percentile(yap, (1, 99.7))))
        overlay = segmentation.mark_boundaries(yap_show, labels, color=(1, 0, 0), mode="thick")
        axes[row, 0].imshow(dapi_show, cmap="gray")
        axes[row, 1].imshow(yap_show, cmap="gray")
        axes[row, 2].imshow(overlay)
        axes[row, 0].set_ylabel(name)
        for col, title in enumerate(("DAPI", "YAP", "Nuclear segmentation")):
            if row == 0:
                axes[row, col].set_title(title)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    output = OUTPUT_DIR / "segmentation_qc.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def save_analysis_figure(cells: pd.DataFrame, field_stats: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

    sample_cells = cells.sample(min(3000, len(cells)), random_state=7)
    axes[0].scatter(
        sample_cells["local_nuclear_fraction_14um"],
        np.log2(sample_cells["yap_nucleus_ring_ratio"]),
        s=8,
        alpha=0.18,
        color="#4c72b0",
        linewidths=0,
    )
    bins = pd.qcut(cells["local_nuclear_fraction_14um"], 8, duplicates="drop")
    trend = cells.assign(bin=bins).groupby("bin", observed=True).agg(
        x=("local_nuclear_fraction_14um", "median"),
        y=("yap_nucleus_ring_ratio", lambda v: np.median(np.log2(v))),
    )
    axes[0].plot(trend["x"], trend["y"], color="#c44e52", marker="o", linewidth=2)
    axes[0].set_xlabel("Local nuclear fraction within 14 µm")
    axes[0].set_ylabel("log2(YAP nucleus / ring)")
    axes[0].set_title("Cell-level association")

    ordered = sorted(field_stats["sample"].unique(), key=lambda x: (int(x[1]), int(x[2])))
    positions = {sample: idx for idx, sample in enumerate(ordered)}
    axes[1].scatter(
        [positions[value] for value in field_stats["sample"]],
        field_stats["spearman_local_fraction"],
        s=34,
        alpha=0.8,
        color="#4c72b0",
    )
    axes[1].axhline(0, color="#777777", linewidth=1)
    axes[1].set_xticks(range(len(ordered)), ordered, rotation=45)
    axes[1].set_ylabel("Within-field Spearman ρ")
    axes[1].set_title("Consistency across fields")

    axes[2].scatter(
        field_stats["nuclei_per_10000_um2"],
        field_stats["ratio_median"],
        s=42,
        alpha=0.8,
        color="#4c72b0",
    )
    for _, row in field_stats.iterrows():
        axes[2].annotate(row["field_id"], (row["nuclei_per_10000_um2"], row["ratio_median"]), fontsize=6, alpha=0.55)
    axes[2].set_xlabel("Nuclei / 10,000 µm²")
    axes[2].set_ylabel("Median YAP nucleus / ring")
    axes[2].set_title("Field density vs field median")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#dddddd", linewidth=0.7, alpha=0.6)
    output = OUTPUT_DIR / "yap_local_crowding_analysis.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_cells = []
    validation_rows = []
    qc = []

    for sample_dir in sorted((p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("D")), key=lambda p: (int(p.name[1]), int(p.name[2]))):
        sample = sample_dir.name
        existing = existing_field_summary(sample)
        fields = sorted(int(path.stem.split("-")[0]) for path in sample_dir.glob("*-C1.tif"))
        for field_number in fields:
            if (sample, field_number) == KNOWN_DUPLICATE:
                continue
            dapi_path = sample_dir / f"{field_number:02d}-C1.tif"
            yap_path = sample_dir / f"{field_number:02d}-C2.tif"
            cells, labels = measure_field(sample, field_number, dapi_path, yap_path)
            if cells.empty:
                continue
            all_cells.append(cells)
            existing_row = existing.loc[existing["field"] == field_number].iloc[0]
            validation_rows.append(
                {
                    "sample": sample,
                    "field": field_number,
                    "field_id": f"{sample}_{field_number:02d}",
                    "recomputed_count": len(cells),
                    "existing_count": int(existing_row["existing_count"]),
                    "recomputed_ratio_median": float(cells["yap_nucleus_ring_ratio"].median()),
                    "existing_ratio_median": float(existing_row["existing_ratio_median"]),
                }
            )
            if (sample, field_number) in {("D30D", 1), ("D43D", 1), ("D50D", 1), ("D53D", 1)}:
                qc.append((f"{sample} field {field_number}", tifffile.imread(dapi_path), tifffile.imread(yap_path), labels))

    cells = pd.concat(all_cells, ignore_index=True)
    cells = cells.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["yap_nucleus_ring_ratio", "local_nuclear_fraction_14um", "nearest_neighbor_um"]
    )
    cells = cells.loc[(cells["yap_nucleus_ring_ratio"] > 0) & (cells["yap_nucleus_ring_ratio"] < 100)].copy()
    cells["log2_yap_ratio"] = np.log2(cells["yap_nucleus_ring_ratio"])
    cells["log2_ratio_within_field"] = within_field_residual(cells, "log2_yap_ratio")
    cells["local_fraction_within_field"] = within_field_residual(cells, "local_nuclear_fraction_14um")

    field_rows = []
    quartile_differences = []
    for field_id, group in cells.groupby("field_id"):
        if len(group) < 10:
            continue
        rho = spearmanr(group["local_nuclear_fraction_14um"], group["log2_yap_ratio"]).statistic
        try:
            quartiles = pd.qcut(group["local_nuclear_fraction_14um"], 4, labels=False, duplicates="drop")
            if quartiles.nunique() == 4:
                q1 = group.loc[quartiles == 0, "log2_yap_ratio"].median()
                q4 = group.loc[quartiles == 3, "log2_yap_ratio"].median()
                quartile_differences.append(float(q4 - q1))
        except ValueError:
            pass
        area_um2 = (800 * PIXEL_UM) ** 2
        field_rows.append(
            {
                "sample": group["sample"].iloc[0],
                "field": int(group["field"].iloc[0]),
                "field_id": field_id,
                "n_cells": len(group),
                "nuclei_per_10000_um2": len(group) / area_um2 * 10000,
                "ratio_median": float(group["yap_nucleus_ring_ratio"].median()),
                "local_fraction_median": float(group["local_nuclear_fraction_14um"].median()),
                "nearest_neighbor_median_um": float(group["nearest_neighbor_um"].median()),
                "spearman_local_fraction": float(rho),
            }
        )
    field_stats = pd.DataFrame(field_rows)
    validation = pd.DataFrame(validation_rows)

    within_rho = float(spearmanr(cells["local_fraction_within_field"], cells["log2_ratio_within_field"]).statistic)
    permutation_p = permutation_pvalue(
        cells,
        "local_fraction_within_field",
        "log2_ratio_within_field",
        within_rho,
    )
    field_rho = float(spearmanr(field_stats["nuclei_per_10000_um2"], field_stats["ratio_median"]).statistic)
    field_p = float(spearmanr(field_stats["nuclei_per_10000_um2"], field_stats["ratio_median"]).pvalue)
    count_validation = spearmanr(validation["recomputed_count"], validation["existing_count"])
    ratio_validation = spearmanr(validation["recomputed_ratio_median"], validation["existing_ratio_median"])
    if quartile_differences:
        quartile_test = wilcoxon(quartile_differences, alternative="two-sided")
        qdiff_median = float(np.median(quartile_differences))
        qdiff_p = float(quartile_test.pvalue)
    else:
        qdiff_median = np.nan
        qdiff_p = np.nan

    cells.to_csv(OUTPUT_DIR / "yap_cell_level_recomputed.csv", index=False)
    field_stats.to_csv(OUTPUT_DIR / "yap_field_level_recomputed.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "segmentation_validation_against_existing.csv", index=False)
    qc_path = save_qc_figure(qc)
    plot_path = save_analysis_figure(cells, field_stats)

    summary = {
        "cells_analyzed": int(len(cells)),
        "fields_analyzed": int(cells["field_id"].nunique()),
        "known_duplicate_excluded": "D43D field 4",
        "count_validation_spearman_rho": float(count_validation.statistic),
        "count_validation_p": float(count_validation.pvalue),
        "field_median_ratio_validation_spearman_rho": float(ratio_validation.statistic),
        "field_median_ratio_validation_p": float(ratio_validation.pvalue),
        "within_field_crowding_ratio_spearman_rho": within_rho,
        "within_field_permutation_p": permutation_p,
        "median_log2_ratio_difference_high_vs_low_crowding_quartile": qdiff_median,
        "paired_field_wilcoxon_p": qdiff_p,
        "field_density_vs_field_median_ratio_spearman_rho": field_rho,
        "field_density_vs_field_median_ratio_p": field_p,
        "qc_figure": str(qc_path),
        "analysis_figure": str(plot_path),
    }
    with (OUTPUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(summary.items())
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()

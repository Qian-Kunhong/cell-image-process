from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import measure


WORKSPACE = Path(__file__).resolve().parent
DEPENDENCY_DIR = WORKSPACE / ".codex_deps" / "oiffile"
if str(DEPENDENCY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPENDENCY_DIR))

import oiffile  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


RAW_DIR = Path(
    r"F:\Naruchit data\1_Data of published manuscripts\3_Hepatocyte differentiation"
    r"\03_Raw data\Immunostaining\Confocal - YAP"
)
MASK_ROOT = (
    WORKSPACE / "analysis_outputs" / "hepatocyte_confocal_audit" / "dapi_3d" / "yap"
)
OUTPUT_DIR = MASK_ROOT / "analysis_summary"

XY_UM = 0.17578
Z_UM = 0.5
SPACING = np.asarray([Z_UM, XY_UM, XY_UM])
VOXEL_VOLUME_UM3 = float(np.prod(SPACING))

SAMPLE_ORDER = [
    "D20D", "D30D", "D31D", "D33D", "D40D", "D41D", "D43D", "D50D", "D51D", "D53D"
]
CONDITIONS = {
    "D20D": (2, -3, "shared_baseline", "共同基线"),
    "D30D": (3, -2, "no_HA", "无HA"),
    "D31D": (3, -2, "single_HA", "单次HA方案"),
    "D33D": (3, -2, "repeated_HA", "连续3天HA方案"),
    "D40D": (4, -1, "no_HA", "无HA"),
    "D41D": (4, -1, "single_HA", "单次HA方案"),
    "D43D": (4, -1, "repeated_HA", "连续3天HA方案"),
    "D50D": (5, 0, "no_HA", "无HA"),
    "D51D": (5, 0, "single_HA", "单次HA方案"),
    "D53D": (5, 0, "repeated_HA", "连续3天HA方案"),
}

# A single, condition-independent plausibility gate. These deliberately broad limits
# remove tiny debris and obvious merged objects without encoding culture day.
QC_LIMITS = {
    "volume_um3": (250.0, 2200.0),
    "max_xy_area_um2": (30.0, 260.0),
    "height_um": (4.0, 20.0),
}
SHELL_INNER_UM = 0.35
SHELL_OUTER_UM = 2.0
DETECTOR_SATURATION_LEVEL = 4095


def ensure_zcyx(array: np.ndarray, axes: str) -> np.ndarray:
    if axes != "ZCYX":
        raise ValueError(f"Expected ZCYX metadata, got {axes} and {array.shape}")
    if array.ndim == 4 and array.shape[0] <= 4 and array.shape[1] > 4:
        return np.transpose(array, (1, 0, 2, 3))
    return array


def border_ids(labels: np.ndarray) -> set[int]:
    ids = set(np.unique(labels[0])) | set(np.unique(labels[-1]))
    ids |= set(np.unique(labels[:, 0])) | set(np.unique(labels[:, -1]))
    ids |= set(np.unique(labels[:, :, 0])) | set(np.unique(labels[:, :, -1]))
    ids.discard(0)
    return ids


def surface_area_um2(mask: np.ndarray) -> float:
    padded = np.pad(mask.astype(np.uint8), 1)
    vertices, faces, _, _ = measure.marching_cubes(padded, level=0.5, spacing=SPACING)
    return float(measure.mesh_surface_area(vertices, faces))


def shell_values(
    yap: np.ndarray,
    labels: np.ndarray,
    label: int,
    bbox: tuple[int, int, int, int, int, int],
) -> np.ndarray:
    pad = np.ceil(SHELL_OUTER_UM / SPACING).astype(int) + 1
    z0, y0, x0, z1, y1, x1 = bbox
    z0p, y0p, x0p = np.maximum([z0, y0, x0] - pad, 0)
    z1p, y1p, x1p = np.minimum([z1, y1, x1] + pad, labels.shape)
    local_labels = labels[z0p:z1p, y0p:y1p, x0p:x1p]
    target = local_labels == label
    distance = ndi.distance_transform_edt(~target, sampling=SPACING)
    shell = (distance >= SHELL_INNER_UM) & (distance <= SHELL_OUTER_UM)
    shell &= local_labels == 0
    return yap[z0p:z1p, y0p:y1p, x0p:x1p][shell]


def quantify_sample(sample: str) -> tuple[pd.DataFrame, dict[str, float]]:
    raw_labels = np.load(MASK_ROOT / sample / "dapi_3d_masks.npz")["raw_labels"].astype(np.int32)
    with oiffile.OifFile(RAW_DIR / f"{sample}.oib") as handle:
        stack = ensure_zcyx(handle.asarray(), handle.axes)
    yap = stack[:, 1].astype(np.float32)
    excluded_border = border_ids(raw_labels)
    rows: list[dict[str, float | int | str | bool]] = []

    for region in measure.regionprops(raw_labels):
        if region.label in excluded_border:
            continue
        coords = region.coords
        z0, y0, x0, z1, y1, x1 = region.bbox
        per_z = np.bincount(coords[:, 0], minlength=raw_labels.shape[0])
        volume = float(region.area * VOXEL_VOLUME_UM3)
        max_area = float(per_z.max() * XY_UM**2)
        height = float((z1 - z0) * Z_UM)
        physical = coords.astype(float) * SPACING
        centered = physical - physical.mean(axis=0)
        covariance = centered.T @ centered / max(len(centered) - 1, 1)
        axes = 4 * np.sqrt(np.clip(np.sort(np.linalg.eigvalsh(covariance))[::-1], 0, None))
        mask_crop = raw_labels[z0:z1, y0:y1, x0:x1] == region.label
        area = surface_area_um2(mask_crop)
        sphericity = (np.pi ** (1 / 3) * (6 * volume) ** (2 / 3) / area) if area else np.nan

        nuclear = yap[raw_labels == region.label]
        shell = shell_values(yap, raw_labels, region.label, region.bbox)
        shell_positive = shell[shell > 0]
        nuclear_median = float(np.median(nuclear))
        shell_median = float(np.median(shell)) if shell.size else np.nan
        shell_positive_median = float(np.median(shell_positive)) if shell_positive.size else np.nan
        ratio_all = nuclear_median / shell_median if shell_median > 0 else np.nan
        ratio_positive = nuclear_median / shell_positive_median if shell_positive_median > 0 else np.nan
        qc_pass = (
            QC_LIMITS["volume_um3"][0] <= volume <= QC_LIMITS["volume_um3"][1]
            and QC_LIMITS["max_xy_area_um2"][0] <= max_area <= QC_LIMITS["max_xy_area_um2"][1]
            and QC_LIMITS["height_um"][0] <= height <= QC_LIMITS["height_um"][1]
        )
        rows.append(
            {
                "sample": sample,
                "label": region.label,
                "qc_pass_uniform": qc_pass,
                "volume_um3": volume,
                "surface_area_um2": area,
                "sphericity": sphericity,
                "height_um": height,
                "max_xy_area_um2": max_area,
                "effective_thickness_um": volume / max_area,
                "pca_axis_1_um": axes[0],
                "pca_axis_2_um": axes[1],
                "pca_axis_3_um": axes[2],
                "pca_elongation_1_over_3": axes[0] / max(axes[2], 1e-9),
                "centroid_z_um": region.centroid[0] * Z_UM,
                "centroid_y_um": region.centroid[1] * XY_UM,
                "centroid_x_um": region.centroid[2] * XY_UM,
                "yap_nuclear_median": nuclear_median,
                "yap_shell_median": shell_median,
                "yap_shell_positive_median": shell_positive_median,
                "yap_shell_positive_fraction": float(shell_positive.size / shell.size) if shell.size else np.nan,
                "yap_nuclear_shell_ratio_all_voxels": ratio_all,
                "yap_nuclear_shell_ratio": ratio_positive,
                "yap_nuclear_saturation_fraction": float(np.mean(nuclear >= DETECTOR_SATURATION_LEVEL)),
                "shell_voxel_count": int(shell.size),
            }
        )

    frame = pd.DataFrame(rows)
    kept = frame[frame["qc_pass_uniform"]].copy()
    if len(kept) >= 2:
        centroids = kept[["centroid_z_um", "centroid_y_um", "centroid_x_um"]].to_numpy()
        distances, _ = cKDTree(centroids).query(centroids, k=2)
        frame.loc[kept.index, "nearest_neighbor_3d_um"] = distances[:, 1]
    meta = {
        "raw_objects": int(raw_labels.max()),
        "border_excluded": len(excluded_border),
        "complete_objects": len(frame),
        "uniform_qc_objects": int(frame["qc_pass_uniform"].sum()),
        "stack_z_planes": int(raw_labels.shape[0]),
        "yap_stack_saturation_fraction": float(np.mean(yap >= DETECTOR_SATURATION_LEVEL)),
    }
    return frame, meta


def add_condition_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["culture_day"] = frame["sample"].map(lambda x: CONDITIONS[x][0])
    frame["paper_day"] = frame["sample"].map(lambda x: CONDITIONS[x][1])
    frame["regimen"] = frame["sample"].map(lambda x: CONDITIONS[x][2])
    frame["regimen_cn"] = frame["sample"].map(lambda x: CONDITIONS[x][3])
    return frame


def condition_summary(cells: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    kept = cells[cells["qc_pass_uniform"]]
    metrics = [
        "volume_um3", "max_xy_area_um2", "height_um", "effective_thickness_um",
        "sphericity", "pca_elongation_1_over_3", "nearest_neighbor_3d_um",
        "yap_nuclear_median", "yap_shell_median", "yap_nuclear_shell_ratio",
    ]
    grouped = kept.groupby("sample", observed=True)
    pieces = [grouped.size().rename("n_nuclei")]
    for metric in metrics:
        pieces.extend(
            [
                grouped[metric].median().rename(f"{metric}_median"),
                grouped[metric].quantile(0.25).rename(f"{metric}_q25"),
                grouped[metric].quantile(0.75).rename(f"{metric}_q75"),
            ]
        )
    summary = pd.concat(pieces, axis=1).reset_index()
    summary = add_condition_columns(summary)
    summary = summary.merge(metadata, on="sample", how="left")
    summary["sample"] = pd.Categorical(summary["sample"], SAMPLE_ORDER, ordered=True)
    return summary.sort_values("sample")


def same_day_contrasts(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    indexed = summary.set_index("sample")
    for day, names in {3: ("D30D", "D31D", "D33D"), 4: ("D40D", "D41D", "D43D"), 5: ("D50D", "D51D", "D53D")}.items():
        control = indexed.loc[names[0]]
        for sample in names[1:]:
            treated = indexed.loc[sample]
            row = {"culture_day": day, "paper_day": CONDITIONS[sample][1], "comparison": f"{sample} vs {names[0]}"}
            for metric in ["volume_um3", "max_xy_area_um2", "height_um", "yap_nuclear_shell_ratio"]:
                base = control[f"{metric}_median"]
                row[f"{metric}_median_percent_change"] = 100 * (treated[f"{metric}_median"] / base - 1)
            rows.append(row)
    return pd.DataFrame(rows)


def plot_nuclear_summary(cells: pd.DataFrame, summary: pd.DataFrame) -> None:
    kept = cells[cells["qc_pass_uniform"]].copy()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    specs = [
        ("max_xy_area_um2", "最大二维核截面积 (µm²)"),
        ("volume_um3", "三维核体积 (µm³)"),
        ("height_um", "核高度 (µm)"),
        ("effective_thickness_um", "体积/最大截面积 (µm)"),
    ]
    palette = {"shared_baseline": "#666666", "no_HA": "#3b82f6", "single_HA": "#f59e0b", "repeated_HA": "#dc2626"}
    for ax, (metric, ylabel) in zip(axes.flat, specs):
        values = [kept.loc[kept["sample"] == sample, metric].dropna().to_numpy() for sample in SAMPLE_ORDER]
        box = ax.boxplot(values, tick_labels=SAMPLE_ORDER, showfliers=False, patch_artist=True)
        for patch, sample in zip(box["boxes"], SAMPLE_ORDER):
            patch.set_facecolor(palette[CONDITIONS[sample][2]])
            patch.set_alpha(0.72)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=35)
    legend_names = {"shared_baseline": "共同基线", "no_HA": "无HA", "single_HA": "单次HA", "repeated_HA": "连续3天HA"}
    handles = [Patch(facecolor=color, alpha=0.72, label=legend_names[name]) for name, color in palette.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("统一物理尺度质控后的三维核形态（每个条件仅1个OIB视野）", y=1.01)
    fig.savefig(OUTPUT_DIR / "nuclear_2d_vs_3d_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    scatter = axes[0].scatter(kept["max_xy_area_um2"], kept["volume_um3"], c=kept["culture_day"], cmap="viridis", alpha=0.55, s=25)
    fig.colorbar(scatter, ax=axes[0], label="接种后培养天数")
    axes[0].set_xlabel("最大二维核截面积 (µm²)")
    axes[0].set_ylabel("三维核体积 (µm³)")
    med = summary.copy()
    axes[1].scatter(med["max_xy_area_um2_median"], med["volume_um3_median"], c=med["culture_day"], cmap="viridis", s=70)
    for _, row in med.iterrows():
        axes[1].annotate(str(row["sample"]), (row["max_xy_area_um2_median"], row["volume_um3_median"]), xytext=(4, 3), textcoords="offset points")
    axes[1].set_xlabel("条件中位最大二维截面积 (µm²)")
    axes[1].set_ylabel("条件中位三维体积 (µm³)")
    axes[0].set_title("单核层面")
    axes[1].set_title("视野/条件层面")
    fig.savefig(OUTPUT_DIR / "nuclear_area_volume_relation.png", dpi=200)
    plt.close(fig)


def plot_yap_summary(cells: pd.DataFrame) -> None:
    kept = cells[cells["qc_pass_uniform"] & np.isfinite(cells["yap_nuclear_shell_ratio"])].copy()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    values = [kept.loc[kept["sample"] == sample, "yap_nuclear_shell_ratio"].dropna().to_numpy() for sample in SAMPLE_ORDER]
    box = axes[0].boxplot(values, tick_labels=SAMPLE_ORDER, showfliers=False, patch_artist=True)
    for patch in box["boxes"]:
        patch.set_facecolor("#94a3b8")
        patch.set_alpha(0.8)
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("YAP 核/阳性周核壳层中位强度比")
    scatter = axes[1].scatter(kept["nearest_neighbor_3d_um"], kept["yap_nuclear_shell_ratio"], c=kept["culture_day"], cmap="viridis", alpha=0.6, s=25)
    fig.colorbar(scatter, ax=axes[1], label="接种后培养天数")
    axes[1].set_xlabel("三维最近核距离 (µm)")
    axes[1].set_ylabel("YAP 核/周核比")
    scatter = axes[2].scatter(kept["volume_um3"], kept["yap_nuclear_shell_ratio"], c=kept["culture_day"], cmap="viridis", alpha=0.6, s=25)
    fig.colorbar(scatter, ax=axes[2], label="接种后培养天数")
    axes[2].set_xlabel("三维核体积 (µm³)")
    axes[2].set_ylabel("YAP 核/周核比")
    fig.suptitle("三维 YAP 探索性定量；周核仅统计非零体素，阳性比例另作质控")
    fig.savefig(OUTPUT_DIR / "yap_3d_exploratory_summary.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_frames = []
    metadata_rows = []
    for sample in SAMPLE_ORDER:
        print(f"Quantifying {sample}...", flush=True)
        frame, metadata = quantify_sample(sample)
        all_frames.append(frame)
        metadata_rows.append({"sample": sample, **metadata})
    cells = add_condition_columns(pd.concat(all_frames, ignore_index=True))
    metadata = pd.DataFrame(metadata_rows)
    summary = condition_summary(cells, metadata)
    contrasts = same_day_contrasts(summary)

    cells.to_csv(OUTPUT_DIR / "nucleus_3d_yap_cell_metrics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "condition_3d_summary.csv", index=False)
    contrasts.to_csv(OUTPUT_DIR / "same_day_regimen_contrasts.csv", index=False)
    plot_nuclear_summary(cells, summary)
    plot_yap_summary(cells)

    kept = cells[cells["qc_pass_uniform"]]
    rho_cell = kept[["max_xy_area_um2", "volume_um3"]].corr(method="spearman").iloc[0, 1]
    rho_condition = summary[["max_xy_area_um2_median", "volume_um3_median"]].corr(method="spearman").iloc[0, 1]
    notes = [
        "Exploratory analysis only: there is one OIB field per condition, so nuclei are technical subsamples, not biological replicates.",
        "A single condition-independent physical-size QC gate was used for every sample.",
        f"Spearman association between maximum 2D area and 3D volume: single nuclei rho={rho_cell:.3f}; condition medians rho={rho_condition:.3f}.",
        "YAP cross-condition absolute intensities were not compared because acquisition settings/saturation differ; the reported ratio is within-image nuclear/perinuclear-shell contrast.",
        "The 3D shell is a geometric perinuclear compartment, not a segmented cytoplasm, so it remains sensitive to cell packing and extracellular background.",
        "Because early stacks contain many zero-valued background voxels, the primary YAP ratio uses the median of positive shell voxels; shell-positive fraction must be inspected alongside it.",
    ]
    (OUTPUT_DIR / "interpretation_limits.txt").write_text("\n".join(notes), encoding="utf-8")
    print(summary[["sample", "n_nuclei", "volume_um3_median", "max_xy_area_um2_median", "height_um_median", "yap_nuclear_shell_ratio_median"]].to_string(index=False), flush=True)
    print(f"Area-volume rho: cells={rho_cell:.3f}, condition medians={rho_condition:.3f}", flush=True)


if __name__ == "__main__":
    main()

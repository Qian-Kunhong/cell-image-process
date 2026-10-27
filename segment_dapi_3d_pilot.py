from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cellpose import models
from scipy import ndimage as ndi
from skimage import exposure, measure, segmentation


WORKSPACE = Path(__file__).resolve().parent
DEPENDENCY_DIR = WORKSPACE / ".codex_deps" / "oiffile"
if str(DEPENDENCY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPENDENCY_DIR))

import oiffile  # noqa: E402


RAW_ROOT = Path(
    r"F:\Naruchit data\1_Data of published manuscripts\3_Hepatocyte differentiation"
    r"\03_Raw data\Immunostaining"
)
DATASET_DIRS = {
    "yap": RAW_ROOT / "Confocal - YAP",
    "ecad_f_actin": RAW_ROOT / "Confocal - E-cad F-actin",
}
OUTPUT_ROOT = WORKSPACE / "analysis_outputs" / "hepatocyte_confocal_audit" / "dapi_3d"
XY_UM = 0.17578
Z_UM = 0.5
ANISOTROPY = Z_UM / XY_UM


def ensure_zcyx(array: np.ndarray, axes: str) -> np.ndarray:
    if axes != "ZCYX":
        raise ValueError(f"Expected ZCYX metadata, got {axes} and {array.shape}")
    if array.ndim == 4 and array.shape[0] <= 4 and array.shape[1] > 4:
        return np.transpose(array, (1, 0, 2, 3))
    return array


def robust_display(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    nonzero = image[image > 0]
    lo, hi = np.percentile(nonzero, (0.5, 99.7))
    return exposure.rescale_intensity(image, in_range=(lo, hi), out_range=(0, 1))


def boundary_labels(labels: np.ndarray) -> tuple[np.ndarray, set[int]]:
    border_ids = set(np.unique(labels[0])) | set(np.unique(labels[-1]))
    border_ids |= set(np.unique(labels[:, 0])) | set(np.unique(labels[:, -1]))
    border_ids |= set(np.unique(labels[:, :, 0])) | set(np.unique(labels[:, :, -1]))
    border_ids.discard(0)
    keep = np.asarray([value for value in np.unique(labels) if value and value not in border_ids])
    filtered = np.where(np.isin(labels, keep), labels, 0)
    filtered, _, _ = segmentation.relabel_sequential(filtered)
    return filtered.astype(np.int32), border_ids


def stage_diameter(sample: str) -> tuple[int, int]:
    stage = int(sample[1])
    if stage <= 3:
        return 47, 100
    if stage == 4:
        return 40, 100
    return 40, 75


def cross_section_filter(labels: np.ndarray, sample: str) -> tuple[np.ndarray, set[int]]:
    min_diameter, max_diameter = stage_diameter(sample)
    min_area = np.pi * (min_diameter / 2) ** 2
    max_area = np.pi * (max_diameter / 2) ** 2
    keep = []
    excluded = set()
    for region in measure.regionprops(labels):
        z_indices = region.coords[:, 0]
        max_area_px = np.bincount(z_indices, minlength=labels.shape[0]).max()
        if min_area <= max_area_px <= max_area:
            keep.append(region.label)
        else:
            excluded.add(region.label)
    filtered = np.where(np.isin(labels, keep), labels, 0)
    filtered, _, _ = segmentation.relabel_sequential(filtered)
    return filtered.astype(np.int32), excluded


def object_metrics(labels: np.ndarray, sample: str, dataset: str) -> pd.DataFrame:
    rows = []
    voxel_volume = Z_UM * XY_UM * XY_UM
    for region in measure.regionprops(labels):
        coords = region.coords.astype(np.float64)
        physical = coords * np.asarray([Z_UM, XY_UM, XY_UM])
        centered = physical - physical.mean(axis=0)
        covariance = centered.T @ centered / max(len(centered) - 1, 1)
        eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
        axis_scales = 4 * np.sqrt(np.clip(eigenvalues, 0, None))
        z0, y0, x0, z1, y1, x1 = region.bbox
        per_z = np.bincount(coords[:, 0].astype(int), minlength=labels.shape[0])
        max_area_px = int(per_z.max())
        max_area_z = int(np.argmax(per_z))
        rows.append(
            {
                "dataset": dataset,
                "sample": sample,
                "label": region.label,
                "voxel_count": region.area,
                "volume_um3": region.area * voxel_volume,
                "centroid_z_um": region.centroid[0] * Z_UM,
                "centroid_y_um": region.centroid[1] * XY_UM,
                "centroid_x_um": region.centroid[2] * XY_UM,
                "height_um": (z1 - z0) * Z_UM,
                "bbox_y_um": (y1 - y0) * XY_UM,
                "bbox_x_um": (x1 - x0) * XY_UM,
                "pca_axis_1_um": axis_scales[0],
                "pca_axis_2_um": axis_scales[1],
                "pca_axis_3_um": axis_scales[2],
                "pca_elongation_1_over_3": axis_scales[0] / max(axis_scales[2], 1e-9),
                "max_xy_area_um2": max_area_px * XY_UM * XY_UM,
                "max_xy_area_z_index": max_area_z,
                "equivalent_sphere_diameter_um": 2 * ((3 * region.area * voxel_volume) / (4 * np.pi)) ** (1 / 3),
            }
        )
    return pd.DataFrame(rows)


def projection_panel(image: np.ndarray, labels: np.ndarray, output: Path, title: str) -> None:
    views = [
        (image.max(axis=0), np.any(labels > 0, axis=0), "XY projection"),
        (image.max(axis=1), np.any(labels > 0, axis=1), "XZ projection"),
        (image.max(axis=2), np.any(labels > 0, axis=2), "YZ projection"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, (view, mask, name) in zip(axes, views):
        shown = robust_display(view)
        overlay = segmentation.mark_boundaries(shown, measure.label(mask), color=(1, 0, 0), mode="thick")
        ax.imshow(overlay)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("auto" if name != "XY projection" else "equal")
    fig.suptitle(title)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def run_sample(model: models.CellposeModel, dataset: str, sample: str, diameter: float, cellprob: float, reuse: bool) -> None:
    path = DATASET_DIRS[dataset] / f"{sample}.oib"
    with oiffile.OifFile(path) as handle:
        stack = ensure_zcyx(handle.asarray(), handle.axes)
    dapi = stack[:, 0].astype(np.float32)
    output_dir = OUTPUT_ROOT / dataset / sample
    mask_path = output_dir / "dapi_3d_masks.npz"
    if reuse and mask_path.exists():
        masks = np.load(mask_path)["raw_labels"].astype(np.int32)
    else:
        masks, flows, styles = model.eval(
            dapi,
            batch_size=2,
            channel_axis=None,
            z_axis=0,
            normalize={"lowhigh": None, "percentile": [1.0, 99.8], "normalize": True},
            diameter=diameter,
            do_3D=True,
            anisotropy=ANISOTROPY,
            flow3D_smooth=1,
            cellprob_threshold=cellprob,
            min_size=300,
            bsize=256,
        )
        masks = np.asarray(masks, dtype=np.int32)
    boundary_filtered, border_ids = boundary_labels(masks)
    filtered, size_ids = cross_section_filter(boundary_filtered, sample)
    metrics = object_metrics(filtered, sample, dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "dapi_3d_masks.npz",
        labels=filtered,
        raw_labels=masks,
        pixel_size_xy_um=XY_UM,
        z_step_um=Z_UM,
    )
    metrics.to_csv(output_dir / "nucleus_3d_metrics.csv", index=False)
    projection_panel(
        dapi,
        filtered,
        output_dir / "orthogonal_projection_qc.png",
        f"{dataset} / {sample}: {len(metrics)} complete 3D nuclei; "
        f"{len(border_ids)} border and {len(size_ids)} size-filtered objects excluded",
    )
    print(
        f"{dataset}/{sample}: raw={int(masks.max())}, complete={len(metrics)}, "
        f"border_excluded={len(border_ids)}, size_excluded={len(size_ids)}, median_volume_um3="
        f"{metrics['volume_um3'].median() if len(metrics) else float('nan'):.1f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASET_DIRS, default="yap")
    parser.add_argument("--samples", nargs="+", default=["D40D"])
    parser.add_argument("--diameter", type=float, default=55.0)
    parser.add_argument("--cellprob", type=float, default=0.0)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()

    model = models.CellposeModel(gpu=True, pretrained_model="cpsam")
    for sample in args.samples:
        run_sample(model, args.dataset, sample, args.diameter, args.cellprob, args.reuse)


if __name__ == "__main__":
    main()

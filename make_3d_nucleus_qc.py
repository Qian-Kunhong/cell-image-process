from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


def ensure_zcyx(array: np.ndarray, axes: str) -> np.ndarray:
    if array.shape[0] <= 4 and array.shape[1] > 4:
        return np.transpose(array, (1, 0, 2, 3))
    return array


def normalize_plane(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    positive = image[image > 0]
    lo, hi = np.percentile(positive, (0.5, 99.7))
    return exposure.rescale_intensity(image, in_range=(lo, hi), out_range=(0, 1))


def choose_labels(labels: np.ndarray, count: int = 6) -> list[int]:
    regions = list(measure.regionprops(labels))
    regions.sort(key=lambda region: region.area)
    if len(regions) <= count:
        return [region.label for region in regions]
    indices = np.linspace(int(0.12 * len(regions)), int(0.88 * len(regions)), count).astype(int)
    return [regions[index].label for index in indices]


def crop_bounds(center: int, radius: int, limit: int) -> tuple[int, int]:
    return max(0, center - radius), min(limit, center + radius + 1)


def overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return segmentation.mark_boundaries(normalize_plane(image), mask.astype(np.uint8), color=(1, 0, 0), mode="thick")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="yap", choices=DATASET_DIRS)
    parser.add_argument("--sample", default="D40D")
    args = parser.parse_args()

    output_dir = OUTPUT_ROOT / args.dataset / args.sample
    labels = np.load(output_dir / "dapi_3d_masks.npz")["labels"]
    with oiffile.OifFile(DATASET_DIRS[args.dataset] / f"{args.sample}.oib") as handle:
        dapi = ensure_zcyx(handle.asarray(), handle.axes)[:, 0]

    selected = choose_labels(labels)
    fig, axes = plt.subplots(len(selected), 3, figsize=(11, 3 * len(selected)), constrained_layout=True)
    for row, label in enumerate(selected):
        region = measure.regionprops((labels == label).astype(np.uint8))[0]
        zc, yc, xc = [int(round(value)) for value in region.centroid]
        y0, y1 = crop_bounds(yc, 65, labels.shape[1])
        x0, x1 = crop_bounds(xc, 65, labels.shape[2])
        z0, z1 = crop_bounds(zc, 14, labels.shape[0])

        xy_image = dapi[zc, y0:y1, x0:x1]
        xy_mask = labels[zc, y0:y1, x0:x1] == label
        xz_image = dapi[z0:z1, yc, x0:x1]
        xz_mask = labels[z0:z1, yc, x0:x1] == label
        yz_image = dapi[z0:z1, y0:y1, xc]
        yz_mask = labels[z0:z1, y0:y1, xc] == label

        axes[row, 0].imshow(overlay(xy_image, xy_mask), extent=(x0 * XY_UM, x1 * XY_UM, y1 * XY_UM, y0 * XY_UM))
        axes[row, 1].imshow(overlay(xz_image, xz_mask), extent=(x0 * XY_UM, x1 * XY_UM, z1 * Z_UM, z0 * Z_UM), aspect="equal")
        axes[row, 2].imshow(overlay(yz_image, yz_mask), extent=(y0 * XY_UM, y1 * XY_UM, z1 * Z_UM, z0 * Z_UM), aspect="equal")
        axes[row, 0].set_ylabel(f"label {label}\nvolume {region.area * Z_UM * XY_UM**2:.0f} µm³")
        for col, title in enumerate(("XY center slice", "XZ center slice", "YZ center slice")):
            if row == 0:
                axes[row, col].set_title(title)
            axes[row, col].set_xlabel("µm")
    output = output_dir / "representative_nuclei_orthogonal_qc.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()

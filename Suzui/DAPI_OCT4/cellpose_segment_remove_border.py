from __future__ import annotations

"""
General Image Segmentation: Cellpose + Border-Object Removal
============================================================

Features
--------
1. Recursively reads tif/tiff/png/jpg/jpeg images from an input directory.
2. Segments cells or nuclei with the pretrained Cellpose ``cpsam`` model.
3. Removes objects touching the image border, which are often incomplete cells.
4. Saves NumPy masks, TIFF masks, and previews with red object boundaries.

Installation
------------
Creating a dedicated Python environment is recommended. Install the dependencies:

    pip install cellpose tifffile numpy

GPU acceleration requires a CUDA-enabled PyTorch build compatible with the
computer's CUDA setup. CPU processing is the default and is recommended on macOS.

Basic usage
-----------
Open a terminal in the directory containing this script and run:

    python3 cellpose_segment_remove_border.py "/path/to/input_images" "/path/to/output"

The first path is the input image directory. The second path is the output
directory. On Windows, enclose any path containing spaces in double quotes.

Examples
--------
1. Standard CPU processing (recommended on macOS):

    python3 cellpose_segment_remove_border.py "/path/to/input_images" "/path/to/output"

2. Overwrite previously generated results:

    python3 cellpose_segment_remove_border.py "/path/to/input_images" "/path/to/output" --overwrite

3. Remove only objects that touch the outermost pixels:

    python3 cellpose_segment_remove_border.py "/path/to/input_images" "/path/to/output" --border-margin 0

4. Remove objects that occur within five pixels of the image border:

    python3 cellpose_segment_remove_border.py "/path/to/input_images" "/path/to/output" --border-margin 5

5. Enable a CUDA GPU on a supported Windows/Linux computer:

    python cellpose_segment_remove_border.py "D:/input_images" "D:/output" --gpu

Output directories
------------------
    output/
    ├── masks_npy/   # .npy label arrays for downstream Python analysis
    ├── masks_tif/   # .tif label images that can be opened in Fiji/ImageJ
    └── previews/    # Original images with red boundaries for visual inspection

In each mask, 0 is the background and 1, 2, 3, ... identify separate objects.

Notes
-----
- CPU processing is used by default and works on macOS, Windows, and Linux.
- The optional ``--gpu`` flag is intended for a correctly configured CUDA GPU.
- The default ``--border-margin`` is 1, so objects touching the outermost pixels
  or lying within one pixel of the edge are removed.
- Existing outputs are skipped by default. Use ``--overwrite`` to process them again.
- This script processes 2D images and does not handle unprojected 3D Z-stacks.
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff
from cellpose import io, models


# Supported input image formats.
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

# Cellpose parameters. Adjust these if segmentation is suboptimal for a new image type.
PRETRAINED_MODEL = "cpsam"
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0
DIAMETER = None


def list_images(input_dir: Path) -> list[Path]:
    """Recursively find supported images in the input directory."""
    return sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def first_channel(image: np.ndarray) -> np.ndarray:
    """Extract one 2D channel for preview generation."""
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[-1] in (1, 3, 4):
        return image[..., 0]
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        return image[0]
    raise ValueError(f"Unsupported image shape: {image.shape}")


def to_cellpose_image(image: np.ndarray) -> np.ndarray:
    """Convert grayscale or channel-first data to Cellpose-compatible H-W-C data."""
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        return image[..., :3]
    if image.ndim == 3 and image.shape[-1] == 1:
        return np.repeat(image, 3, axis=-1)
    if image.ndim == 3 and image.shape[0] in (3, 4):
        return np.moveaxis(image[:3], 0, -1)
    if image.ndim == 3 and image.shape[0] == 1:
        return np.repeat(np.moveaxis(image, 0, -1), 3, axis=-1)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def relabel_sequential(mask: np.ndarray) -> np.ndarray:
    """Renumber the remaining labels sequentially as 1, 2, 3, and so on."""
    labels = np.unique(mask)
    labels = labels[labels > 0]
    if len(labels) == 0:
        return np.zeros_like(mask, dtype=np.int32)
    lookup = np.zeros(int(mask.max()) + 1, dtype=np.int32)
    lookup[labels] = np.arange(1, len(labels) + 1, dtype=np.int32)
    return lookup[mask]


def remove_border_objects(mask: np.ndarray, margin_px: int = 1) -> tuple[np.ndarray, int]:
    """Remove all objects touching the border or its specified inner margin."""
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D label mask, got: {mask.shape}")
    if margin_px < 0:
        raise ValueError("margin_px must be >= 0")
    if mask.size == 0 or mask.max() == 0:
        return mask.astype(np.int32, copy=False), 0

    width = min(margin_px + 1, mask.shape[0], mask.shape[1])
    edge_labels = np.unique(np.concatenate([
        mask[:width, :].ravel(), mask[-width:, :].ravel(),
        mask[:, :width].ravel(), mask[:, -width:].ravel(),
    ]))
    edge_labels = edge_labels[edge_labels > 0]
    if len(edge_labels) == 0:
        return mask.astype(np.int32, copy=False), 0

    cleaned = mask.copy()
    cleaned[np.isin(cleaned, edge_labels)] = 0
    return relabel_sequential(cleaned), int(len(edge_labels))


def normalize_to_u8(image: np.ndarray) -> np.ndarray:
    """Scale the 1st-99th percentile intensity range to 0-255 for previews."""
    array = image.astype(np.float32)
    low, high = np.percentile(array, [1, 99])
    array = np.clip((array - low) / (high - low + 1e-6), 0, 1)
    return (array * 255).astype(np.uint8)


def make_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Draw the retained object boundaries in red over a grayscale image."""
    gray = normalize_to_u8(image)
    overlay = np.stack([gray, gray, gray], axis=-1)
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    boundary &= mask > 0
    overlay[boundary] = (255, 0, 0)
    return overlay


def parse_args() -> argparse.Namespace:
    """Read input/output directories and processing options from the command line."""
    parser = argparse.ArgumentParser(
        description="Segment images with Cellpose and remove border-touching objects."
    )
    parser.add_argument("input_dir", type=Path, help="Folder containing input images")
    parser.add_argument("output_dir", type=Path, help="Folder for masks and previews")
    parser.add_argument(
        "--border-margin", type=int, default=1,
        help="Inner border width in pixels (default: 1; use 0 for exact edge only)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use a configured CUDA GPU (CPU is the cross-platform default)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Convert user-supplied relative paths to absolute paths.
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    mask_npy_dir = output_dir / "masks_npy"
    mask_tif_dir = output_dir / "masks_tif"
    preview_dir = output_dir / "previews"

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    files = list_images(input_dir)
    if not files:
        raise RuntimeError(f"No supported images found in: {input_dir}")

    # Create the three output directories automatically.
    for directory in (mask_npy_dir, mask_tif_dir, preview_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # CPU is the safe cross-platform default, including Intel and Apple Silicon Macs.
    model = models.CellposeModel(gpu=args.gpu, pretrained_model=PRETRAINED_MODEL)
    done = skipped = failed = 0

    for image_path in files:
        relative = image_path.relative_to(input_dir)
        output_stem = "__".join(relative.with_suffix("").parts)
        mask_npy_path = mask_npy_dir / f"{output_stem}_mask.npy"
        mask_tif_path = mask_tif_dir / f"{output_stem}_mask.tif"
        preview_path = preview_dir / f"{output_stem}_overlay.tif"

        if mask_npy_path.exists() and not args.overwrite:
            print(f"[skip] {relative} (output already exists)")
            skipped += 1
            continue

        try:
            # Read the image and run Cellpose segmentation.
            image = np.asarray(io.imread(str(image_path)))
            masks, _, _ = model.eval(
                to_cellpose_image(image),
                flow_threshold=FLOW_THRESHOLD,
                cellprob_threshold=CELLPROB_THRESHOLD,
                diameter=DIAMETER,
            )
            # Remove all border-touching objects before saving the results.
            masks, removed_count = remove_border_objects(
                masks.astype(np.int32, copy=False), args.border_margin
            )

            # Save an analysis array, an ImageJ-compatible TIFF, and a QC preview.
            np.save(mask_npy_path, masks)
            tiff.imwrite(mask_tif_path, masks, photometric="minisblack")
            tiff.imwrite(preview_path, make_overlay(first_channel(image), masks))
            print(
                f"[ok] {relative} | kept={int(masks.max())} | "
                f"border_removed={removed_count}"
            )
            done += 1
        except Exception as exc:
            print(f"[fail] {relative}: {exc}")
            failed += 1

    print("=" * 60)
    print(f"Done: {done} | Skipped: {skipped} | Failed: {failed}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile as tiff
from cellpose import models

from double_work_utils import (
    DEFAULT_DAPI_TOKEN_REGEX,
    is_excluded_channel_file,
    list_image_files,
    load_module_from_path,
    make_overlay_preview,
    save_preview_image,
    select_channel_files,
    sibling_single_work_dir,
)


# =========================================================
# 02_segment_dapi_double.py
# ---------------------------------------------------------
# Segment nuclei from DAPI images only.
# This double-staining workflow intentionally ignores Oct-4
# at the segmentation stage.
# =========================================================

mode = globals().get("mode", 1)
SUZUI_ROOT = Path(globals().get("SUZUI_ROOT", r"F:\Suzui"))
ANALYSIS_ROOT = Path(globals().get("ANALYSIS_ROOT", SUZUI_ROOT / "analysis_out"))
TRAINING_ROOT = Path(globals().get("TRAINING_ROOT", SUZUI_ROOT / "training data"))
DATASET_NAME = globals().get("DATASET_NAME", r"paper_Oct-4\Tic-SNL,Rac1,Oct-4x")
OUTPUT_DATASET_NAME = globals().get(
    "OUTPUT_DATASET_NAME",
    DATASET_NAME.replace("\\", "_").replace("/", "_"),
)

DAPI_TOKEN_REGEX = globals().get("DAPI_TOKEN_REGEX", DEFAULT_DAPI_TOKEN_REGEX)

if mode == 1:
    DAPI_IMAGE_DIR = Path(globals().get("DAPI_IMAGE_DIR", SUZUI_ROOT / Path(DATASET_NAME)))
    OUT_DIR = Path(globals().get("MASK_DIR", ANALYSIS_ROOT / OUTPUT_DATASET_NAME / "masks_double"))
    print("processing double-staining inference data")
elif mode == 2:
    DAPI_IMAGE_DIR = Path(globals().get("DAPI_IMAGE_DIR", TRAINING_ROOT / "SNL"))
    OUT_DIR = Path(globals().get("MASK_DIR", ANALYSIS_ROOT / "masks_training_double" / "SNL"))
    print("processing double-staining SNL training data")
elif mode == 3:
    DAPI_IMAGE_DIR = Path(globals().get("DAPI_IMAGE_DIR", TRAINING_ROOT / "MEF"))
    OUT_DIR = Path(globals().get("MASK_DIR", ANALYSIS_ROOT / "masks_training_double" / "MEF"))
    print("processing double-staining MEF training data")
else:
    raise ValueError("Unsupported mode. Expected 1, 2, or 3.")

OUT_DIR.mkdir(parents=True, exist_ok=True)

SAVE_PREVIEW = True
PREVIEW_DIR = OUT_DIR / "preview_tif"
PREVIEW_PNG_DIR = OUT_DIR / "preview_png"
if SAVE_PREVIEW:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_PNG_DIR.mkdir(parents=True, exist_ok=True)

USE_GPU = True
PRETRAINED_MODEL = "cpsam"
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0
DIAMETER = None
DEFAULT_PIXEL_SIZE_UM = globals().get("DEFAULT_PIXEL_SIZE_UM", None)
MIN_MASK_AREA_UM2 = globals().get("MIN_MASK_AREA_UM2", None)
MAX_MASK_AREA_UM2 = globals().get("MAX_MASK_AREA_UM2", None)


def normalize_to_u8(img: np.ndarray) -> np.ndarray:
    arr = img.astype(np.float32)
    p1, p99 = np.percentile(arr, [1, 99])
    arr = np.clip((arr - p1) / (p99 - p1 + 1e-6), 0, 1)
    return (arr * 255).astype(np.uint8)


def to_three_channel(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.ndim == 3:
        if img.shape[-1] == 3:
            return img
        if img.shape[-1] == 1:
            return np.repeat(img, 3, axis=-1)
        if img.shape[0] == 3 and img.shape[-1] != 3:
            return np.moveaxis(img, 0, -1)
        if img.shape[0] == 1 and img.shape[-1] != 1:
            return np.repeat(np.moveaxis(img, 0, -1), 3, axis=-1)
    raise ValueError(f"Unsupported image shape for Cellpose input: {img.shape}")


def first_channel(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[-1] >= 1:
        return img[..., 0]
    if img.ndim == 3 and img.shape[0] >= 1:
        return img[0]
    raise ValueError(f"Unsupported image shape: {img.shape}")


def relabel_sequential(mask: np.ndarray) -> np.ndarray:
    uniq = np.unique(mask)
    uniq = uniq[uniq > 0]
    if len(uniq) == 0:
        return np.zeros_like(mask, dtype=np.int32)
    lut = np.zeros(int(mask.max()) + 1, dtype=np.int32)
    lut[uniq] = np.arange(1, len(uniq) + 1, dtype=np.int32)
    return lut[mask]


def filter_masks_by_area_um2(
    mask: np.ndarray,
    pixel_area_um2: float,
    min_area_um2: float | None = None,
    max_area_um2: float | None = None,
) -> np.ndarray:
    if min_area_um2 is None and max_area_um2 is None:
        return mask.astype(np.int32, copy=False)

    areas = np.bincount(mask.ravel())
    keep = np.ones(len(areas), dtype=bool)
    keep[0] = False
    if min_area_um2 is not None:
        keep &= (areas * pixel_area_um2) >= float(min_area_um2)
    if max_area_um2 is not None:
        keep &= (areas * pixel_area_um2) <= float(max_area_um2)

    out = mask.copy()
    lut = np.zeros(len(areas), dtype=bool)
    lut[np.where(~keep)[0]] = True
    out[lut[out]] = 0
    return relabel_sequential(out)


def summarize_mask_physical_stats(mask: np.ndarray, row_um: float, col_um: float) -> dict[str, float]:
    areas_px = np.bincount(mask.ravel())[1:]
    if len(areas_px) == 0:
        return {"count": 0}
    pixel_area_um2 = row_um * col_um
    areas_um2 = areas_px * pixel_area_um2
    eqdiam_um = 2.0 * np.sqrt(areas_um2 / np.pi)
    return {
        "count": int(mask.max()),
        "median_area_um2": float(np.median(areas_um2)),
        "median_eqdiam_um": float(np.median(eqdiam_um)),
        "q25_eqdiam_um": float(np.percentile(eqdiam_um, 25)),
        "q75_eqdiam_um": float(np.percentile(eqdiam_um, 75)),
    }


def main():
    if not DAPI_IMAGE_DIR.exists():
        raise FileNotFoundError(f"DAPI image folder not found: {DAPI_IMAGE_DIR}")

    all_files = list_image_files(DAPI_IMAGE_DIR)
    excluded_files = [p for p in all_files if is_excluded_channel_file(p)]
    files = select_channel_files(all_files, DAPI_TOKEN_REGEX)
    if not files:
        raise RuntimeError(f"No DAPI tif files found in: {DAPI_IMAGE_DIR}")

    print(f"DAPI image folder : {DAPI_IMAGE_DIR}")
    print(f"Mask output folder: {OUT_DIR}")
    print(f"Found {len(files)} DAPI tif files")
    if excluded_files:
        print(f"Excluded {len(excluded_files)} brightfield tif files")
        for path in excluded_files[:10]:
            print(f"  - {path.name}")

    base_dir = sibling_single_work_dir(Path(__file__))
    base03_path = base_dir / "03_feature_extractior.py"
    base03 = load_module_from_path("base03_double_seg", base03_path)
    base03.DEFAULT_PIXEL_SIZE_UM = DEFAULT_PIXEL_SIZE_UM

    model = models.CellposeModel(gpu=USE_GPU, pretrained_model=PRETRAINED_MODEL)

    done = 0
    skipped = 0
    failed = 0

    for img_path in files:
        out_npy = OUT_DIR / f"{img_path.stem}_mask.npy"
        if out_npy.exists():
            print(f"[skip] {img_path.name} -> mask exists")
            skipped += 1
            continue

        try:
            img = tiff.imread(img_path)
            gray = first_channel(img)
            img_cp = to_three_channel(img)
            row_um, col_um, pixel_source = base03.extract_pixel_size_um(img_path)
            pixel_area_um2 = row_um * col_um
            masks, _, _ = model.eval(
                img_cp,
                flow_threshold=FLOW_THRESHOLD,
                cellprob_threshold=CELLPROB_THRESHOLD,
                diameter=DIAMETER,
            )
            masks = filter_masks_by_area_um2(
                masks.astype(np.int32, copy=False),
                pixel_area_um2=pixel_area_um2,
                min_area_um2=MIN_MASK_AREA_UM2,
                max_area_um2=MAX_MASK_AREA_UM2,
            )

            np.save(out_npy, masks.astype(np.int32))
            if SAVE_PREVIEW:
                preview = make_overlay_preview(gray, masks)
                save_preview_image(
                    preview,
                    out_tif=PREVIEW_DIR / f"{img_path.stem}_overlay.tif",
                    out_png=PREVIEW_PNG_DIR / f"{img_path.stem}_overlay.png",
                )

            stats = summarize_mask_physical_stats(masks, row_um=row_um, col_um=col_um)
            print(
                f"[ok] {img_path.name} -> {out_npy.name} | nuclei={int(masks.max())} | "
                f"pixel_size_mean_um={(row_um + col_um) / 2.0:.4f} ({pixel_source}) | "
                f"median_eqdiam_um={stats.get('median_eqdiam_um', float('nan')):.2f} | "
                f"median_area_um2={stats.get('median_area_um2', float('nan')):.2f}"
            )
            done += 1
        except Exception as exc:
            print(f"[fail] {img_path.name}: {exc}")
            failed += 1

    print("=" * 60)
    print(f"Done   : {done}")
    print(f"Skipped: {skipped}")
    print(f"Failed : {failed}")
    print(f"Masks saved to: {OUT_DIR}")
    if SAVE_PREVIEW:
        print(f"Preview saved to: {PREVIEW_DIR}")
        print(f"PNG preview saved to: {PREVIEW_PNG_DIR}")


if __name__ == "__main__":
    main()

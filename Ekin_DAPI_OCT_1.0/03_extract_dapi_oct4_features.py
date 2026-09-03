from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff

from double_work_utils import (
    DEFAULT_DAPI_TOKEN_REGEX,
    DEFAULT_OCT4_TOKEN_REGEX,
    add_pair_metadata,
    build_pair_records,
    make_overlay_preview,
    save_preview_image,
    sibling_single_work_dir,
    load_module_from_path,
)


# =========================================================
# 03_extract_dapi_oct4_features.py
# ---------------------------------------------------------
# Double-staining feature extraction:
# - DAPI image: morphology + neighborhood features only
# - Oct-4 image: intensity features only
# - The same DAPI-derived mask is reused for both channels
# =========================================================

mode = globals().get("mode", 2)
SUZUI_ROOT = Path(globals().get("SUZUI_ROOT", r"F:\Suzui"))
ANALYSIS_ROOT = Path(globals().get("ANALYSIS_ROOT", SUZUI_ROOT / "analysis_out"))
TRAINING_ROOT = Path(globals().get("TRAINING_ROOT", SUZUI_ROOT / "training data"))
DATASET_NAME = globals().get("DATASET_NAME", r"paper_Oct-4\Tic-SNL,Rac1,Oct-4x")
OUTPUT_DATASET_NAME = globals().get(
    "OUTPUT_DATASET_NAME",
    DATASET_NAME.replace("\\", "_").replace("/", "_"),
)

DAPI_TOKEN_REGEX = globals().get("DAPI_TOKEN_REGEX", DEFAULT_DAPI_TOKEN_REGEX)
OCT4_TOKEN_REGEX = globals().get("OCT4_TOKEN_REGEX", DEFAULT_OCT4_TOKEN_REGEX)

DEFAULT_PIXEL_SIZE_UM = globals().get("DEFAULT_PIXEL_SIZE_UM", None)
RUN_QC_AFTER_EXTRACTION = globals().get("RUN_QC_AFTER_EXTRACTION", True)
SAVE_OCT4_OVERLAY_PREVIEW = globals().get("SAVE_OCT4_OVERLAY_PREVIEW", True)

# =========================
# Path settings / 路径设置
# Keep this section close to the single-channel script so the
# editable dataset paths are obvious at a glance.
# =========================
if mode == 1:
    DATASET_LABEL = f"{OUTPUT_DATASET_NAME}_double"
    DAPI_IMAGE_DIR = Path(globals().get("DAPI_IMAGE_DIR", SUZUI_ROOT / Path(DATASET_NAME)))
    OCT4_IMAGE_DIR = Path(globals().get("OCT4_IMAGE_DIR", DAPI_IMAGE_DIR))
    MASK_DIR = Path(globals().get("MASK_DIR", ANALYSIS_ROOT / OUTPUT_DATASET_NAME / "masks_double"))
    OUT_DIR = Path(globals().get("OUT_DIR", ANALYSIS_ROOT / OUTPUT_DATASET_NAME / "features_double"))
    print("processing double-staining inference data")
elif mode == 2:
    TRAINING_SET_NAME = globals().get("TRAINING_SET_NAME", "SNL")
    DATASET_LABEL = f"{TRAINING_SET_NAME}_training_double"
    DAPI_IMAGE_DIR = Path(globals().get("DAPI_IMAGE_DIR", TRAINING_ROOT / TRAINING_SET_NAME))
    OCT4_IMAGE_DIR = Path(globals().get("OCT4_IMAGE_DIR", DAPI_IMAGE_DIR))
    MASK_DIR = Path(globals().get("MASK_DIR", ANALYSIS_ROOT / "masks_training_double" / TRAINING_SET_NAME))
    OUT_DIR = Path(globals().get("OUT_DIR", ANALYSIS_ROOT / "features_training_double" / TRAINING_SET_NAME))
    print(f"processing {TRAINING_SET_NAME} training data")
elif mode == 3:
    TRAINING_SET_NAME = globals().get("TRAINING_SET_NAME", "MEF")
    DATASET_LABEL = f"{TRAINING_SET_NAME}_training_double"
    DAPI_IMAGE_DIR = Path(globals().get("DAPI_IMAGE_DIR", TRAINING_ROOT / TRAINING_SET_NAME))
    OCT4_IMAGE_DIR = Path(globals().get("OCT4_IMAGE_DIR", DAPI_IMAGE_DIR))
    MASK_DIR = Path(globals().get("MASK_DIR", ANALYSIS_ROOT / "masks_training_double" / TRAINING_SET_NAME))
    OUT_DIR = Path(globals().get("OUT_DIR", ANALYSIS_ROOT / "features_training_double" / TRAINING_SET_NAME))
    print(f"processing {TRAINING_SET_NAME} training data")
else:
    raise ValueError("Unsupported mode. Expected 1, 2, or 3.")

OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_image_summary(summary: dict, pair_key: str, dapi_path: Path, oct4_path: Path, stain_source: str) -> dict:
    out = dict(summary)
    out["pair_key"] = pair_key
    out["dapi_image_name"] = dapi_path.stem
    out["oct4_image_name"] = oct4_path.stem
    out["intensity_source_stain"] = stain_source
    return out


def keep_image_intensity_columns(df: pd.DataFrame, base03) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    metadata_cols = [
        "image_name",
        "pair_key",
        "dapi_image_name",
        "oct4_image_name",
        "intensity_source_stain",
    ]
    keep_cols = [c for c in metadata_cols if c in df.columns]
    keep_cols += [
        c for c in df.columns
        if c not in keep_cols and base03.is_intensity_related_column(c)
    ]
    return df.loc[:, keep_cols].copy()


def run_qc_stage(dataset_label: str, dapi_dir: Path, mask_dir: Path, out_dir: Path) -> None:
    qc_script = Path(__file__).with_name("03-1_qc_filter_double.py")
    if not qc_script.exists():
        raise FileNotFoundError(f"QC script not found: {qc_script}")

    print("\n" + "=" * 70)
    print(f"Starting double-staining QC stage for: {dataset_label}")
    print("=" * 70)

    runpy.run_path(
        str(qc_script),
        run_name="__main__",
        init_globals={
            "DATASET_LABEL": dataset_label,
            "DAPI_IMAGE_DIR": dapi_dir,
            "MASK_DIR": mask_dir,
            "FEATURE_DIR": out_dir,
        },
    )


def save_oct4_overlay_previews(pair_records, out_dir: Path, base03) -> None:
    if not SAVE_OCT4_OVERLAY_PREVIEW:
        return

    preview_tif_dir = out_dir / "preview_oct4_tif"
    preview_png_dir = out_dir / "preview_oct4_png"
    preview_tif_dir.mkdir(parents=True, exist_ok=True)
    preview_png_dir.mkdir(parents=True, exist_ok=True)

    qc_keep_dir = out_dir / "qc_training" / "qc_keep_masks"
    saved = 0
    skipped = 0

    for record in pair_records:
        qc_keep_mask = qc_keep_dir / f"{record.image_name}_qc_keep_mask.npy"
        mask_path = qc_keep_mask if qc_keep_mask.exists() else record.mask_path
        if not mask_path.exists():
            skipped += 1
            continue

        mask = np.load(mask_path)
        oct4_img = base03.ensure_2d_image(tiff.imread(record.oct4_path))
        if mask.shape != oct4_img.shape:
            skipped += 1
            continue

        preview = make_overlay_preview(oct4_img, mask)
        save_preview_image(
            preview,
            out_tif=preview_tif_dir / f"{record.image_name}_overlay_on_oct4.tif",
            out_png=preview_png_dir / f"{record.image_name}_overlay_on_oct4.png",
        )
        saved += 1

    print(f"Oct-4 overlay previews saved: {saved}")
    if skipped:
        print(f"Oct-4 overlay previews skipped: {skipped}")
    print(f"Oct-4 overlay tif dir : {preview_tif_dir}")
    print(f"Oct-4 overlay png dir : {preview_png_dir}")


def main() -> None:
    nucleus_csv = OUT_DIR / "nucleus_features.csv"
    image_csv = OUT_DIR / "image_features.csv"
    intensity_csv = OUT_DIR / "nucleus_intensity_features.csv"
    image_intensity_csv = OUT_DIR / "image_intensity_features.csv"
    pairing_report_csv = OUT_DIR / "double_pairing_report.csv"

    base_dir = sibling_single_work_dir(Path(__file__))
    base03_path = base_dir / "03_feature_extractior.py"
    base03 = load_module_from_path("base03_double_extract", base03_path)
    base03.DEFAULT_PIXEL_SIZE_UM = DEFAULT_PIXEL_SIZE_UM

    pair_records, pairing_report = build_pair_records(
        dapi_dir=DAPI_IMAGE_DIR,
        oct4_dir=OCT4_IMAGE_DIR,
        mask_dir=MASK_DIR,
        dapi_token_regex=DAPI_TOKEN_REGEX,
        oct4_token_regex=OCT4_TOKEN_REGEX,
        out_report_csv=pairing_report_csv,
    )

    print(f"DAPI image folder : {DAPI_IMAGE_DIR}")
    print(f"Oct-4 image folder: {OCT4_IMAGE_DIR}")
    print(f"Mask folder       : {MASK_DIR}")
    print(f"Output folder     : {OUT_DIR}")
    print(f"Valid paired triplets: {len(pair_records)}")
    if len(pairing_report) != len(pair_records):
        print(f"[warn] Some files were skipped. Pairing report saved to: {pairing_report_csv}")

    nucleus_feature_tables: list[pd.DataFrame] = []
    nucleus_intensity_tables: list[pd.DataFrame] = []
    image_feature_rows: list[dict] = []
    image_intensity_rows: list[dict] = []

    done = 0
    failed = 0

    for record in pair_records:
        try:
            mask = np.load(record.mask_path)
            dapi_img = base03.ensure_2d_image(tiff.imread(record.dapi_path))
            oct4_img = base03.ensure_2d_image(tiff.imread(record.oct4_path))
            if mask.shape != dapi_img.shape:
                raise ValueError(
                    f"Mask shape {mask.shape} does not match DAPI image shape {dapi_img.shape} "
                    f"for {record.image_name}"
                )
            if oct4_img.shape != dapi_img.shape:
                raise ValueError(
                    f"Oct-4 image shape {oct4_img.shape} does not match DAPI image shape {dapi_img.shape} "
                    f"for {record.image_name}"
                )

            row_um, col_um, pixel_source = base03.extract_pixel_size_um(record.dapi_path)
            try:
                oct4_row_um, oct4_col_um, oct4_pixel_source = base03.extract_pixel_size_um(record.oct4_path)
            except Exception:
                oct4_row_um, oct4_col_um, oct4_pixel_source = row_um, col_um, "fallback_to_dapi"

            print(
                "[meta] "
                f"{record.image_name} -> DAPI pixel_size_row_um={row_um:.6f}, "
                f"pixel_size_col_um={col_um:.6f}, source={pixel_source}; "
                f"Oct-4 source={oct4_pixel_source}"
            )

            dapi_full_df = base03.extract_nucleus_features(
                image=dapi_img,
                mask=mask,
                image_name=record.image_name,
                pixel_size_row_um=row_um,
                pixel_size_col_um=col_um,
            )
            dapi_full_df = add_pair_metadata(dapi_full_df, record.dapi_path, record.oct4_path, record.pair_key)
            dapi_full_df["feature_source_stain"] = "DAPI"
            dapi_feature_df, _, _ = base03.split_feature_and_intensity_tables(
                dapi_full_df,
                base03.NUCLEUS_KEY_COLS,
            )

            oct4_full_df = base03.extract_nucleus_features(
                image=oct4_img,
                mask=mask,
                image_name=record.image_name,
                pixel_size_row_um=oct4_row_um,
                pixel_size_col_um=oct4_col_um,
            )
            oct4_full_df = add_pair_metadata(oct4_full_df, record.dapi_path, record.oct4_path, record.pair_key)
            oct4_full_df["intensity_source_stain"] = "Oct-4"
            _, oct4_intensity_df, _ = base03.split_feature_and_intensity_tables(
                oct4_full_df,
                base03.NUCLEUS_KEY_COLS,
            )

            image_feature_summary = base03.summarize_image_features(
                dapi_feature_df,
                image_name=record.image_name,
                image_shape=dapi_img.shape,
                pixel_size_row_um=row_um,
                pixel_size_col_um=col_um,
                pixel_size_source=pixel_source,
            )
            image_feature_summary = build_image_summary(
                image_feature_summary,
                pair_key=record.pair_key,
                dapi_path=record.dapi_path,
                oct4_path=record.oct4_path,
                stain_source="DAPI",
            )

            image_intensity_summary = base03.summarize_image_features(
                oct4_intensity_df,
                image_name=record.image_name,
                image_shape=oct4_img.shape,
                pixel_size_row_um=oct4_row_um,
                pixel_size_col_um=oct4_col_um,
                pixel_size_source=oct4_pixel_source,
            )
            image_intensity_summary = build_image_summary(
                image_intensity_summary,
                pair_key=record.pair_key,
                dapi_path=record.dapi_path,
                oct4_path=record.oct4_path,
                stain_source="Oct-4",
            )

            nucleus_feature_tables.append(dapi_feature_df)
            nucleus_intensity_tables.append(oct4_intensity_df)
            image_feature_rows.append(image_feature_summary)
            image_intensity_rows.append(image_intensity_summary)

            print(
                f"[ok] {record.image_name} -> nuclei={len(dapi_feature_df)} | "
                f"DAPI morphology saved, Oct-4 intensity saved"
            )
            done += 1
        except Exception as exc:
            print(f"[fail] {record.image_name}: {exc}")
            failed += 1

    if not nucleus_feature_tables:
        raise RuntimeError("No paired images were processed successfully.")

    nucleus_feature_df = pd.concat(nucleus_feature_tables, ignore_index=True)
    nucleus_intensity_df = pd.concat(nucleus_intensity_tables, ignore_index=True)
    image_feature_df = pd.DataFrame(image_feature_rows)
    image_intensity_df = pd.DataFrame(image_intensity_rows)
    image_intensity_df = keep_image_intensity_columns(image_intensity_df, base03)

    nucleus_feature_df.to_csv(nucleus_csv, index=False, encoding="utf-8-sig")
    nucleus_intensity_df.to_csv(intensity_csv, index=False, encoding="utf-8-sig")
    image_feature_df.to_csv(image_csv, index=False, encoding="utf-8-sig")
    image_intensity_df.to_csv(image_intensity_csv, index=False, encoding="utf-8-sig")

    print("=" * 72)
    print(f"Done                       : {done}")
    print(f"Failed                     : {failed}")
    print(f"DAPI feature CSV           : {nucleus_csv}")
    print(f"Oct-4 intensity CSV        : {intensity_csv}")
    print(f"DAPI image feature CSV     : {image_csv}")
    print(f"Oct-4 image intensity CSV  : {image_intensity_csv}")
    print(f"Pairing report             : {pairing_report_csv}")
    print("Feature split:")
    print("- nucleus_features.csv contains DAPI morphology and neighborhood features only")
    print("- nucleus_intensity_features.csv contains Oct-4 intensity features only")

    if RUN_QC_AFTER_EXTRACTION:
        run_qc_stage(DATASET_LABEL, DAPI_IMAGE_DIR, MASK_DIR, OUT_DIR)

    save_oct4_overlay_previews(pair_records, OUT_DIR, base03)


if __name__ == "__main__":
    main()

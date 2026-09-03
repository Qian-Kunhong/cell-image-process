from __future__ import annotations

from pathlib import Path

import pandas as pd

from double_work_utils import load_module_from_path, sibling_single_work_dir


# =========================================================
# 03-1_qc_filter_double.py
# ---------------------------------------------------------
# QC worker for the double-staining workflow.
#
# Design choice:
# - 03_extract_dapi_oct4_features.py is the only place that decides
#   mode, dataset, and all paths.
# - This script only consumes explicit directories passed in from 03.
# - If you want to run this file alone, set the three path globals
#   below directly instead of using another mode switch.
# =========================================================

DATASET_LABEL = globals().get("DATASET_LABEL", "double_staining_dataset")
DAPI_IMAGE_DIR = Path(globals().get("DAPI_IMAGE_DIR", ""))
MASK_DIR = Path(globals().get("MASK_DIR", ""))
FEATURE_DIR = Path(globals().get("FEATURE_DIR", ""))


def require_explicit_path(path_value: Path, name: str) -> Path:
    if str(path_value).strip() in {"", "."}:
        raise ValueError(
            f"{name} is not configured for 03-1_qc_filter_double.py.\n"
            "Pass it from 03_extract_dapi_oct4_features.py or set it explicitly "
            "at the top of this file before running standalone."
        )
    return path_value


def configure_base_qc(base_module, dataset_label: str, img_dir: Path, mask_dir: Path, feature_dir: Path) -> None:
    # Double-staining default:
    # keep original masks intact and export QC masks separately.
    base_module.MINIMAL_OUTPUT_MODE = False
    base_module.SAVE_QC_MASKS = True
    base_module.SAVE_QC_PREVIEWS = False
    base_module.OVERWRITE_MASK_WITH_QC = False
    base_module.OVERWRITE_NUCLEUS_FEATURE_CSV_WITH_QC_KEEP = True
    base_module.KEEP_REMOVED_OBJECTS_CSV = False
    base_module.KEEP_IMAGE_SUMMARY_CSV = False
    base_module.KEEP_THRESHOLDS_JSON = False

    base_module.CFG = {
        "name": dataset_label,
        "IMG_DIR": img_dir,
        "MASK_DIR": mask_dir,
        "FEATURE_DIR": feature_dir,
        "message": f"processing {dataset_label}",
    }
    base_module.IMG_DIR = img_dir
    base_module.MASK_DIR = mask_dir
    base_module.FEATURE_DIR = feature_dir

    base_module.NUCLEUS_CSV = feature_dir / "nucleus_features.csv"
    base_module.IMAGE_CSV = feature_dir / "image_features.csv"
    base_module.NUCLEUS_INTENSITY_CSV = feature_dir / "nucleus_intensity_features.csv"
    base_module.IMAGE_INTENSITY_CSV = feature_dir / "image_intensity_features.csv"

    base_module.OUT_DIR = feature_dir / "qc_training"
    base_module.OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_module.OUT_QC_KEEP = base_module.OUT_DIR / "nucleus_features_qc.csv"
    base_module.OUT_QC_REMOVE = base_module.OUT_DIR / "nucleus_features_removed.csv"
    base_module.OUT_IMAGE_SUMMARY = base_module.OUT_DIR / "image_qc_summary.csv"
    base_module.OUT_THRESHOLDS_JSON = base_module.OUT_DIR / "qc_thresholds.json"

    base_module.QC_KEEP_MASK_DIR = base_module.OUT_DIR / "qc_keep_masks"
    base_module.QC_REMOVE_MASK_DIR = base_module.OUT_DIR / "qc_removed_masks"
    base_module.QC_PREVIEW_DIR = base_module.OUT_DIR / "qc_preview_tif"
    if base_module.SAVE_QC_MASKS:
        base_module.QC_KEEP_MASK_DIR.mkdir(parents=True, exist_ok=True)
        base_module.QC_REMOVE_MASK_DIR.mkdir(parents=True, exist_ok=True)
    if base_module.SAVE_QC_PREVIEWS:
        base_module.QC_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    def merge_intensity_table_double_safe(
        feature_df: pd.DataFrame,
        intensity_csv: Path,
        key_cols: list[str],
        table_name: str,
    ) -> pd.DataFrame:
        if not intensity_csv.exists():
            return feature_df

        intensity_df = pd.read_csv(intensity_csv)
        merge_keys = [c for c in key_cols if c in feature_df.columns and c in intensity_df.columns]
        if len(merge_keys) != len(key_cols):
            raise ValueError(f"{table_name} is missing required merge keys: {key_cols}")

        is_image_level_table = "image" in intensity_csv.name.lower()
        if is_image_level_table:
            keep_cols = set(merge_keys)
            keep_cols.update(
                c for c in intensity_df.columns
                if getattr(base_module, "is_intensity_related_column")(c)
            )
            keep_cols.update(
                c for c in [
                    "pair_key",
                    "dapi_image_name",
                    "oct4_image_name",
                    "intensity_source_stain",
                ]
                if c in intensity_df.columns
            )
            intensity_df = intensity_df.loc[:, [c for c in intensity_df.columns if c in keep_cols]].copy()
            intensity_df.to_csv(intensity_csv, index=False, encoding="utf-8-sig")

        dup_mask = intensity_df.duplicated(subset=merge_keys, keep=False)
        if dup_mask.any():
            raise ValueError(f"{table_name} has duplicated rows for keys: {merge_keys}")

        value_cols = [c for c in intensity_df.columns if c not in merge_keys]
        if not value_cols:
            return feature_df

        overlap_cols = [c for c in value_cols if c in feature_df.columns]
        if overlap_cols:
            intensity_df = intensity_df.drop(columns=overlap_cols)
            value_cols = [c for c in intensity_df.columns if c not in merge_keys]
            if not value_cols:
                return feature_df

        return feature_df.merge(intensity_df, on=merge_keys, how="left")

    base_module.merge_intensity_table = merge_intensity_table_double_safe


def main() -> None:
    img_dir = require_explicit_path(DAPI_IMAGE_DIR, "DAPI_IMAGE_DIR")
    mask_dir = require_explicit_path(MASK_DIR, "MASK_DIR")
    feature_dir = require_explicit_path(FEATURE_DIR, "FEATURE_DIR")

    print(f"Running double-staining QC for: {DATASET_LABEL}")

    base_dir = sibling_single_work_dir(Path(__file__))
    base_qc_path = base_dir / "03-1_qc_filter.py"
    base_qc = load_module_from_path("base_qc_double_wrapper", base_qc_path)
    configure_base_qc(base_qc, DATASET_LABEL, img_dir, mask_dir, feature_dir)
    base_qc.main()


if __name__ == "__main__":
    main()

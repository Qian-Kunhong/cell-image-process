from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from double_work_utils import (
    load_module_from_path,
    save_score_intensity_relationship,
    sibling_single_work_dir,
)


# =========================================================
# 05_infer_dapi_oct4.py
# ---------------------------------------------------------
# Train and infer with DAPI-derived features only, then
# compare the predicted DAPI deviation score against
# Oct-4 mean intensity in the new data.
# =========================================================

SUZUI_ROOT = Path(globals().get("SUZUI_ROOT", r"F:\Suzui"))
ANALYSIS_ROOT = Path(globals().get("ANALYSIS_ROOT", SUZUI_ROOT / "analysis_out"))
TRAINING_ROOT = Path(globals().get("TRAINING_ROOT", SUZUI_ROOT / "training data"))
TRAINING_SET_NAME = globals().get("TRAINING_SET_NAME", "SNL")
PIPELINE_VARIANT = globals().get("PIPELINE_VARIANT", "05c")
DATASET_NAME = globals().get("DATASET_NAME", r"paper_Oct-4\Tic-SNL,Rac1,Oct-4x")
OUTPUT_DATASET_NAME = globals().get(
    "OUTPUT_DATASET_NAME",
    DATASET_NAME.replace("\\", "_").replace("/", "_"),
)
TRAIN_CLUSTER_SUBDIR = globals().get(
    "TRAIN_CLUSTER_SUBDIR",
    "cluster_neighbor_innerfit_conservative_double",
)

TRAIN_FEATURE_CSV = Path(
    globals().get(
        "TRAIN_FEATURE_CSV",
        ANALYSIS_ROOT / "features_training_double" / TRAINING_SET_NAME / "nucleus_features.csv",
    )
)
TRAIN_LABEL_CSV = Path(
    globals().get(
        "TRAIN_LABEL_CSV",
        ANALYSIS_ROOT
        / "features_training_double"
        / TRAINING_SET_NAME
        / TRAIN_CLUSTER_SUBDIR
        / "nucleus_features_qc_clustered.csv",
    )
)

# =========================
# Inference path settings / 推理路径设置
# Usually you only need to edit DATASET_NAME.
# =========================
INFERENCE_OUTPUT_LABEL = globals().get("INFERENCE_OUTPUT_LABEL", OUTPUT_DATASET_NAME)
INFER_DAPI_IMAGE_DIR = Path(globals().get("INFER_DAPI_IMAGE_DIR", SUZUI_ROOT / Path(DATASET_NAME)))
INFER_FEATURE_DIR = Path(globals().get("INFER_FEATURE_DIR", ANALYSIS_ROOT / OUTPUT_DATASET_NAME / "features_double"))
INFER_FEATURE_CSV = Path(globals().get("INFER_FEATURE_CSV", INFER_FEATURE_DIR / "nucleus_features.csv"))
INFER_INTENSITY_CSV = Path(globals().get("INFER_INTENSITY_CSV", INFER_FEATURE_DIR / "nucleus_intensity_features.csv"))
INFER_MASK_DIR = Path(globals().get("INFER_MASK_DIR", ANALYSIS_ROOT / OUTPUT_DATASET_NAME / "masks_double"))


def configure_base05(base_module) -> None:
    if PIPELINE_VARIANT not in base_module.PIPELINE_VARIANTS:
        raise ValueError(f"Unsupported PIPELINE_VARIANT: {PIPELINE_VARIANT}")

    base_module.SUZUI_ROOT = SUZUI_ROOT
    base_module.ANALYSIS_ROOT = ANALYSIS_ROOT
    base_module.TRAINING_ROOT = TRAINING_ROOT
    base_module.TRAINING_SET_NAME = TRAINING_SET_NAME
    base_module.INFERENCE_SET_NAME = INFERENCE_OUTPUT_LABEL
    base_module.PIPELINE_VARIANT = PIPELINE_VARIANT
    base_module.PIPELINE_CONFIG = base_module.PIPELINE_VARIANTS[PIPELINE_VARIANT]

    base_module.TRAIN_FEATURE_CSV = TRAIN_FEATURE_CSV
    base_module.TRAIN_LABEL_CSV = TRAIN_LABEL_CSV
    base_module.TRAIN_LABEL_COL = "final_state_label"

    base_module.INFER_IMAGE_DIR = INFER_DAPI_IMAGE_DIR
    base_module.INFER_FEATURE_CSV = INFER_FEATURE_CSV
    base_module.INFER_MASK_DIR = INFER_MASK_DIR

    output_root = ANALYSIS_ROOT / f"{INFERENCE_OUTPUT_LABEL}{base_module.PIPELINE_CONFIG['output_suffix']}_double"
    base_module.OUTPUT_ROOT = output_root
    base_module.ABSOLUTE_FEATURE_NORMALIZATION = base_module.PIPELINE_CONFIG["absolute_feature_normalization"]
    base_module.INFER_UMAP_MODE = base_module.PIPELINE_CONFIG["infer_umap_mode"]
    base_module.EXCLUDE_INTENSITY_FEATURES = True


def resolve_train_label_csv() -> Path:
    candidates = []
    if TRAIN_FEATURE_CSV not in candidates:
        candidates.append(TRAIN_FEATURE_CSV)
    if TRAIN_LABEL_CSV not in candidates:
        candidates.append(TRAIN_LABEL_CSV)

    checked = []
    for candidate in candidates:
        checked.append(candidate)
        if not candidate.exists():
            continue
        try:
            label_head = pd.read_csv(candidate, nrows=5)
        except Exception as exc:
            raise RuntimeError(f"Failed to read training label CSV: {candidate}") from exc
        if "final_state_label" in label_head.columns:
            return candidate

    checked_text = "\n".join(f"- {path}" for path in checked)
    raise RuntimeError(
        "No training label CSV contains `final_state_label`.\n"
        f"Checked:\n{checked_text}\n"
        "Please run 04_cluster_dapi_oct4.py first or point TRAIN_LABEL_CSV to the clustered training table."
    )


def validate_inference_inputs() -> None:
    missing: list[Path] = []
    for path in [INFER_FEATURE_CSV, INFER_INTENSITY_CSV]:
        if not path.exists():
            missing.append(path)

    if not missing:
        return

    missing_text = "\n".join(f"- {path}" for path in missing)
    raise FileNotFoundError(
        "Inference input files are missing.\n"
        f"{missing_text}\n"
        "Please run 03_extract_dapi_oct4_features.py with mode = 1 first "
        "to generate the double-staining inference features."
    )


def main() -> None:
    base_dir = sibling_single_work_dir(Path(__file__))
    base05_path = base_dir / "05_distinguish cell type.py"
    base05 = load_module_from_path("base05_double_infer", base05_path)
    resolved_train_label_csv = resolve_train_label_csv()
    validate_inference_inputs()
    configure_base05(base05)
    base05.TRAIN_LABEL_CSV = resolved_train_label_csv

    print("Running double-staining 05 inference:")
    print(f"- Training DAPI features : {TRAIN_FEATURE_CSV}")
    print(f"- Training label CSV     : {resolved_train_label_csv}")
    print(f"- Inference DAPI features: {INFER_FEATURE_CSV}")
    print(f"- Inference Oct-4 CSV    : {INFER_INTENSITY_CSV}")
    print(f"- DAPI image dir         : {INFER_DAPI_IMAGE_DIR}")

    base05.main()

    pred_csv = base05.OUTPUT_ROOT / base05.PREDICTION_CSV_NAME
    pred_df = pd.read_csv(pred_csv)

    relation_csv = base05.OUTPUT_ROOT / "predicted_deviation_score_vs_oct4_mean_intensity.csv"
    relation_fig = base05.OUTPUT_ROOT / "predicted_deviation_score_vs_oct4_mean_intensity.png"
    relation_json = base05.OUTPUT_ROOT / "predicted_deviation_score_vs_oct4_mean_intensity_summary.json"

    summary = save_score_intensity_relationship(
        scored_df=pred_df,
        intensity_csv=INFER_INTENSITY_CSV,
        out_csv=relation_csv,
        out_fig=relation_fig,
        out_summary_json=relation_json,
        score_candidates=["deviation_score", "p_deviated"],
        intensity_col="mean_intensity",
        state_candidates=["predicted_label"],
        score_label="Predicted deviation score",
        intensity_label="Oct-4 mean intensity",
        title_prefix="Predicted DAPI deviation score vs Oct-4 mean intensity",
    )

    print(f"[ok] Oct-4 inference validation CSV saved: {relation_csv}")
    print(f"[ok] Oct-4 inference validation figure saved: {relation_fig}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

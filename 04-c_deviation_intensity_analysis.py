from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


BASE04_PATH = Path(__file__).with_name("04_unsupervised_learning.py")


def load_stage04_module():
    spec = importlib.util.spec_from_file_location("stage04_base", BASE04_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load base module from: {BASE04_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage04 = load_stage04_module()


SUZUI_ROOT = Path(r"F:\Suzui")
ANALYSIS_ROOT = SUZUI_ROOT / "analysis_out"
TRAINING_ROOT = SUZUI_ROOT / "training data"
TRAINING_SET_NAME = "SNL"

INPUT_CSV = ANALYSIS_ROOT / "features_training" / TRAINING_SET_NAME / "nucleus_features.csv"
INTENSITY_CSV = ANALYSIS_ROOT / "features_training" / TRAINING_SET_NAME / "nucleus_intensity_features.csv"
IMAGE_DIR = TRAINING_ROOT / TRAINING_SET_NAME
MASK_DIR = ANALYSIS_ROOT / "masks_training" / TRAINING_SET_NAME

OUT_DIR = ANALYSIS_ROOT / "features_training" / TRAINING_SET_NAME / "deviation_intensity_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_RELATION_CSV = OUT_DIR / "deviation_score_vs_mean_intensity.csv"
OUT_RELATION_FIG = OUT_DIR / "deviation_score_vs_mean_intensity.png"
OUT_SUMMARY_JSON = OUT_DIR / "deviation_intensity_summary.json"


def configure_stage04_globals():
    stage04.INPUT_CSV = INPUT_CSV
    stage04.INTENSITY_CSV = INTENSITY_CSV
    stage04.IMAGE_DIR = IMAGE_DIR
    stage04.ORIG_MASK_DIR = MASK_DIR
    stage04.QC_KEEP_MASK_DIR = MASK_DIR


def main():
    configure_stage04_globals()

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")
    if not INTENSITY_CSV.exists():
        raise FileNotFoundError(f"Intensity CSV not found: {INTENSITY_CSV}")

    print(f"Reading features : {INPUT_CSV}")
    print(f"Reading intensity: {INTENSITY_CSV}")

    df = pd.read_csv(INPUT_CSV)
    df = stage04.add_feature_aliases(df)
    if len(df) == 0:
        raise RuntimeError("Input CSV is empty.")

    features, log1p_cols, feature_weights = stage04.resolve_feature_specs(df, stage04.FEATURE_SPECS, "INPUT_CSV")
    print("[info] Resolved deviation-score model features:")
    print(features)

    keep_meta_cols = [c for c in stage04.PREFERRED_META_COLS if c in df.columns]
    df_meta = df[keep_meta_cols].copy()

    edge_df = stage04.compute_edge_metrics_for_all_rows(df_meta)
    fit_mask = stage04.make_inner_fit_mask(edge_df)
    print(f"[info] Inner-fit nuclei: {int(fit_mask.sum())} / {len(fit_mask)}")

    df_work = df.copy()
    transformed_feature_cols = []
    for col in features:
        new_col = f"{col}__model"
        if col in log1p_cols:
            df_work[new_col] = stage04.safe_log1p(df_work[col])
        else:
            df_work[new_col] = pd.to_numeric(df_work[col], errors="coerce")
        transformed_feature_cols.append(new_col)

    X_df = df_work[transformed_feature_cols].copy()
    _, _, _, _, X_pca_all, _ = stage04.fit_transform_pipeline_on_inner(
        X_df,
        fit_mask,
        feature_weights,
    )

    gmm = GaussianMixture(
        n_components=stage04.N_CLUSTERS,
        covariance_type=stage04.GMM_COVARIANCE_TYPE,
        random_state=stage04.RANDOM_STATE,
    )
    gmm.fit(X_pca_all[fit_mask])
    gmm_prob = gmm.predict_proba(X_pca_all)

    final_labels, p_dev_raw, p_undiff_raw, margin_raw = stage04.assign_state_labels(gmm_prob=gmm_prob)

    df_out = df.copy()
    df_out["gmm_prob_deviated_raw"] = p_dev_raw
    df_out["gmm_prob_undifferentiated_raw"] = p_undiff_raw
    df_out["gmm_prob_margin_dev_minus_undiff_raw"] = margin_raw
    df_out["deviated_score"] = p_dev_raw
    df_out["final_state_label"] = final_labels

    rel_info = stage04.save_deviation_intensity_relation(
        df_out=df_out,
        intensity_csv=INTENSITY_CSV,
        out_csv=OUT_RELATION_CSV,
        out_fig=OUT_RELATION_FIG,
    )
    if rel_info is None:
        raise RuntimeError("Failed to generate deviation-intensity relation outputs.")

    summary = {
        "input_csv": str(INPUT_CSV),
        "intensity_csv": str(INTENSITY_CSV),
        "feature_columns": features,
        "inner_fit_count": int(fit_mask.sum()),
        "total_count": int(len(df_out)),
        "pearson_r": rel_info["pearson_r"],
        "spearman_r": rel_info["spearman_r"],
        "n_valid_rows": rel_info["n_valid_rows"],
        "relation_csv": str(OUT_RELATION_CSV),
        "relation_figure": str(OUT_RELATION_FIG),
    }
    with open(OUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[saved] {OUT_SUMMARY_JSON}")
    print(f"[done] Pearson r = {rel_info['pearson_r']:.4f}, Spearman rho = {rel_info['spearman_r']:.4f}")


if __name__ == "__main__":
    main()

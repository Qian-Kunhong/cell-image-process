from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from cluster_xgb_analysis_common import (
    DEFAULT_INPUT_CSV,
    fit_stage04_cluster_xgb,
    save_json,
    stage05,
)


# ============================================================
# 07_SHAP.py
# ------------------------------------------------------------
# SHAP analysis on the XGBoost model trained from stage-04
# clustered nucleus features.
# ============================================================

# Default: the clustered training CSV overwritten in-place by stage 04.
# If you want another 04-cluster result file, change INPUT_CSV here.
INPUT_CSV = DEFAULT_INPUT_CSV
OUTPUT_DIR = INPUT_CSV.parent / "07_SHAP"
TOP_N_DISPLAY = 10
MAX_DISPLAY = TOP_N_DISPLAY
SHOW_PLOT = False


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def abbreviate_feature_name(name: str) -> str:
    label = str(name)
    replacements = [
        ("__img_rel", "_ir"),
        ("adaptive_", "ad_"),
        ("fixed_", "fx_"),
        ("neighbor", "nb"),
        ("neighbour", "nb"),
        ("distance", "dist"),
        ("intensity", "int"),
        ("equivalent", "eq"),
        ("diameter", "dia"),
        ("circularity", "circ"),
        ("eccentricity", "ecc"),
        ("perimeter", "peri"),
        ("major_axis", "majax"),
        ("minor_axis", "minax"),
        ("local_density", "ldens"),
        ("deviation", "dev"),
        ("cluster", "clu"),
        ("probability", "prob"),
    ]
    for old, new in replacements:
        label = label.replace(old, new)

    if len(label) <= 18:
        return label

    tokens = [t for t in label.replace("-", "_").split("_") if t]
    if len(tokens) >= 2:
        compressed = "".join(t[0] for t in tokens[:-1]) + "_" + tokens[-1][:6]
        if len(compressed) <= 18:
            return compressed

    return label[:18]


def build_unique_short_names(feature_names: list[str]) -> list[str]:
    used: set[str] = set()
    short_names: list[str] = []

    for name in feature_names:
        base = abbreviate_feature_name(name)
        candidate = base
        suffix_idx = 2
        while candidate in used:
            suffix = f"_{suffix_idx}"
            trimmed = base[: max(4, 18 - len(suffix))]
            candidate = f"{trimmed}{suffix}"
            suffix_idx += 1
        used.add(candidate)
        short_names.append(candidate)

    return short_names


def normalize_shap_values(raw_shap_values) -> np.ndarray:
    shap_values = raw_shap_values
    if isinstance(shap_values, list):
        if len(shap_values) == 0:
            raise RuntimeError("SHAP returned an empty list.")
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        if shap_values.shape[-1] == 2:
            shap_values = shap_values[:, :, 1]
        elif shap_values.shape[0] == 2:
            shap_values = shap_values[1]
        else:
            raise RuntimeError(f"Unsupported SHAP output shape: {shap_values.shape}")

    if shap_values.ndim != 2:
        raise RuntimeError(f"Unsupported SHAP output shape: {shap_values.shape}")

    return shap_values


def save_summary_plot(
    shap_values: np.ndarray,
    X_imp: pd.DataFrame,
    feature_display_names: list[str],
    output_png: Path,
) -> None:
    shap.summary_plot(
        shap_values,
        X_imp,
        feature_names=feature_display_names,
        max_display=min(TOP_N_DISPLAY, X_imp.shape[1]),
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_png, dpi=220, bbox_inches="tight")
    if SHOW_PLOT:
        plt.show()
    plt.close()


def save_bar_plot(
    shap_values: np.ndarray,
    X_imp: pd.DataFrame,
    feature_display_names: list[str],
    output_png: Path,
) -> None:
    shap.summary_plot(
        shap_values,
        X_imp,
        feature_names=feature_display_names,
        plot_type="bar",
        max_display=min(TOP_N_DISPLAY, X_imp.shape[1]),
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_png, dpi=220, bbox_inches="tight")
    if SHOW_PLOT:
        plt.show()
    plt.close()


def compute_shap_values_with_fallback(model, X_imp: pd.DataFrame) -> tuple[np.ndarray, str]:
    try:
        explainer = shap.TreeExplainer(model)
        raw_shap_values = explainer.shap_values(X_imp)
        return normalize_shap_values(raw_shap_values), "tree_explainer"
    except Exception as tree_error:
        print(f"[warn] TreeExplainer failed, fallback to XGBoost pred_contribs: {tree_error}")

    try:
        import xgboost as xgb

        booster = model.get_booster()
        dmatrix = xgb.DMatrix(X_imp, feature_names=list(X_imp.columns))
        contrib = booster.predict(
            dmatrix,
            pred_contribs=True,
            validate_features=False,
        )
        contrib = np.asarray(contrib)
        if contrib.ndim != 2 or contrib.shape[1] != X_imp.shape[1] + 1:
            raise RuntimeError(f"Unexpected pred_contribs shape: {contrib.shape}")
        # Last column is model bias term; SHAP feature values are the first N columns.
        return contrib[:, :-1], "xgb_pred_contribs"
    except Exception as fallback_error:
        raise RuntimeError(
            "Both SHAP TreeExplainer and XGBoost pred_contribs fallback failed."
        ) from fallback_error


def main() -> None:
    ensure_dir(OUTPUT_DIR)

    bundle = fit_stage04_cluster_xgb(input_csv=INPUT_CSV)
    X_imp = bundle.prepared.X_imp

    print(f"[info] Input CSV           : {INPUT_CSV}")
    print(f"[info] Output dir          : {OUTPUT_DIR}")
    print(f"[info] Training rows used  : {len(X_imp)}")
    print(f"[info] Final feature count : {len(bundle.prepared.feature_cols)}")

    shap_values, shap_backend = compute_shap_values_with_fallback(bundle.model, X_imp)
    feature_names = list(X_imp.columns)
    feature_short_names = build_unique_short_names(feature_names)

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "feature_short": feature_short_names,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "mean_shap": shap_values.mean(axis=0),
            "std_abs_shap": np.abs(shap_values).std(axis=0),
        }
    ).sort_values(by="mean_abs_shap", ascending=False)

    importance_csv = OUTPUT_DIR / "shap_feature_importance.csv"
    importance_top10_csv = OUTPUT_DIR / "shap_feature_importance_top10.csv"
    feature_abbrev_map_csv = OUTPUT_DIR / "feature_abbrev_mapping.csv"
    beeswarm_png = OUTPUT_DIR / "shap_summary_beeswarm.png"
    bar_png = OUTPUT_DIR / "shap_summary_bar.png"
    meta_json = OUTPUT_DIR / "shap_run_info.json"

    pd.DataFrame(
        {
            "feature": feature_names,
            "feature_short": feature_short_names,
        }
    ).to_csv(feature_abbrev_map_csv, index=False, encoding="utf-8-sig")

    importance_df.to_csv(importance_csv, index=False, encoding="utf-8-sig")
    importance_df.head(TOP_N_DISPLAY).to_csv(importance_top10_csv, index=False, encoding="utf-8-sig")
    save_summary_plot(shap_values, X_imp, feature_short_names, beeswarm_png)
    save_bar_plot(shap_values, X_imp, feature_short_names, bar_png)

    save_json(
        {
            "input_csv": INPUT_CSV,
            "output_dir": OUTPUT_DIR,
            "label_col": bundle.prepared.label_col,
            "image_col": bundle.prepared.image_col,
            "nucleus_col": bundle.prepared.nucleus_col,
            "feature_count": len(bundle.prepared.feature_cols),
            "feature_columns": bundle.prepared.feature_cols,
            "removed_feature_columns": bundle.prepared.removed_feature_cols,
            "normalized_absolute_source_columns": bundle.prepared.normalized_absolute_source_columns,
            "label_counts": bundle.prepared.label_counts,
            "xgb_validation_metrics": bundle.metrics,
            "shap": {
                "predicted_probability_label": stage05.OUTPUT_CLUSTER2_LABEL,
                "max_display": MAX_DISPLAY,
                "top_n_display": TOP_N_DISPLAY,
                "n_rows_explained": int(len(X_imp)),
                "backend": shap_backend,
                "display_name_mode": "abbreviated",
            },
            "outputs": {
                "importance_csv": importance_csv,
                "importance_top10_csv": importance_top10_csv,
                "feature_abbrev_mapping_csv": feature_abbrev_map_csv,
                "beeswarm_png": beeswarm_png,
                "bar_png": bar_png,
            },
        },
        meta_json,
    )

    print("[ok] SHAP analysis finished.")
    print(f"[ok] SHAP backend   : {shap_backend}")
    print(f"[ok] Importance CSV : {importance_csv}")
    print(f"[ok] Top10 CSV      : {importance_top10_csv}")
    print(f"[ok] Abbrev map CSV : {feature_abbrev_map_csv}")
    print(f"[ok] Beeswarm PNG   : {beeswarm_png}")
    print(f"[ok] Bar PNG        : {bar_png}")
    print(f"[info] Top {TOP_N_DISPLAY} features by mean |SHAP|:")
    print(
        importance_df.head(TOP_N_DISPLAY)[
            ["feature", "feature_short", "mean_abs_shap", "mean_shap", "std_abs_shap"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

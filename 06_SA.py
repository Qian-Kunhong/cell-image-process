from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SALib.analyze import sobol
from SALib.sample import saltelli

from cluster_xgb_analysis_common import (
    DEFAULT_INPUT_CSV,
    fit_stage04_cluster_xgb,
    save_json,
    stage05,
)


# ============================================================
# 06_SA.py
# ------------------------------------------------------------
# Sobol sensitivity analysis for stage-04 clustered features.
# Also exports feature-by-feature meaning + calculation notes.
# ============================================================

INPUT_CSV = DEFAULT_INPUT_CSV
OUTPUT_DIR = INPUT_CSV.parent / "06_SA"
SOBOL_BASE_EXPONENT = 10
CALC_SECOND_ORDER = True
TOP_N_DISPLAY = 10
SHOW_PLOT = False

TITLE_FONT_SIZE = 20
AXIS_LABEL_FONT_SIZE = 16
TICK_FONT_SIZE = 14
LEGEND_FONT_SIZE = 14


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_problem_from_training_matrix(X_imp: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    bounds_rows: list[dict[str, float | str]] = []
    bounds: list[list[float]] = []

    for col in X_imp.columns:
        values = pd.to_numeric(X_imp[col], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            lo, hi = -1.0, 1.0
        else:
            lo = float(np.min(values))
            hi = float(np.max(values))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                center = lo if np.isfinite(lo) else 0.0
                delta = 1.0 if abs(center) < 1e-9 else abs(center) * 0.05
                lo = center - delta
                hi = center + delta

        bounds.append([lo, hi])
        bounds_rows.append({"feature": col, "lower_bound": lo, "upper_bound": hi})

    problem = {
        "num_vars": len(X_imp.columns),
        "names": list(X_imp.columns),
        "bounds": bounds,
    }
    return problem, pd.DataFrame(bounds_rows)


def save_sa_csv(result_df: pd.DataFrame, output_csv: Path) -> None:
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "ST", "ST_conf", "S1", "S1_conf"])
        for row in result_df.itertuples(index=False):
            writer.writerow([row.feature, row.ST, row.ST_conf, row.S1, row.S1_conf])


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


def explain_feature_meaning(feature: str) -> str:
    f = str(feature).lower()
    meanings: list[str] = []

    if "nn1_distance" in f:
        meanings.append("distance to nearest neighbor nucleus")
    if "knn6_distance_mean" in f:
        meanings.append("mean distance to 6 nearest neighbors")
    if "knn6_distance_std" in f:
        meanings.append("std of distances to 6 nearest neighbors")
    if "adaptive_radius" in f:
        meanings.append("adaptive neighborhood radius (k-th nearest distance)")
    if "local_density" in f:
        meanings.append("local cell density in neighborhood")
    if "neighbor_count" in f:
        meanings.append("number of neighboring nuclei")
    if "nb_area" in f:
        meanings.append("neighbor nuclei area statistics")
    if "nb_circularity" in f:
        meanings.append("neighbor nuclei circularity statistics")
    if "nb_eccentricity" in f:
        meanings.append("neighbor nuclei eccentricity statistics")
    if "nb_aspect_ratio" in f:
        meanings.append("neighbor nuclei aspect-ratio statistics")
    if "nb_distance" in f:
        meanings.append("distance statistics from current nucleus to neighbors")
    if "area" in f and "nb_area" not in f:
        meanings.append("nucleus area-related shape feature")
    if "perimeter" in f:
        meanings.append("nucleus perimeter")
    if "equivalent_diameter" in f or "diameter" in f:
        meanings.append("nucleus size / equivalent diameter")
    if "major_axis" in f or "minor_axis" in f:
        meanings.append("ellipse axis length from nucleus shape")
    if "aspect_ratio" in f and "nb_" not in f:
        meanings.append("elongation ratio of nucleus")
    if "eccentricity" in f and "nb_" not in f:
        meanings.append("shape elongation (eccentricity)")
    if "circularity" in f and "nb_" not in f:
        meanings.append("shape roundness")
    if "solidity" in f:
        meanings.append("shape solidity")
    if "extent" in f:
        meanings.append("bbox occupancy ratio")
    if "boundary_distance" in f:
        meanings.append("distance to colony boundary")
    if "edge_band" in f:
        meanings.append("edge-band threshold width")
    if "inner_fit_cutoff" in f:
        meanings.append("inner-fit region threshold")
    if "colony_support_dilate" in f:
        meanings.append("colony support dilation parameter")
    if "pixel_size" in f:
        meanings.append("image pixel physical-size metadata")
    if "fit_used_for_model" in f:
        meanings.append("whether this nucleus is used for fitting")
    if f.startswith("is_"):
        meanings.append("binary indicator flag")
    if "__img_rel" in f:
        meanings.append("image-relative normalized feature")
    if "__model" in f:
        meanings.append("model-transformed feature")
    if "_um" in f or "_um2" in f or "_per_um2" in f:
        meanings.append("feature in physical units")

    if not meanings:
        meanings.append("morphology or neighborhood-derived feature")

    # Deduplicate while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for m in meanings:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return "; ".join(out)


def explain_feature_calculation(feature: str) -> str:
    f = str(feature).lower()
    rules: list[str] = []

    # Base feature formula hints.
    if "nn1_distance" in f:
        rules.append("nn1_distance = min distance from nucleus i to all other nuclei")
    if "knn6_distance_mean" in f:
        rules.append("knn6_distance_mean = mean of distances to 6 nearest neighbors")
    if "knn6_distance_std" in f:
        rules.append("knn6_distance_std = std of distances to 6 nearest neighbors")
    if "adaptive_radius" in f:
        rules.append("adaptive_radius = distance to k-th nearest neighbor (k=6)")
    if "local_density_per_um2" in f:
        rules.append("local_density_per_um2 = neighbor_count / (pi * adaptive_radius_um^2)")
    elif "local_density" in f:
        rules.append("local_density = neighbor_count / (pi * adaptive_radius^2)")
    if "nb_area_mean" in f:
        rules.append("nb_area_mean = mean(area of neighbors)")
    if "nb_area_std" in f:
        rules.append("nb_area_std = std(area of neighbors)")
    if "nb_circularity_mean" in f:
        rules.append("nb_circularity_mean = mean(circularity of neighbors)")
    if "nb_circularity_std" in f:
        rules.append("nb_circularity_std = std(circularity of neighbors)")
    if "nb_eccentricity_mean" in f:
        rules.append("nb_eccentricity_mean = mean(eccentricity of neighbors)")
    if "nb_eccentricity_std" in f:
        rules.append("nb_eccentricity_std = std(eccentricity of neighbors)")
    if "nb_aspect_ratio_mean" in f:
        rules.append("nb_aspect_ratio_mean = mean(aspect_ratio of neighbors)")
    if "nb_aspect_ratio_std" in f:
        rules.append("nb_aspect_ratio_std = std(aspect_ratio of neighbors)")
    if "nb_distance_mean" in f:
        rules.append("nb_distance_mean = mean(distance from nucleus i to its neighbors)")
    if "nb_distance_std" in f:
        rules.append("nb_distance_std = std(distance from nucleus i to its neighbors)")
    if "adaptive_neighbor_count" in f:
        rules.append("adaptive_neighbor_count = number of nuclei inside adaptive radius")
    if "fixed_neighbor_count" in f:
        rules.append("fixed_neighbor_count = number of nuclei inside fixed radius")

    # Geometry conversions used in stage 03.
    if f.endswith("_um2"):
        rules.append("physical area = pixel_area * (pixel_size_row_um * pixel_size_col_um)")
    elif "_um" in f and "_per_um2" not in f:
        rules.append("physical length = pixel_length * sqrt(pixel_size_row_um * pixel_size_col_um)")

    # Stage-04 transforms.
    if "__model" in f:
        rules.append(
            "model transform (stage 04): if feature is in log1p list -> x_model = log(1 + x_raw); else x_model = x_raw"
        )
    if "__img_rel" in f:
        rules.append(
            "image-relative normalization (stage 05): x_img_rel = (x - median(x in same image)) / IQR(x in same image)"
        )

    if f.startswith("is_") and not rules:
        rules.append("binary indicator computed by threshold/condition in pipeline (0/1)")
    if "fit_used_for_model" in f:
        rules.append("fit_used_for_model = 1 if nucleus is inside inner-fit region, else 0")
    if "boundary_distance" in f:
        rules.append("distance from nucleus location to colony boundary (distance transform)")
    if "edge_band" in f:
        rules.append("edge_band threshold derived from colony geometry and equivalent diameter")
    if "inner_fit_cutoff" in f:
        rules.append("inner_fit_cutoff derived from colony geometry and equivalent diameter")

    if not rules:
        rules.append("computed from morphology / neighborhood statistics in stage 03/04 pipeline")

    # Deduplicate while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for r in rules:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return " | ".join(out)


def explain_negative_bound_reason(feature: str, has_negative_bound: bool) -> str:
    if not has_negative_bound:
        return "no negative bound"
    f = str(feature).lower()
    if "__img_rel" in f:
        return "negative values are expected after image-relative normalization around image median"
    if f.startswith("is_"):
        return "indicator encoding can include negative values in some preprocessing branches"
    return "feature range after preprocessing includes negative values"


def build_feature_explanation_table(problem: dict, bounds_df: pd.DataFrame) -> pd.DataFrame:
    bound_map = bounds_df.set_index("feature")[["lower_bound", "upper_bound"]]
    rows = []
    for feat in problem["names"]:
        lo = float(bound_map.loc[feat, "lower_bound"])
        hi = float(bound_map.loc[feat, "upper_bound"])
        neg = bool((lo < 0.0) or (hi < 0.0))
        rows.append(
            {
                "feature": feat,
                "feature_short": abbreviate_feature_name(feat),
                "meaning_cn": explain_feature_meaning(feat),
                "calc_cn": explain_feature_calculation(feat),
                "lower_bound": lo,
                "upper_bound": hi,
                "has_negative_bound": neg,
                "negative_reason": explain_negative_bound_reason(feat, neg),
            }
        )
    return pd.DataFrame(rows)


def write_parameter_full_explanation(explain_df: pd.DataFrame, output_txt: Path) -> None:
    lines = [
        "Feature-by-Feature Meaning And Calculation",
        "==========================================",
        "",
        "Each entry includes: meaning, how to calculate, Sobol bounds, and negative-bound reason.",
        "",
    ]

    for i, row in enumerate(explain_df.itertuples(index=False), start=1):
        lines.append(f"{i}. {row.feature}")
        lines.append(f"   - short name: {row.feature_short}")
        lines.append(f"   - meaning: {row.meaning_cn}")
        lines.append(f"   - calculation: {row.calc_cn}")
        lines.append(f"   - bounds: [{row.lower_bound:.6g}, {row.upper_bound:.6g}]")
        lines.append(f"   - negative-bound note: {row.negative_reason}")
        lines.append("")

    output_txt.write_text("\n".join(lines), encoding="utf-8")


def write_parameter_meaning_note(output_txt: Path) -> None:
    lines = [
        "How To Read feature_parameter_meaning.csv",
        "=========================================",
        "",
        "Columns",
        "-------",
        "feature: original feature name used by model.",
        "feature_short: abbreviated name used in plot x-axis.",
        "meaning_cn: plain-language meaning of this feature.",
        "calc_cn: how this feature is computed (formula or pipeline step).",
        "lower_bound / upper_bound: Sobol sampling bounds from processed training matrix.",
        "has_negative_bound: whether this feature has negative sampling boundary.",
        "negative_reason: why negative bound appears for this feature.",
        "",
        "Important",
        "---------",
        "Bounds are based on processed features (after model transforms and normalization),",
        "not raw microscope pixel values.",
    ]
    output_txt.write_text("\n".join(lines), encoding="utf-8")


def plot_sobol_indices(result_df: pd.DataFrame, output_png: Path) -> None:
    plot_df = result_df.head(TOP_N_DISPLAY).copy()
    plot_df["feature_short"] = plot_df["feature"].map(abbreviate_feature_name)

    n_features = len(plot_df)
    fig_width = max(16, min(30, n_features * 0.8))
    plt.figure(figsize=(fig_width, 10))
    x = np.arange(n_features)
    plt.bar(x - 0.2, plot_df["ST"], width=0.4, color="royalblue", alpha=0.80, label="ST")
    plt.bar(x + 0.2, plot_df["S1"], width=0.4, color="crimson", alpha=0.75, label="S1")
    plt.xticks(x, plot_df["feature_short"], rotation=35, ha="right", fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    plt.ylabel("Sobol Index", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.xlabel("Feature (Abbrev.)", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.title(
        f"Stage-04 Cluster XGBoost Sobol Sensitivity Analysis (Top {TOP_N_DISPLAY})",
        fontsize=TITLE_FONT_SIZE,
    )
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(output_png, dpi=220)
    if SHOW_PLOT:
        plt.show()
    plt.close()


def main() -> None:
    ensure_dir(OUTPUT_DIR)

    bundle = fit_stage04_cluster_xgb(input_csv=INPUT_CSV)
    X_imp = bundle.prepared.X_imp
    problem, bounds_df = build_problem_from_training_matrix(X_imp)

    base_sample_size = 2 ** SOBOL_BASE_EXPONENT
    total_samples = base_sample_size * (2 * problem["num_vars"] + 2)

    print(f"[info] Input CSV               : {INPUT_CSV}")
    print(f"[info] Output dir              : {OUTPUT_DIR}")
    print(f"[info] Training rows used      : {len(X_imp)}")
    print(f"[info] Final feature count     : {problem['num_vars']}")
    print(f"[info] Sobol base sample size  : {base_sample_size}")
    print(f"[info] Sobol total evaluations : {total_samples}")

    samples = saltelli.sample(problem, base_sample_size, calc_second_order=CALC_SECOND_ORDER)
    samples_df = pd.DataFrame(samples, columns=problem["names"])
    y_pred = bundle.model.predict_proba(samples_df)[:, 1]

    sa = sobol.analyze(
        problem,
        y_pred,
        calc_second_order=CALC_SECOND_ORDER,
        print_to_console=False,
    )

    result_df = pd.DataFrame(
        {
            "feature": problem["names"],
            "S1": sa["S1"],
            "S1_conf": sa["S1_conf"],
            "ST": sa["ST"],
            "ST_conf": sa["ST_conf"],
        }
    ).sort_values(by="ST", ascending=False, na_position="last")

    explain_df = build_feature_explanation_table(problem, bounds_df)

    samples_csv = OUTPUT_DIR / f"sobol_samples_{total_samples}.csv"
    pred_csv = OUTPUT_DIR / f"sobol_predictions_{total_samples}.csv"
    bounds_csv = OUTPUT_DIR / "feature_bounds.csv"
    result_csv = OUTPUT_DIR / "sobol_indices.csv"
    result_top10_abbrev_csv = OUTPUT_DIR / "sobol_indices_top10_abbrev.csv"
    feature_abbrev_map_csv = OUTPUT_DIR / "feature_abbrev_mapping.csv"
    parameter_meaning_csv = OUTPUT_DIR / "feature_parameter_meaning.csv"
    parameter_meaning_note_txt = OUTPUT_DIR / "feature_parameter_meaning_note.txt"
    parameter_meaning_full_txt = OUTPUT_DIR / "feature_parameter_meaning_full.txt"
    figure_png = OUTPUT_DIR / "sobol_indices.png"
    meta_json = OUTPUT_DIR / "sobol_run_info.json"

    samples_df.to_csv(samples_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame({"cluster2_probability": y_pred}).to_csv(pred_csv, index=False, encoding="utf-8-sig")
    bounds_df.to_csv(bounds_csv, index=False, encoding="utf-8-sig")
    save_sa_csv(result_df, result_csv)

    pd.DataFrame(
        {
            "feature": problem["names"],
            "feature_short": [abbreviate_feature_name(name) for name in problem["names"]],
        }
    ).drop_duplicates(subset=["feature_short", "feature"]).to_csv(
        feature_abbrev_map_csv, index=False, encoding="utf-8-sig"
    )

    explain_df.to_csv(parameter_meaning_csv, index=False, encoding="utf-8-sig")

    result_df.head(TOP_N_DISPLAY).assign(
        feature_short=lambda df: df["feature"].map(abbreviate_feature_name)
    )[
        ["feature", "feature_short", "ST", "ST_conf", "S1", "S1_conf"]
    ].to_csv(result_top10_abbrev_csv, index=False, encoding="utf-8-sig")

    write_parameter_meaning_note(parameter_meaning_note_txt)
    write_parameter_full_explanation(explain_df, parameter_meaning_full_txt)
    plot_sobol_indices(result_df, figure_png)

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
            "sobol": {
                "base_exponent": SOBOL_BASE_EXPONENT,
                "base_sample_size": base_sample_size,
                "total_model_evaluations": total_samples,
                "calc_second_order": CALC_SECOND_ORDER,
                "top_n_display": TOP_N_DISPLAY,
                "predicted_probability_label": stage05.OUTPUT_CLUSTER2_LABEL,
            },
            "outputs": {
                "samples_csv": samples_csv,
                "predictions_csv": pred_csv,
                "feature_bounds_csv": bounds_csv,
                "sobol_indices_csv": result_csv,
                "sobol_indices_top10_abbrev_csv": result_top10_abbrev_csv,
                "feature_abbrev_mapping_csv": feature_abbrev_map_csv,
                "feature_parameter_meaning_csv": parameter_meaning_csv,
                "feature_parameter_meaning_note_txt": parameter_meaning_note_txt,
                "feature_parameter_meaning_full_txt": parameter_meaning_full_txt,
                "figure_png": figure_png,
            },
        },
        meta_json,
    )

    print("[ok] Sobol sensitivity analysis finished.")
    print(f"[ok] Bounds CSV   : {bounds_csv}")
    print(f"[ok] Result CSV   : {result_csv}")
    print(f"[ok] Top10 map CSV: {result_top10_abbrev_csv}")
    print(f"[ok] Full map CSV : {feature_abbrev_map_csv}")
    print(f"[ok] Meaning CSV  : {parameter_meaning_csv}")
    print(f"[ok] Meaning TXT  : {parameter_meaning_note_txt}")
    print(f"[ok] Full TXT     : {parameter_meaning_full_txt}")
    print(f"[ok] Figure PNG   : {figure_png}")
    print(f"[info] Top {TOP_N_DISPLAY} features by ST:")
    print(result_df.head(TOP_N_DISPLAY).to_string(index=False))


if __name__ == "__main__":
    main()

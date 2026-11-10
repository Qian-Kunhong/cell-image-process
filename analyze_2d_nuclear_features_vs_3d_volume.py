from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from skimage import measure


WORKSPACE = Path(__file__).resolve().parent
DEPENDENCY_DIR = WORKSPACE / ".codex_deps" / "oiffile"
if str(DEPENDENCY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPENDENCY_DIR))

import oiffile  # noqa: E402


def _data_root() -> Path:
    """数据根从 naruchit_pipeline/config/project.json 读；可用 NARUCHIT_DATA_ROOT 覆盖。"""
    override = os.environ.get("NARUCHIT_DATA_ROOT")
    if override:
        return Path(override)
    cfg = WORKSPACE / "naruchit_pipeline" / "config" / "project.json"
    if not cfg.exists():
        raise RuntimeError(f"找不到配置 {cfg}，也没有设置 NARUCHIT_DATA_ROOT")
    return Path(json.loads(cfg.read_text(encoding="utf-8-sig"))["data_root"])


RAW_DIR = _data_root() / "03_Raw data" / "Immunostaining" / "Confocal - YAP"
ANALYSIS_ROOT = (
    WORKSPACE / "analysis_outputs" / "hepatocyte_confocal_audit" / "dapi_3d" / "yap"
)
INPUT_METRICS = ANALYSIS_ROOT / "analysis_summary" / "nucleus_3d_yap_cell_metrics.csv"
OUTPUT_DIR = ANALYSIS_ROOT / "area_volume_correlations"

XY_UM = 0.17578
SAMPLE_ORDER = [
    "D20D", "D30D", "D31D", "D33D", "D40D",
    "D41D", "D43D", "D50D", "D51D", "D53D",
]

FEATURE_LABELS = {
    "max_xy_area_um2": "Max-section area",
    "projected_area_um2": "Projected area",
    "perimeter_crofton_um": "Perimeter",
    "equivalent_diameter_um": "Equivalent diameter",
    "major_axis_um": "Major axis",
    "minor_axis_um": "Minor axis",
    "aspect_ratio": "Aspect ratio",
    "circularity": "Circularity",
    "eccentricity": "Eccentricity",
    "solidity": "Solidity",
    "extent": "Extent",
    "dapi_mean_norm": "DAPI mean (normalized)",
    "dapi_std_norm": "DAPI SD (normalized)",
    "dapi_cv": "DAPI CV",
    "dapi_iqr_norm": "DAPI IQR (normalized)",
    "dapi_entropy_bits": "DAPI intensity entropy",
}

SHAPE_FEATURES = [
    "max_xy_area_um2",
    "projected_area_um2",
    "perimeter_crofton_um",
    "equivalent_diameter_um",
    "major_axis_um",
    "minor_axis_um",
    "aspect_ratio",
    "circularity",
    "eccentricity",
    "solidity",
    "extent",
]
INTENSITY_FEATURES = [
    "dapi_mean_norm",
    "dapi_std_norm",
    "dapi_cv",
    "dapi_iqr_norm",
    "dapi_entropy_bits",
]
FEATURES = SHAPE_FEATURES + INTENSITY_FEATURES

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25


def ensure_zcyx(array: np.ndarray, axes: str) -> np.ndarray:
    if axes != "ZCYX":
        raise ValueError(f"Expected ZCYX metadata, got axes={axes}, shape={array.shape}")
    if array.ndim == 4 and array.shape[0] <= 4 and array.shape[1] > 4:
        return np.transpose(array, (1, 0, 2, 3))
    return array


def robust_normalize_image(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = image[np.isfinite(image) & (image > 0)]
    if values.size == 0:
        return np.zeros_like(image, dtype=np.float32), 0.0, 1.0
    low, high = np.percentile(values, [1.0, 99.5])
    if not np.isfinite(high) or high <= low:
        high = low + 1.0
    normalized = np.clip((image.astype(np.float32) - low) / (high - low), 0.0, 1.0)
    return normalized, float(low), float(high)


def intensity_entropy(values: np.ndarray, bins: int = 16) -> float:
    if values.size < 2:
        return np.nan
    counts, _ = np.histogram(np.clip(values, 0.0, 1.0), bins=bins, range=(0.0, 1.0))
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def extract_sample_features(sample: str, accepted: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    mask_path = ANALYSIS_ROOT / sample / "dapi_3d_masks.npz"
    labels = np.load(mask_path)["raw_labels"].astype(np.int32)
    with oiffile.OifFile(RAW_DIR / f"{sample}.oib") as handle:
        stack = ensure_zcyx(handle.asarray(), handle.axes)
    dapi_raw = stack[:, 0].astype(np.float32)
    dapi_norm, norm_low, norm_high = robust_normalize_image(dapi_raw)

    rows: list[dict[str, float | int | str]] = []
    for record in accepted.itertuples(index=False):
        label = int(record.label)
        coords = np.argwhere(labels == label)
        if coords.size == 0:
            continue
        z_counts = np.bincount(coords[:, 0], minlength=labels.shape[0])
        best_z = int(np.argmax(z_counts))
        mask_2d = labels[best_z] == label
        projection = np.any(labels == label, axis=0)
        props = measure.regionprops(mask_2d.astype(np.uint8))[0]
        area_um2 = float(props.area * XY_UM**2)
        perimeter_um = float(props.perimeter_crofton * XY_UM)
        circularity = (
            float(4.0 * np.pi * area_um2 / perimeter_um**2)
            if perimeter_um > 0 else np.nan
        )
        major_um = float(props.axis_major_length * XY_UM)
        minor_um = float(props.axis_minor_length * XY_UM)
        intensities_norm = dapi_norm[best_z][mask_2d]
        intensities_raw = dapi_raw[best_z][mask_2d]
        q25, q75 = np.percentile(intensities_norm, [25, 75])
        raw_mean = float(np.mean(intensities_raw))
        raw_std = float(np.std(intensities_raw, ddof=1)) if intensities_raw.size > 1 else 0.0
        rows.append(
            {
                "sample": sample,
                "label": label,
                "max_slice_z_index": best_z,
                "max_xy_area_um2": area_um2,
                "projected_area_um2": float(projection.sum() * XY_UM**2),
                "perimeter_crofton_um": perimeter_um,
                "equivalent_diameter_um": float(props.equivalent_diameter_area * XY_UM),
                "major_axis_um": major_um,
                "minor_axis_um": minor_um,
                "aspect_ratio": major_um / minor_um if minor_um > 0 else np.nan,
                "circularity": circularity,
                "eccentricity": float(props.eccentricity),
                "solidity": float(props.solidity),
                "extent": float(props.extent),
                "dapi_mean_norm": float(np.mean(intensities_norm)),
                "dapi_std_norm": float(np.std(intensities_norm, ddof=1)) if intensities_norm.size > 1 else 0.0,
                "dapi_cv": raw_std / raw_mean if raw_mean > 0 else np.nan,
                "dapi_iqr_norm": float(q75 - q25),
                "dapi_entropy_bits": intensity_entropy(intensities_norm),
            }
        )
    metadata = {
        "sample": sample,
        "dapi_normalization_p01": norm_low,
        "dapi_normalization_p995": norm_high,
        "accepted_nuclei_requested": int(len(accepted)),
        "accepted_nuclei_extracted": int(len(rows)),
    }
    return pd.DataFrame(rows), metadata


def safe_correlations(frame: pd.DataFrame, x: str, y: str = "volume_um3") -> dict:
    """核级相关系数（只报效应量与 n，**不报 p 值**）。

    统计单位是视野，每个条件只有 1 个视野：把核当独立样本算 p 值会伪造样本量
    （同一视野内的核彼此不独立），因此本流水线不提供任何核级显著性检验。
    rho / r 只作"同一张图内两个测量量的描述性关联"，不能当组间推断。
    """
    pair = frame[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 4 or pair[x].nunique() < 2 or pair[y].nunique() < 2:
        return {"n": int(len(pair)), "spearman_rho": np.nan, "pearson_r": np.nan}
    sr = spearmanr(pair[x], pair[y])
    pr = pearsonr(pair[x], pair[y])
    return {
        "n": int(len(pair)),
        "spearman_rho": float(sr.statistic),
        "pearson_r": float(pr.statistic),
    }


def condition_stratified_spearman(frame: pd.DataFrame, feature: str) -> tuple[float, int]:
    pair = frame[["sample", feature, "volume_um3"]].replace([np.inf, -np.inf], np.nan).dropna()
    pair = pair.groupby("sample", observed=True).filter(lambda x: len(x) >= 4)
    if len(pair) < 4:
        return np.nan, int(len(pair))
    pair["feature_rank_within_sample"] = pair.groupby("sample", observed=True)[feature].rank(pct=True)
    pair["volume_rank_within_sample"] = pair.groupby("sample", observed=True)["volume_um3"].rank(pct=True)
    result = pearsonr(pair["feature_rank_within_sample"], pair["volume_rank_within_sample"])
    return float(result.statistic), int(len(pair))


def correlation_tables(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    condition_rows = []
    for feature in FEATURES:
        pooled = safe_correlations(cells, feature)
        stratified_rho, stratified_n = condition_stratified_spearman(cells, feature)
        overall_rows.append(
            {
                "feature": feature,
                "feature_label": FEATURE_LABELS[feature],
                "feature_family": "shape" if feature in SHAPE_FEATURES else "intensity",
                **pooled,
                "condition_stratified_spearman_rho": stratified_rho,
                "condition_stratified_n": stratified_n,
            }
        )
        for sample in SAMPLE_ORDER:
            subset = cells[cells["sample"] == sample]
            condition_rows.append(
                {
                    "sample": sample,
                    "culture_day": int(subset["culture_day"].iloc[0]),
                    "paper_day": int(subset["paper_day"].iloc[0]),
                    "regimen": subset["regimen"].iloc[0],
                    "feature": feature,
                    "feature_label": FEATURE_LABELS[feature],
                    "feature_family": "shape" if feature in SHAPE_FEATURES else "intensity",
                    **safe_correlations(subset, feature),
                }
            )
    overall = pd.DataFrame(overall_rows).sort_values(
        "condition_stratified_spearman_rho", key=lambda s: s.abs(), ascending=False
    )
    by_condition = pd.DataFrame(condition_rows)
    return overall, by_condition


def log_area_volume_models(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, subset in [("all", cells)] + [(sample, cells[cells["sample"] == sample]) for sample in SAMPLE_ORDER]:
        pair = subset[["max_xy_area_um2", "volume_um3"]].dropna()
        pair = pair[(pair > 0).all(axis=1)]
        if len(pair) < 4:
            continue
        x = np.log(pair["max_xy_area_um2"].to_numpy())
        y = np.log(pair["volume_um3"].to_numpy())
        slope, intercept = np.polyfit(x, y, 1)
        prediction = intercept + slope * x
        ss_res = float(np.sum((y - prediction) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        rows.append(
            {
                "group": label,
                "n": int(len(pair)),
                "log_log_slope": float(slope),
                "log_log_intercept": float(intercept),
                "r2_log_scale": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_pooled_area(cells: pd.DataFrame, overall: pd.DataFrame) -> None:
    pooled = overall.set_index("feature").loc["max_xy_area_um2"]
    fig, ax = plt.subplots(figsize=(9.5, 7.2), constrained_layout=True)
    palette = dict(zip(SAMPLE_ORDER, plt.get_cmap("tab10").colors))
    for sample in SAMPLE_ORDER:
        subset = cells[cells["sample"] == sample]
        ax.scatter(
            subset["max_xy_area_um2"], subset["volume_um3"],
            s=26, alpha=0.65, label=sample, color=palette[sample], edgecolors="none",
        )
    ax.set_xlabel("Maximum 2D nuclear cross-sectional area (µm²)")
    ax.set_ylabel("3D nuclear volume (µm³)")
    ax.set_title(
        "2D nuclear area versus 3D nuclear volume\n"
        f"pooled Spearman ρ={pooled['spearman_rho']:.3f}; "
        f"condition-stratified ρ={pooled['condition_stratified_spearman_rho']:.3f}"
    )
    ax.legend(title="Sample", ncol=2, fontsize=8, frameon=True)
    fig.savefig(OUTPUT_DIR / "pooled_area_volume_scatter.png", dpi=220)
    plt.close(fig)


def plot_condition_facets(cells: pd.DataFrame, by_condition: pd.DataFrame) -> None:
    area_stats = by_condition[by_condition["feature"] == "max_xy_area_um2"].set_index("sample")
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True, sharex=False, sharey=False)
    for ax, sample in zip(axes.flat, SAMPLE_ORDER):
        subset = cells[cells["sample"] == sample]
        ax.scatter(subset["max_xy_area_um2"], subset["volume_um3"], s=22, alpha=0.65, color="#2563eb", edgecolors="none")
        stats = area_stats.loc[sample]
        ax.set_title(f"{sample}  n={int(stats['n'])}\nSpearman ρ={stats['spearman_rho']:.3f}")
        ax.set_xlabel("2D area (µm²)")
        ax.set_ylabel("3D volume (µm³)")
    fig.suptitle("Area–volume association within each sample/condition", fontsize=16)
    fig.savefig(OUTPUT_DIR / "condition_area_volume_facets.png", dpi=220)
    plt.close(fig)


def plot_heatmap(by_condition: pd.DataFrame) -> None:
    matrix = by_condition.pivot(index="feature_label", columns="sample", values="spearman_rho")
    matrix = matrix.reindex([FEATURE_LABELS[f] for f in FEATURES])
    fig, ax = plt.subplots(figsize=(13, 9), constrained_layout=True)
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                ax.text(
                    col, row, f"{value:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(value) > 0.55 else "black",
                )
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=0)
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
    colorbar.set_label("Within-condition Spearman ρ")
    ax.grid(False)
    ax.set_xlabel("Sample / condition")
    ax.set_ylabel("2D feature")
    ax.set_title("Correlation of 2D nuclear features with 3D nuclear volume")
    fig.savefig(OUTPUT_DIR / "feature_volume_correlation_heatmap.png", dpi=220)
    plt.close(fig)


def plot_overall_ranking(overall: pd.DataFrame) -> None:
    plot = overall.sort_values("condition_stratified_spearman_rho")
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    y = np.arange(len(plot))
    ax.barh(y - 0.18, plot["spearman_rho"], height=0.34, label="Pooled", color="#94a3b8")
    ax.barh(y + 0.18, plot["condition_stratified_spearman_rho"], height=0.34, label="Condition-stratified", color="#2563eb")
    ax.set_yticks(y, plot["feature_label"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman association with 3D nuclear volume")
    ax.set_title("Pooled correlation versus within-condition association")
    ax.legend(frameon=False)
    fig.savefig(OUTPUT_DIR / "overall_feature_volume_correlations.png", dpi=220)
    plt.close(fig)


def write_report(
    cells: pd.DataFrame,
    overall: pd.DataFrame,
    by_condition: pd.DataFrame,
    models: pd.DataFrame,
) -> None:
    area_overall = overall.set_index("feature").loc["max_xy_area_um2"]
    area_conditions = by_condition[by_condition["feature"] == "max_xy_area_um2"].copy()
    area_conditions = area_conditions.sort_values("sample", key=lambda s: s.map({v: i for i, v in enumerate(SAMPLE_ORDER)}))
    strongest = overall.reindex(overall["condition_stratified_spearman_rho"].abs().sort_values(ascending=False).index).head(8)
    all_model = models.set_index("group").loc["all"]

    condition_lines = [
        f"| {row.sample} | {int(row.n)} | {row.spearman_rho:.3f} | {row.pearson_r:.3f} |"
        for row in area_conditions.itertuples(index=False)
    ]
    feature_lines = [
        f"| {row.feature_label} | {row.spearman_rho:.3f} | {row.condition_stratified_spearman_rho:.3f} |"
        for row in strongest.itertuples(index=False)
    ]
    text = f"""# 2D nuclear features versus 3D nuclear volume

## Scope

This analysis uses the YAP-panel 3D DAPI segmentation. The measured object is the **nucleus**, not the whole cell. Each accepted 3D nucleus was represented in 2D by its largest XY cross-section; projection area and additional shape/intensity features were also extracted.

There are {len(cells):,} QC-passed nuclei from {cells['sample'].nunique()} OIB stacks. Each sample/condition has only one OIB field, so nuclei are technical subsamples rather than independent biological replicates.

## Main area–volume result

- Pooled Spearman correlation: **ρ = {area_overall['spearman_rho']:.3f}** (n = {int(area_overall['n'])}).
- Condition-stratified Spearman association: **ρ = {area_overall['condition_stratified_spearman_rho']:.3f}**. This removes the ranking shift between samples and asks whether larger 2D nuclei are also larger in 3D within the same condition.
- Pooled Pearson correlation: **r = {area_overall['pearson_r']:.3f}**.
- A log–log area-only model gives **R² = {all_model['r2_log_scale']:.3f}** and exponent **{all_model['log_log_slope']:.3f}**.

## Area–volume correlation within each condition

| Sample | N nuclei | Spearman ρ | Pearson r |
|---|---:|---:|---:|
{chr(10).join(condition_lines)}

## Strongest 2D associations with 3D volume

| 2D feature | Pooled Spearman ρ | Condition-stratified ρ |
|---|---:|---:|
{chr(10).join(feature_lines)}

## Interpretation for future Hoechst analysis

1. Maximum 2D nuclear area can be used as a practical proxy for 3D nuclear volume, but it is not a replacement for 3D measurement.
2. Pooled correlations can be inflated by differences between culture days or treatments. The condition-stratified result and the per-condition table are the more relevant evidence for a general 2D proxy.
3. Shape features are more transferable to future Hoechst imaging than absolute intensity. Intensity features here were normalized within each OIB stack, and cross-image brightness should not be interpreted as biology.
4. Area and volume originate from the same 3D segmentation mask. Their association is partly geometric and should be validated prospectively against independently acquired 2D live-Hoechst images.
5. For the future model, retain area, equivalent diameter, major/minor axes, circularity, eccentricity, solidity and normalized intensity heterogeneity. Remove redundant features only after checking correlations and cross-run validation.

## Files

- `nucleus_2d_features_vs_3d_volume.csv`: one row per nucleus.
- `overall_feature_volume_correlations.csv`: pooled and condition-stratified correlations.
- `condition_feature_volume_correlations.csv`: correlations within D20D–D53D.
- `log_area_volume_models.csv`: log–log area-only fits.
- PNG figures: pooled scatter, condition facets, feature heatmap and overall ranking.
"""
    (OUTPUT_DIR / "analysis_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(INPUT_METRICS)
    accepted = existing[existing["qc_pass_uniform"]].copy()
    frames = []
    metadata = []
    for sample in SAMPLE_ORDER:
        print(f"Extracting 2D features for {sample}...", flush=True)
        sample_features, sample_metadata = extract_sample_features(
            sample, accepted[accepted["sample"] == sample]
        )
        frames.append(sample_features)
        metadata.append(sample_metadata)
    two_d = pd.concat(frames, ignore_index=True)
    merge_columns = [
        "sample", "label", "volume_um3", "surface_area_um2", "height_um", "max_xy_area_um2",
        "effective_thickness_um", "sphericity", "pca_elongation_1_over_3",
        "nearest_neighbor_3d_um", "culture_day", "paper_day", "regimen", "regimen_cn",
    ]
    cells = accepted[merge_columns].merge(two_d, on=["sample", "label"], how="inner", validate="one_to_one")
    if not np.allclose(cells["max_xy_area_um2_x"], cells["max_xy_area_um2_y"], rtol=0, atol=1e-9):
        raise RuntimeError("Recomputed maximum XY area does not match the existing 3D metrics table")
    cells = cells.drop(columns=["max_xy_area_um2_x"]).rename(columns={"max_xy_area_um2_y": "max_xy_area_um2"})

    overall, by_condition = correlation_tables(cells)
    models = log_area_volume_models(cells)

    cells.to_csv(OUTPUT_DIR / "nucleus_2d_features_vs_3d_volume.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metadata).to_csv(OUTPUT_DIR / "extraction_metadata.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(OUTPUT_DIR / "overall_feature_volume_correlations.csv", index=False, encoding="utf-8-sig")
    by_condition.to_csv(OUTPUT_DIR / "condition_feature_volume_correlations.csv", index=False, encoding="utf-8-sig")
    models.to_csv(OUTPUT_DIR / "log_area_volume_models.csv", index=False, encoding="utf-8-sig")

    plot_pooled_area(cells, overall)
    plot_condition_facets(cells, by_condition)
    plot_heatmap(by_condition)
    plot_overall_ranking(overall)
    write_report(cells, overall, by_condition, models)

    settings = {
        "target": "volume_um3",
        "2d_representation": "largest XY cross-section per 3D nucleus",
        "dapi_channel_index": 0,
        "xy_pixel_size_um": XY_UM,
        "n_nuclei": int(len(cells)),
        "n_samples": int(cells["sample"].nunique()),
        "features": FEATURES,
        "limitations": [
            "one OIB field per condition",
            "nuclei are technical subsamples",
            "2D and 3D measures derive from the same segmentation mask",
            "intensity normalized independently within each OIB stack",
        ],
    }
    (OUTPUT_DIR / "analysis_settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    area = overall.set_index("feature").loc["max_xy_area_um2"]
    print(f"Done: {len(cells)} nuclei", flush=True)
    print(
        f"Area-volume pooled Spearman={area['spearman_rho']:.3f}; "
        f"condition-stratified Spearman={area['condition_stratified_spearman_rho']:.3f}",
        flush=True,
    )
    print(overall[["feature", "spearman_rho", "condition_stratified_spearman_rho"]].head(10).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

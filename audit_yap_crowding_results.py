from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = (
    Path(__file__).resolve().parent
    / "analysis_outputs"
    / "hepatocyte_confocal_audit"
    / "yap_crowding"
)
SOURCE_ROOT = Path(
    r"F:\Naruchit data\1_Data of published manuscripts\3_Hepatocyte differentiation"
    r"\03_Raw data\Immunostaining\YAP quantification"
)
FIELD_AREA_UM2 = 140.624**2


def residualize_by_sample(frame: pd.DataFrame, column: str, log: bool = False) -> pd.Series:
    values = np.log2(frame[column]) if log else frame[column]
    return values - values.groupby(frame["sample"]).transform("median")


def report(label: str, x: pd.Series, y: pd.Series) -> dict:
    result = spearmanr(x, y)
    return {"analysis": label, "n": len(x), "spearman_rho": result.statistic, "p": result.pvalue}


def main() -> None:
    fields = pd.read_csv(ROOT / "yap_field_level_recomputed.csv")
    validation = pd.read_csv(ROOT / "segmentation_validation_against_existing.csv")
    merged = fields.merge(validation, on=["sample", "field", "field_id"], how="inner")
    merged["existing_density_per_10000_um2"] = merged["existing_count"] / FIELD_AREA_UM2 * 10000
    merged["recomputed_density_sample_residual"] = residualize_by_sample(
        merged, "nuclei_per_10000_um2"
    )
    merged["existing_density_sample_residual"] = residualize_by_sample(
        merged, "existing_density_per_10000_um2"
    )
    merged["recomputed_log_ratio_sample_residual"] = residualize_by_sample(
        merged, "ratio_median", log=True
    )
    merged["existing_log_ratio_sample_residual"] = residualize_by_sample(
        merged, "existing_ratio_median", log=True
    )

    intensity_rows = []
    for sample in merged["sample"].unique():
        nuclei_path = next((SOURCE_ROOT / sample).rglob(f"{sample}_Nuclei.csv"))
        ring_path = next((SOURCE_ROOT / sample).rglob(f"{sample}_PerinuclearRing.csv"))
        nuclei = pd.read_csv(nuclei_path)
        rings = pd.read_csv(ring_path)
        for field, group in nuclei.groupby("ImageNumber"):
            if sample == "D43D" and int(field) == 4:
                continue
            ring_group = rings.loc[rings["ImageNumber"] == field]
            intensity_rows.append(
                {
                    "sample": sample,
                    "field": int(field),
                    "field_id": f"{sample}_{int(field):02d}",
                    "nuclear_yap_median": group["Intensity_MeanIntensity_YAP"].median(),
                    "ring_yap_median": ring_group["Intensity_MeanIntensity_YAP"].median(),
                }
            )
    intensity = pd.DataFrame(intensity_rows)
    merged = merged.merge(intensity, on=["sample", "field", "field_id"], how="left")
    merged["nuclear_yap_sample_residual"] = residualize_by_sample(
        merged, "nuclear_yap_median", log=True
    )
    merged["ring_yap_sample_residual"] = residualize_by_sample(
        merged, "ring_yap_median", log=True
    )

    rows = [
        report(
            "Across fields: recomputed density vs existing YAP median",
            merged["nuclei_per_10000_um2"],
            merged["existing_ratio_median"],
        ),
        report(
            "Within condition: recomputed density vs existing YAP median",
            merged["recomputed_density_sample_residual"],
            merged["existing_log_ratio_sample_residual"],
        ),
        report(
            "Within condition: existing count-density vs existing YAP median",
            merged["existing_density_sample_residual"],
            merged["existing_log_ratio_sample_residual"],
        ),
        report(
            "Within condition: recomputed density vs recomputed YAP median",
            merged["recomputed_density_sample_residual"],
            merged["recomputed_log_ratio_sample_residual"],
        ),
        report(
            "Within condition: density vs nuclear YAP intensity",
            merged["existing_density_sample_residual"],
            merged["nuclear_yap_sample_residual"],
        ),
        report(
            "Within condition: density vs perinuclear-ring YAP intensity",
            merged["existing_density_sample_residual"],
            merged["ring_yap_sample_residual"],
        ),
    ]
    output = pd.DataFrame(rows)
    output.to_csv(ROOT / "crowding_sensitivity_summary.csv", index=False)
    merged.to_csv(ROOT / "field_level_merged_for_sensitivity.csv", index=False)
    print(output.to_string(index=False))

    sample_rows = []
    for sample, group in merged.groupby("sample"):
        if len(group) < 3:
            continue
        ratio_result = spearmanr(group["existing_density_per_10000_um2"], group["existing_ratio_median"])
        nuclear_result = spearmanr(group["existing_density_per_10000_um2"], group["nuclear_yap_median"])
        sample_rows.append(
            {
                "sample": sample,
                "n_fields": len(group),
                "density_vs_ratio_rho": ratio_result.statistic,
                "density_vs_nuclear_yap_rho": nuclear_result.statistic,
            }
        )
    by_sample = pd.DataFrame(sample_rows).sort_values("sample")
    by_sample.to_csv(ROOT / "crowding_correlations_by_condition.csv", index=False)
    print("\n", by_sample.to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parent / "analysis_outputs" / "hepatocyte_confocal_audit"


def main() -> None:
    structure = pd.read_csv(BASE / "confocal_structure_and_saturation_audit.csv")
    yap_structure = structure.loc[structure["dataset"] == "yap", [
        "sample",
        "nuclei_per_10000_um2_proxy",
        "dapi_mask_fraction",
        "dapi_z80_span_um",
    ]].rename(
        columns={
            "nuclei_per_10000_um2_proxy": "raw_oib_nuclear_density_proxy",
            "dapi_mask_fraction": "raw_oib_dapi_coverage",
            "dapi_z80_span_um": "raw_oib_dapi_z80_span_um",
        }
    )
    yap = pd.read_csv(BASE / "existing_yap_sample_summary_deduplicated.csv")[[
        "sample",
        "n_cells",
        "n_fields",
        "ratio_median",
        "ratio_q10",
        "ratio_q90",
    ]].rename(
        columns={
            "n_cells": "yap_cells_after_deduplication",
            "n_fields": "yap_exported_fields_after_deduplication",
            "ratio_median": "yap_nucleus_ring_ratio_median",
            "ratio_q10": "yap_ratio_q10",
            "ratio_q90": "yap_ratio_q90",
        }
    )
    architecture = pd.read_csv(BASE / "ecad_actin" / "ecad_actin_architecture_metrics.csv")[[
        "sample",
        "tissue_coverage_fraction",
        "ecad_continuity_ge50_fraction",
        "actin_local_coherence_median",
    ]].rename(
        columns={
            "tissue_coverage_fraction": "ecad_stack_tissue_coverage",
            "ecad_continuity_ge50_fraction": "ecad_continuity_proxy",
            "actin_local_coherence_median": "f_actin_local_coherence_proxy",
        }
    )
    merged = yap.merge(yap_structure, on="sample", how="left").merge(architecture, on="sample", how="left")
    merged = merged.sort_values("sample", key=lambda values: values.map(lambda value: (int(value[1]), int(value[2]))))
    output = BASE / "condition_level_exploratory_summary.csv"
    merged.to_csv(output, index=False)
    print(merged.to_string(index=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()

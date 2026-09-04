"""Re-measure YAP only from existing masks/results. Never fit a morphology model.

PyCharm: run directly for composite/40x. Outputs go to a NEW versioned folder;
old results, masks, DAPI features, embeddings and full posteriors are preserved.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from yap_40x_trial import (YAPImageSet, measure_yap_posthoc, save_ring_qc_overlay,
                          save_yap_ratio_overlay, save_json, build_descriptive_tables)
from yap_ratio_qc import ALGORITHM_VERSION, load_config, qc_summary
from yap_ratio_display import save_comparison_figure, save_qc_summary_figure

MAGNIFICATION = "40x"
FEATURE_SET = "composite"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_posthoc(original, measurements):
    core = original.drop(columns=[c for c in original if c.startswith("posthoc_yap_")])
    keys = ["cell_id", "image_id", "label"]
    if core.cell_id.duplicated().any() or measurements.cell_id.duplicated().any():
        raise ValueError("Duplicate cell identifiers")
    result = core.merge(measurements, on=keys, how="left", validate="one_to_one", sort=False)
    if result["posthoc_yap_algorithm_version"].isna().any():
        raise ValueError("Some modeled cells are missing from YAP measurements")
    pd.testing.assert_frame_equal(result[core.columns], core, check_exact=True)
    return result


def refresh(source: Path, output: Path, config, background_roi_dir=None, figures=True):
    source, output = source.resolve(), output.resolve()
    if output == source or output in source.parents:
        raise ValueError("Output must not replace the source run or its parent")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}. Choose a NEW --output-root.")
    input_csv = source / "tables" / "model_a_single_cell_results.csv"
    original = pd.read_csv(input_csv)
    if original.magnification.nunique() != 1:
        raise ValueError("Refresh one independently fitted magnification at a time")
    image_rows = original.groupby("image_id", sort=False).first().reset_index()
    mask_paths = {row.image_id: source / "segmentation" / "masks" / f"{row.image_id}_mask.npy"
                  for row in image_rows.itertuples(index=False)}
    protected = [p for p in (source / "tables").iterdir() if p.is_file()]
    protected += list(mask_paths.values())
    if (source / "run_info.json").exists():
        protected.append(source / "run_info.json")
    before_hashes = {str(path): sha256(path) for path in protected}
    tables_dir = output / "tables"
    tables_dir.mkdir(parents=True, exist_ok=False)
    save_json(dict(status="incomplete", source_run=str(source), algorithm=ALGORITHM_VERSION), output / "refresh_info.json")
    all_tables, infos, image_items = [], [], []
    input_hashes = {}
    for row in image_rows.itertuples(index=False):
        item = YAPImageSet(str(row.experimental_group_label), str(row.seeding_density_folder_label),
                           float(row.seeding_density_cells_per_cm2), row.magnification,
                           int(row.export_index), row.image_id, Path(row.image_set_folder),
                           Path(row.dapi_org_path), Path(row.yap_af488_org_path), Path(row.merge_image_path))
        for path in (item.yap_org_path, item.merge_path):
            input_hashes[str(path)] = sha256(path)
        if background_roi_dir:
            roi = background_roi_dir / f"{item.image_id}_background.png"
            if roi.exists():
                input_hashes[str(roi.resolve())] = sha256(roi)
        table, rings, info = measure_yap_posthoc(item, mask_paths[item.image_id], config, background_roi_dir)
        all_tables.append(table)
        infos.append(info)
        image_items.append(item)
        if figures:
            save_ring_qc_overlay(item, mask_paths[item.image_id], rings, info,
                                 output / "figures" / "ring_qc" / f"{item.image_id}_ring_qc.png", table)
        print(f"[posthoc-only] {item.image_id}: sampling={int(table.posthoc_yap_sampling_valid.sum())}, raw ratios={int(table.posthoc_yap_raw_ratio_valid.sum())}, corrected ratios={int(table.posthoc_yap_ratio_valid.sum())}, segmented nuclei={len(table)}", flush=True)
    measurements = pd.concat(all_tables, ignore_index=True)
    result = replace_posthoc(original, measurements)
    probability_columns = [c for c in result if c.startswith("P_phenotype_")]
    if not probability_columns or not np.allclose(result[probability_columns].sum(axis=1), 1, atol=1e-8):
        raise AssertionError("Original full GMM posterior is missing or invalid")
    measurements.to_csv(tables_dir / "yap_nuclear_perinuclear_measurements.csv", index=False, encoding="utf-8-sig")
    result.to_csv(tables_dir / "model_a_single_cell_results.csv", index=False, encoding="utf-8-sig")
    summary = qc_summary(result)
    summary.to_csv(tables_dir / "yap_technical_qc_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(infos).to_csv(tables_dir / "yap_ring_and_background_qc.csv", index=False, encoding="utf-8-sig")
    reasons = result.assign(qc_reason=result.posthoc_yap_qc_reasons.str.split(";")).explode("qc_reason")
    reasons.groupby(["image_id", "qc_reason"]).size().rename("n_cells").to_csv(tables_dir / "yap_qc_reason_counts.csv", encoding="utf-8-sig")
    warnings = result.assign(warning=result.posthoc_yap_qc_warnings.str.split(";")).explode("warning")
    warnings.groupby(["image_id", "warning"]).size().rename("n_cells").to_csv(tables_dir / "yap_quality_warning_counts.csv", encoding="utf-8-sig")
    phenotype_qc = result.groupby(["image_id", "dominant_phenotype"]).agg(
        n_dapi_cells=("cell_id", "size"), n_ratio_pass=("posthoc_yap_ratio_valid", "sum"),
        median_ratio=("posthoc_yap_nuclear_perinuclear_ratio", "median"))
    phenotype_qc["pass_fraction"] = phenotype_qc.n_ratio_pass / phenotype_qc.n_dapi_cells
    phenotype_qc.to_csv(tables_dir / "phenotype_posthoc_yap_summary.csv", encoding="utf-8-sig")
    for name, table in build_descriptive_tables(result).items():
        table.to_csv(tables_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    if figures:
        for item in image_items:
            save_yap_ratio_overlay(item, mask_paths[item.image_id], result,
                                   output / "figures" / "yap_ratio_overlays" / f"{item.image_id}_yap_ratio.png")
            save_yap_ratio_overlay(item, mask_paths[item.image_id], result,
                                   output / "figures" / "yap_ratio_raw_uncorrected" / f"{item.image_id}_yap_ratio_raw.png", raw=True)
            save_comparison_figure(np.load(mask_paths[item.image_id]),
                                   result[result.image_id.eq(item.image_id)], item.image_id,
                                   output / "figures" / "v1_v3_comparison" / f"{item.image_id}_comparison.png")
        save_qc_summary_figure(result, output / "figures" / "yap_qc_summary.png")
    after_hashes = {str(path): sha256(path) for path in protected}
    if before_hashes != after_hashes:
        raise AssertionError("Source run changed during refresh; inspect before using results")
    if any(sha256(path) != value for path, value in input_hashes.items()):
        raise AssertionError("Input images/ROIs changed during refresh")
    core_cols = [c for c in original if not c.startswith("posthoc_yap_")]
    persisted = pd.read_csv(tables_dir / "model_a_single_cell_results.csv")
    pd.testing.assert_frame_equal(persisted[core_cols], original[core_cols], check_exact=False, rtol=1e-12, atol=1e-12)
    report = dict(status="complete", algorithm_version=ALGORITHM_VERSION,
                  source_run=str(source), output_run=str(output), config=asdict(config),
                  n_dapi_cells=len(result), n_ratio_qc_pass=int(result.posthoc_yap_ratio_valid.sum()),
                  n_sampling_available=int(result.posthoc_yap_sampling_valid.sum()),
                  n_raw_ratio_available=int(result.posthoc_yap_raw_ratio_valid.sum()),
                  n_corrected_ratio_available=int(result.posthoc_yap_ratio_valid.sum()),
                  all_sampling_pixels_exclude_all_dapi_nuclei=all(i["sampling_pixels_overlapping_dapi_nuclei"] == 0 for i in infos),
                  ratio_valid_interpretation="corrected ratio available, not quality validation; raw ratios retained separately",
                  n_fields=len(image_rows), n_fields_background_unavailable=sum(not i["background_adequate"] for i in infos),
                  dapi_feature_metadata_embedding_and_full_posterior_unchanged=True,
                  source_files_byte_identical=True, original_files_sha256=before_hashes,
                  marker_inputs_sha256=input_hashes,
                  limitation="DAPI-excluded perinuclear proxy, NOT validated cytoplasm; small/noisy regions retained with warnings, missing background never replaced silently")
    save_json(report, output / "refresh_info.json")
    print(f"[complete] {report['n_sampling_available']}/{len(result)} cells have sampling pixels; {report['n_raw_ratio_available']} raw ratios, {report['n_corrected_ratio_available']} corrected ratios. All original model data unchanged. {output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--fit-magnification", default=MAGNIFICATION, choices=["20x", "40x"])
    parser.add_argument("--feature-set", default=FEATURE_SET, choices=["baseline", "composite"])
    parser.add_argument("--yap-qc-config", type=Path)
    parser.add_argument("--yap-background-roi-dir", type=Path)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    source = args.source_root or Path(__file__).resolve().parent / "outputs" / args.feature_set / args.fit_magnification / "all_fields"
    output = args.output_root or source / ("yap_dapi_excluded_v3_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    refresh(source, output, load_config(args.yap_qc_config), args.yap_background_roi_dir, not args.no_figures)


if __name__ == "__main__":
    main()

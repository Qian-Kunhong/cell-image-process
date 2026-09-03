"""Regenerate phenotype displays from saved results, without refitting any model."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    from yap_40x_trial import discover_image_sets, save_phenotype_overlay
    root = args.output_root.resolve()
    result_path = root / "tables" / "model_a_single_cell_results.csv"
    result_hash = digest(result_path)
    result = pd.read_csv(result_path)
    info = json.loads((root / "run_info.json").read_text(encoding="utf-8"))
    sets, _ = discover_image_sets(Path(info["data_root"]))
    selected = [item for item in sets if item.image_id in set(result.image_id)]
    if len(selected) != result.image_id.nunique():
        raise ValueError("Cannot resolve every result image")
    audit = {"purpose": "display only; no refitting or change to cell results", "images": []}
    for item in selected:
        mask_path = root / "segmentation" / "masks" / f"{item.image_id}_mask.npy"
        before = digest(mask_path)
        path = root / "figures" / "phenotype_overlays" / f"{item.image_id}_phenotypes.png"
        backup = root / "figures" / "phenotype_overlays_before_display_fix" / path.name
        if path.exists() and not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
        save_phenotype_overlay(item, mask_path, result, path)
        if digest(mask_path) != before:
            raise AssertionError("Raw segmentation changed")
        audit["images"].append({"image_id": item.image_id, "raw_mask_sha256": before, "raw_mask_unchanged": True})
        print(f"[replotted] {item.image_id}", flush=True)
    if digest(result_path) != result_hash:
        raise AssertionError("Cell results changed")
    audit["results_sha256"] = result_hash
    audit["results_unchanged"] = True
    (root / "tables" / "phenotype_display_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

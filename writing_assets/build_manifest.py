"""Build a stable, machine-readable index for manuscript code and figures.

The default action only writes manifests.  Use ``--copy publication`` to create
a self-contained figure bundle without touching the analysis outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
CODE_SUFFIXES = {".py", ".ps1"}
SKIP_PARTS = {".git", "__pycache__", "writing_assets"}


def relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_role(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("test_") or "tests" in path.parts:
        return "test"
    if name.startswith("run_") or name in {"day2_trial.py"}:
        return "entrypoint"
    if name.startswith("make_") or name.startswith("refresh_"):
        return "figure_or_posthoc_entrypoint"
    if name.startswith("analyze_") or name.startswith("evaluate_") or name.startswith("summarize_"):
        return "analysis_entrypoint"
    if "shared" in path.parts or "ekin_dapi_yap" in path.parts:
        return "library"
    return "workflow_script"


def image_kind(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "figures_ppt" in parts or "figures_ha_main" in parts:
        return "publication_candidate"
    if "phenotype_overlays" in parts:
        return "field_overlay"
    if "ring_qc" in parts or "segmentation" in parts or "overlays" in parts:
        return "qc"
    if "figures" in parts:
        return "analysis_figure"
    return "other_image"


def dataset_name(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).parts[0]


def source_status(path: Path) -> str:
    top = dataset_name(path)
    if top == "Ekin_DAPI_YAP - Doubao":
        return "alternate_snapshot"
    if top == "Hochest_OCT4_1.0":
        return "legacy_output"
    if path.name.startswith("tmp_"):
        return "temporary"
    return "main"


def discover_code() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in CODE_SUFFIXES:
            continue
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        rows.append(
            {
                "dataset": dataset_name(path),
                "source_status": source_status(path),
                "role": code_role(path),
                "name": path.name,
                "relative_path": relative(path),
                "absolute_path": str(path.resolve()),
            }
        )
    return rows


def discover_images() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        kind = image_kind(path)
        status = source_status(path)
        rows.append(
            {
                "dataset": dataset_name(path),
                "source_status": status,
                "kind": kind,
                "publication_candidate": kind == "publication_candidate",
                "recommended_for_writing": kind == "publication_candidate" and status == "main",
                "format": path.suffix.lower().lstrip("."),
                "bytes": path.stat().st_size,
                "modified": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                "relative_path": relative(path),
                "absolute_path": str(path.resolve()),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def copy_images(rows: list[dict[str, object]], selection: str, destination: Path) -> int:
    if selection == "all":
        chosen = rows
    elif selection == "publication":
        chosen = [row for row in rows if row["publication_candidate"]]
    else:
        chosen = [row for row in rows if row["recommended_for_writing"]]
    for row in chosen:
        source = Path(str(row["absolute_path"]))
        target = destination / Path(str(row["relative_path"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return len(chosen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--copy",
        choices=("none", "recommended", "publication", "all"),
        default="none",
        help="Optionally copy figures into a self-contained bundle.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=EXPORT_ROOT / "exported_figures",
        help="Copy destination (used only when --copy is not 'none').",
    )
    args = parser.parse_args()

    code = discover_code()
    images = discover_images()
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(EXPORT_ROOT / "code_manifest.csv", code)
    write_csv(EXPORT_ROOT / "image_manifest.csv", images)

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT.resolve()),
        "counts": {
            "code_files": len(code),
            "images": len(images),
            "publication_candidates": sum(bool(row["publication_candidate"]) for row in images),
            "recommended_for_writing": sum(bool(row["recommended_for_writing"]) for row in images),
        },
        "code": code,
        "images": images,
    }
    (EXPORT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    copied = 0
    if args.copy != "none":
        copied = copy_images(images, args.copy, args.destination.resolve())
    print(
        f"Indexed {len(code)} code files and {len(images)} images "
        f"({manifest['counts']['publication_candidates']} publication candidates); copied {copied}."
    )


if __name__ == "__main__":
    main()

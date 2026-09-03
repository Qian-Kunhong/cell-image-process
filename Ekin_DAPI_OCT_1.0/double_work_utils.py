from __future__ import annotations

import importlib.util
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


IMG_EXTENSIONS = (".tif", ".tiff")
MASK_SUFFIX = "_mask.npy"
DEFAULT_DAPI_TOKEN_REGEX = r"(?i)dapi"
DEFAULT_OCT4_TOKEN_REGEX = r"(?i)(oct[\s\-_]*4|texas[\s\-_]*red)"
DEFAULT_EXCLUDE_TOKEN_REGEX = r"(?i)(tl[\s\-_]*brightfield|bright[\s\-_]*field)"

DEFAULT_LABEL_COLORS = {
    "cluster1": "#3cb65a",
    "cluster2": "#d63a3a",
    "uncertain": "#e7b93a",
    "deviated": "#d63a3a",
    "undifferentiated": "#3cb65a",
}


@dataclass(frozen=True)
class PairRecord:
    pair_key: str
    image_name: str
    mask_path: Path
    dapi_path: Path
    oct4_path: Path


def load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sibling_single_work_dir(current_file: Path) -> Path:
    return current_file.resolve().parent.parent / "cell image"


def strip_mask_suffix(stem: str) -> str:
    return stem[:-len("_mask")] if stem.endswith("_mask") else stem


def normalize_pair_key(name: str) -> str:
    key = strip_mask_suffix(Path(name).stem).lower()
    key = re.sub(DEFAULT_DAPI_TOKEN_REGEX, " ", key)
    key = re.sub(DEFAULT_OCT4_TOKEN_REGEX, " ", key)
    key = re.sub(DEFAULT_EXCLUDE_TOKEN_REGEX, " ", key)
    key = re.sub(r"(?i)\bwv\b", " ", key)
    key = re.sub(r"[^a-z0-9]+", " ", key)
    return key.strip()


def list_image_files(img_dir: Path) -> list[Path]:
    files = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS]
    return sorted(files)


def is_excluded_channel_file(path: Path, exclude_token_regex: str = DEFAULT_EXCLUDE_TOKEN_REGEX) -> bool:
    return bool(re.search(exclude_token_regex, path.stem))


def select_channel_files(
    files: Iterable[Path],
    token_regex: str,
    exclude_token_regex: str = DEFAULT_EXCLUDE_TOKEN_REGEX,
) -> list[Path]:
    files = sorted(files)
    filtered = [p for p in files if not is_excluded_channel_file(p, exclude_token_regex)]
    matched = [p for p in filtered if re.search(token_regex, p.stem)]
    if matched:
        return matched
    return filtered


def build_channel_index(files: Iterable[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(files):
        key = normalize_pair_key(path.stem)
        if key and key not in index:
            index[key] = path
    return index


def build_pair_records(
    dapi_dir: Path,
    oct4_dir: Path,
    mask_dir: Path,
    dapi_token_regex: str = DEFAULT_DAPI_TOKEN_REGEX,
    oct4_token_regex: str = DEFAULT_OCT4_TOKEN_REGEX,
    exclude_token_regex: str = DEFAULT_EXCLUDE_TOKEN_REGEX,
    out_report_csv: Path | None = None,
) -> tuple[list[PairRecord], pd.DataFrame]:
    if not dapi_dir.exists():
        raise FileNotFoundError(f"DAPI image folder not found: {dapi_dir}")
    if not oct4_dir.exists():
        raise FileNotFoundError(f"Oct-4 image folder not found: {oct4_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask folder not found: {mask_dir}")

    dapi_files = select_channel_files(list_image_files(dapi_dir), dapi_token_regex, exclude_token_regex)
    oct4_files = select_channel_files(list_image_files(oct4_dir), oct4_token_regex, exclude_token_regex)
    dapi_index = build_channel_index(dapi_files)
    oct4_index = build_channel_index(oct4_files)

    mask_files = sorted(mask_dir.glob(f"*{MASK_SUFFIX}"))
    if not mask_files:
        raise RuntimeError(f"No mask files found in: {mask_dir}")

    records: list[PairRecord] = []
    report_rows: list[dict] = []

    for mask_path in mask_files:
        image_name = strip_mask_suffix(mask_path.stem)
        pair_key = normalize_pair_key(image_name)

        dapi_path = None
        for ext in IMG_EXTENSIONS:
            candidate = dapi_dir / f"{image_name}{ext}"
            if candidate.exists() and not is_excluded_channel_file(candidate, exclude_token_regex):
                dapi_path = candidate
                break
        if dapi_path is None:
            dapi_path = dapi_index.get(pair_key)

        oct4_path = oct4_index.get(pair_key)
        status = "ok"
        if dapi_path is None and oct4_path is None:
            status = "missing_dapi_and_oct4"
        elif dapi_path is None:
            status = "missing_dapi"
        elif oct4_path is None:
            status = "missing_oct4"

        report_rows.append(
            {
                "pair_key": pair_key,
                "image_name": image_name,
                "mask_path": str(mask_path),
                "dapi_path": str(dapi_path) if dapi_path is not None else None,
                "oct4_path": str(oct4_path) if oct4_path is not None else None,
                "status": status,
            }
        )
        if status == "ok":
            records.append(
                PairRecord(
                    pair_key=pair_key,
                    image_name=dapi_path.stem,
                    mask_path=mask_path,
                    dapi_path=dapi_path,
                    oct4_path=oct4_path,
                )
            )

    report_df = pd.DataFrame(report_rows)
    if out_report_csv is not None:
        out_report_csv.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(out_report_csv, index=False, encoding="utf-8-sig")

    if not records:
        raise RuntimeError(
            "No valid DAPI/Oct-4/mask triplets were found. "
            "Please check the channel folders and filename pairing rules."
        )

    return records, report_df


def add_pair_metadata(df: pd.DataFrame, dapi_path: Path, oct4_path: Path, pair_key: str) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    out["pair_key"] = pair_key
    out["dapi_image_name"] = dapi_path.stem
    out["oct4_image_name"] = oct4_path.stem
    return out


def normalize_to_u8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    p1, p99 = np.percentile(arr, [1, 99])
    arr = np.clip((arr - p1) / (p99 - p1 + 1e-6), 0, 1)
    return (arr * 255).astype(np.uint8)


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    boundary = np.zeros_like(mask, dtype=bool)
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return boundary & (mask > 0)


def make_overlay_preview(
    img_gray: np.ndarray,
    mask: np.ndarray,
    boundary_rgb: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    base = normalize_to_u8(img_gray)
    rgb = np.stack([base, base, base], axis=-1)
    rgb[mask_boundary(mask)] = np.array(boundary_rgb, dtype=np.uint8)
    return rgb


def save_preview_image(preview: np.ndarray, out_tif: Path | None = None, out_png: Path | None = None) -> None:
    arr = np.asarray(preview)
    if out_tif is not None:
        out_tif.parent.mkdir(parents=True, exist_ok=True)
        tiff_arr = arr.astype(np.uint8, copy=False)
        import tifffile as tiff
        tiff.imwrite(str(out_tif), tiff_arr)
    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr.astype(np.uint8, copy=False)).save(out_png)


def choose_existing_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Missing required columns. Candidates checked: {candidates}")


def save_score_intensity_relationship(
    scored_df: pd.DataFrame,
    intensity_csv: Path,
    out_csv: Path,
    out_fig: Path,
    *,
    score_candidates: list[str],
    intensity_col: str = "mean_intensity",
    state_candidates: list[str] | None = None,
    score_label: str = "deviation_score",
    intensity_label: str = "Oct-4 mean intensity",
    title_prefix: str = "Deviation score vs Oct-4 mean intensity",
    out_summary_json: Path | None = None,
) -> dict:
    if not intensity_csv.exists():
        raise FileNotFoundError(f"Intensity CSV not found: {intensity_csv}")

    intensity_df = pd.read_csv(intensity_csv)
    required = ["image_name", "label", intensity_col]
    missing = [c for c in required if c not in intensity_df.columns]
    if missing:
        raise KeyError(f"Intensity CSV missing required columns: {missing}")

    score_col = choose_existing_column(scored_df, score_candidates)
    merge_cols = ["image_name", "label", score_col]
    state_col = None
    if state_candidates:
        for candidate in state_candidates:
            if candidate in scored_df.columns:
                state_col = candidate
                merge_cols.append(candidate)
                break

    merged = scored_df.loc[:, merge_cols].merge(
        intensity_df.loc[:, ["image_name", "label", intensity_col]],
        on=["image_name", "label"],
        how="inner",
    )

    merged[score_col] = pd.to_numeric(merged[score_col], errors="coerce")
    merged[intensity_col] = pd.to_numeric(merged[intensity_col], errors="coerce")
    valid = merged[score_col].notna() & merged[intensity_col].notna()
    rel_df = merged.loc[valid].copy()
    if len(rel_df) == 0:
        raise RuntimeError("No valid rows available for score/intensity relationship analysis.")

    pearson_r = float(rel_df[score_col].corr(rel_df[intensity_col], method="pearson"))
    spearman_r = float(rel_df[score_col].corr(rel_df[intensity_col], method="spearman"))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rel_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(7.8, 6.2))
    if state_col is not None:
        for state_name, sub_df in rel_df.groupby(state_col, sort=False):
            color = DEFAULT_LABEL_COLORS.get(str(state_name), "#808080")
            plt.scatter(
                sub_df[intensity_col],
                sub_df[score_col],
                s=12,
                alpha=0.35,
                color=color,
                label=str(state_name),
            )
        plt.legend(frameon=True)
    else:
        plt.scatter(
            rel_df[intensity_col],
            rel_df[score_col],
            s=12,
            alpha=0.35,
            color="#4f81bd",
        )

    plt.xlabel(intensity_label)
    plt.ylabel(score_label)
    plt.title(
        f"{title_prefix}\n"
        f"Pearson r = {pearson_r:.3f}, Spearman rho = {spearman_r:.3f}, n = {len(rel_df)}"
    )
    plt.tight_layout()
    plt.savefig(out_fig, dpi=220, bbox_inches="tight")
    plt.close()

    summary = {
        "n_valid": int(len(rel_df)),
        "score_column": score_col,
        "intensity_column": intensity_col,
        "pearson_r": None if math.isnan(pearson_r) else pearson_r,
        "spearman_rho": None if math.isnan(spearman_r) else spearman_r,
        "output_csv": str(out_csv),
        "output_figure": str(out_fig),
    }
    if state_col is not None:
        summary["state_column"] = state_col

    if out_summary_json is not None:
        with open(out_summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary

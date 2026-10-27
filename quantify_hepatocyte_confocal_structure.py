from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation


WORKSPACE = Path(__file__).resolve().parent
DEPENDENCY_DIR = WORKSPACE / ".codex_deps" / "oiffile"
if str(DEPENDENCY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPENDENCY_DIR))

import oiffile  # noqa: E402


RAW_ROOT = Path(
    r"F:\Naruchit data\1_Data of published manuscripts\3_Hepatocyte differentiation"
    r"\03_Raw data\Immunostaining"
)
DATASETS = {
    "ecad_f_actin": RAW_ROOT / "Confocal - E-cad F-actin",
    "yap": RAW_ROOT / "Confocal - YAP",
}
OUTPUT_DIR = WORKSPACE / "analysis_outputs" / "hepatocyte_confocal_audit"


def ensure_zcyx(array: np.ndarray, axes: str) -> np.ndarray:
    if axes != "ZCYX":
        raise ValueError(f"Expected ZCYX metadata, received {axes} with shape {array.shape}")
    if array.ndim == 4 and array.shape[0] <= 4 and array.shape[1] > 4:
        return np.transpose(array, (1, 0, 2, 3))
    return array


def focus_slice_index(array: np.ndarray) -> int:
    dapi = array[:, 0].astype(np.float32)
    score = np.mean(np.abs(np.diff(dapi, axis=1)), axis=(1, 2))
    score += np.mean(np.abs(np.diff(dapi, axis=2)), axis=(1, 2))
    return int(np.argmax(score))


def dapi_mask_and_count(dapi_mip: np.ndarray) -> tuple[np.ndarray, int]:
    image = dapi_mip.astype(np.float32)
    positive = image[image > 0]
    if positive.size == 0:
        return np.zeros(image.shape, dtype=bool), 0
    lo, hi = np.percentile(positive, (1, 99.5))
    norm = np.clip((image - lo) / max(hi - lo, 1.0), 0, 1)
    smooth = filters.gaussian(norm, sigma=1.2, preserve_range=True)
    threshold = filters.threshold_otsu(smooth)
    mask = smooth > threshold
    mask = morphology.remove_small_objects(mask, min_size=60)
    mask = morphology.remove_small_holes(mask, area_threshold=80)

    distance = ndi.distance_transform_edt(mask)
    # Conservative markers: this gives a density proxy, not a validated count.
    maxima = morphology.h_maxima(distance, 2.2)
    markers = measure.label(maxima)
    labels = segmentation.watershed(-distance, markers, mask=mask)
    areas = np.bincount(labels.ravel())[1:]
    plausible = np.flatnonzero((areas >= 80) & (areas <= 8000)) + 1
    cleaned = np.isin(labels, plausible)
    return cleaned, int(plausible.size)


def robust_z_span(dapi: np.ndarray, mask_2d: np.ndarray) -> tuple[float, float, float, float]:
    # Per-plane nuclear signal after subtracting a low background percentile.
    if not np.any(mask_2d):
        return np.nan, np.nan, np.nan, np.nan
    traces = []
    for plane in dapi.astype(np.float32):
        background = float(np.percentile(plane, 20))
        traces.append(float(np.clip(plane[mask_2d] - background, 0, None).sum()))
    trace = np.asarray(traces)
    if trace.sum() <= 0:
        return np.nan, np.nan, np.nan, np.nan
    cumulative = np.cumsum(trace) / trace.sum()
    z10 = int(np.searchsorted(cumulative, 0.10))
    z50 = int(np.searchsorted(cumulative, 0.50))
    z90 = int(np.searchsorted(cumulative, 0.90))
    peak = int(np.argmax(trace))
    return float(z10), float(z50), float(z90), float(peak)


def audit_file(dataset: str, path: Path) -> dict:
    with oiffile.OifFile(path) as handle:
        array = ensure_zcyx(handle.asarray(), handle.axes)
        settings = handle.mainfile
        z_step_um = float(settings.get("Axis 3 Parameters Common", {}).get("Interval", 0)) / 1000
        dapi = array[:, 0]
        dapi_mip = dapi.max(axis=0)
        mask, count_proxy = dapi_mask_and_count(dapi_mip)
        z10, z50, z90, zpeak = robust_z_span(dapi, mask)
        row = {
            "dataset": dataset,
            "sample": path.stem,
            "z_planes": array.shape[0],
            "focus_z_index": focus_slice_index(array),
            "dapi_mask_fraction": float(mask.mean()),
            "nucleus_count_proxy": count_proxy,
            "nuclei_per_10000_um2_proxy": count_proxy / (140.624 * 140.624) * 10000,
            "dapi_z10_index": z10,
            "dapi_z50_index": z50,
            "dapi_z90_index": z90,
            "dapi_z80_span_um": (z90 - z10) * z_step_um,
            "dapi_signal_peak_z_index": zpeak,
        }
        for channel_idx in range(array.shape[1]):
            values = array[:, channel_idx]
            prefix = f"ch{channel_idx + 1}"
            row[f"{prefix}_fraction_at_4095"] = float(np.mean(values >= 4095))
            nonzero = values > 0
            row[f"{prefix}_nonzero_fraction_at_4095"] = (
                float(np.mean(values[nonzero] >= 4095)) if np.any(nonzero) else np.nan
            )
        return row


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, directory in DATASETS.items():
        for path in sorted(directory.glob("*.oib")):
            print(f"Quantifying {dataset}/{path.name}", flush=True)
            rows.append(audit_file(dataset, path))
    fieldnames = list(rows[0])
    output = OUTPUT_DIR / "confocal_structure_and_saturation_audit.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()

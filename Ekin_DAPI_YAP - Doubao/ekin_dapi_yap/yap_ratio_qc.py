"""Post-hoc YAP nuclear/perinuclear sampling with DAPI nuclear exclusion.

No foreground pixel selection is used to compute intensities. Any nonempty
ring is measured; support, noise and clipping are warnings, not hard gates.
Raw and background-corrected ratios are separate, never mixed by a fallback.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops
from skimage.segmentation import expand_labels

ALGORITHM_VERSION = "dapi_excluded_perinuclear_v3.0"


@dataclass(frozen=True)
class YAPQCConfig:
    inner_gap_fraction: float = 0.0  # Exclude actual nuclear masks, no extra empty buffer.
    outer_distance_fraction: float = 0.20
    wide_distance_fraction: float = 0.35
    background_distance_fraction: float = 1.5
    min_background_pixels: int = 100
    min_background_image_fraction: float = 0.005
    noise_floor_dn: float = 1.0  # One exported 8-bit intensity unit, not camera noise calibration.
    min_snr: float = 3.0
    min_ring_pixels: int = 1
    warning_ring_pixels: int = 20  # Descriptive small-sample flag only.
    min_ring_nucleus_area_ratio: float = 0.20
    min_signal_fraction: float = 0.70
    angular_sectors: int = 8
    min_supported_sectors: int = 6
    min_sector_pixels: int = 3
    min_sector_signal_fraction: float = 0.50
    saturation_dn: float = 255.0
    max_saturated_fraction: float = 0.01

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid YAP QC setting: {name}={value}")
        if not 0 <= self.inner_gap_fraction < self.outer_distance_fraction <= self.wide_distance_fraction:
            raise ValueError("Require 0 <= inner < outer <= wide ring fractions")
        if self.background_distance_fraction <= self.wide_distance_fraction:
            raise ValueError("Background distance must exceed both sampling rings")
        for name in ("min_background_pixels", "min_ring_pixels", "warning_ring_pixels", "angular_sectors", "min_supported_sectors", "min_sector_pixels"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_supported_sectors > self.angular_sectors:
            raise ValueError("Supported sectors cannot exceed total sectors")
        if self.noise_floor_dn <= 0 or self.min_snr <= 0 or self.saturation_dn <= 0:
            raise ValueError("Noise floor, SNR and saturation level must be positive")
        for name in ("min_signal_fraction", "min_sector_signal_fraction", "max_saturated_fraction", "min_background_image_fraction"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")


def load_config(path: Path | None = None) -> YAPQCConfig:
    return YAPQCConfig(**json.loads(path.read_text(encoding="utf-8"))) if path else YAPQCConfig()


def build_ring_labels(mask: np.ndarray, outer_fraction: float = 0.35,
                      inner_fraction: float = 0.08) -> tuple[np.ndarray, int, int, float]:
    props = regionprops(mask)
    if not props:
        raise ValueError("Cannot construct rings without segmented nuclei")
    diameter = float(np.median([p.equivalent_diameter_area for p in props]))
    inner = 0 if inner_fraction == 0 else max(1, int(round(inner_fraction * diameter)))
    outer = max(inner + 2, int(round(outer_fraction * diameter)))
    rings = expand_labels(mask, distance=outer)
    rings[expand_labels(mask, distance=inner) > 0] = 0
    if np.any((rings > 0) & (mask > 0)):
        raise AssertionError("Sampling pixels must not overlap ANY segmented DAPI nucleus")
    return rings.astype(np.int32), inner, outer, diameter


def estimate_background(yap: np.ndarray, mask: np.ndarray, diameter: float,
                        config: YAPQCConfig, background_roi: np.ndarray | None = None):
    minimum = max(config.min_background_pixels, int(np.ceil(config.min_background_image_fraction * mask.size)))
    if background_roi is None:
        distance = max(2, int(round(config.background_distance_fraction * diameter)))
        selected = distance_transform_edt(mask == 0) > distance
        method = "inferred_far_from_all_nuclei; not independently verified cell-free"
    else:
        if background_roi.shape != mask.shape or background_roi.ndim != 2:
            raise ValueError("Background ROI must match the 2D nuclear mask")
        selected = background_roi.astype(bool)
        if np.any(selected & (mask > 0)):
            raise ValueError("Background ROI overlaps segmented nuclei; review the ROI")
        method = "user_supplied_cell_free_background_ROI"
    n = int(selected.sum())
    adequate = n >= minimum
    values = yap[selected].astype(float)
    bg = float(np.median(values)) if adequate else np.nan
    mad_sigma = float(1.4826 * np.median(np.abs(values - bg))) if adequate else np.nan
    noise = max(config.noise_floor_dn, mad_sigma) if adequate else np.nan
    info = dict(background=bg, background_method=method, background_n_pixels=n,
                background_min_required_pixels=minimum, background_adequate=adequate,
                background_mad_sigma_dn=mad_sigma, background_noise_scale_dn=noise,
                background_assumption="Inferred empty space may include unsegmented cells/cytoplasm; inspect background QC")
    return info, selected


def _positive_ratio(n: float, c: float) -> float:
    return float(n / c) if np.isfinite(n) and np.isfinite(c) and n > 0 and c > 0 else np.nan


def _stat(values, kind="median"):
    return float(getattr(np, kind)(values)) if len(values) else np.nan


def measure_array(yap: np.ndarray, mask: np.ndarray, image_id: str,
                  config: YAPQCConfig | None = None, background_roi: np.ndarray | None = None):
    config = config or YAPQCConfig()
    if yap.ndim != 2 or mask.ndim != 2 or yap.shape != mask.shape:
        raise ValueError("YAP image and nuclear mask must have identical 2D shapes")
    if not np.issubdtype(mask.dtype, np.integer) or np.any(mask < 0) or not np.isfinite(yap).all():
        raise ValueError("Require nonnegative integer nuclear labels and finite intensities")
    rings, inner, outer, diameter = build_ring_labels(mask, config.outer_distance_fraction, config.inner_gap_fraction)
    wide, _, wide_outer, _ = build_ring_labels(mask, config.wide_distance_fraction, config.inner_gap_fraction)
    bg_info, _ = estimate_background(yap, mask, diameter, config, background_roi)
    bg, noise = bg_info["background"], bg_info["background_noise_scale_dn"]

    # Reproduce the v1 median measurement for audit only (including its fallback).
    legacy, _, legacy_outer, _ = build_ring_labels(mask, 0.35, 0.08)
    legacy_far = distance_transform_edt(mask == 0) > max(legacy_outer + 2, int(round(1.5 * diameter)))
    legacy_has_bg = int(legacy_far.sum()) >= max(100, int(0.005 * mask.size))
    legacy_bg = float(np.median(yap[legacy_far])) if legacy_has_bg else float(np.percentile(yap, 1))

    rows = []
    for prop in regionprops(mask):
        label = int(prop.label)
        # Crop for efficiency; all neighboring nuclei stay in the full label map.
        r0, c0, r1, c1 = prop.bbox
        pad = max(outer, wide_outer, legacy_outer) + 1
        sl = (slice(max(0, r0 - pad), min(mask.shape[0], r1 + pad)),
              slice(max(0, c0 - pad), min(mask.shape[1], c1 + pad)))
        nucleus = yap[prop.slice][prop.image].astype(float)
        local_ring = rings[sl] == label
        values = yap[sl][local_ring].astype(float)
        wide_values = yap[sl][wide[sl] == label].astype(float)
        legacy_values = yap[sl][legacy[sl] == label].astype(float)
        n_med, r_med = _stat(nucleus), _stat(values)
        n_mean, r_mean = _stat(nucleus, "mean"), _stat(values, "mean")
        n_corr, r_corr = n_med - bg, r_med - bg
        ratio_unfiltered = _positive_ratio(n_corr, r_corr)
        nuclear_snr, ring_snr = n_corr / noise, r_corr / noise
        area_fraction = len(values) / len(nucleus)
        sampling_valid = len(values) >= config.min_ring_pixels
        coverage = len(values) >= config.warning_ring_pixels and area_fraction >= config.min_ring_nucleus_area_ratio
        clipped = min(r0, c0, mask.shape[0] - r1, mask.shape[1] - c1) <= outer
        saturation_n = float(np.mean(nucleus >= config.saturation_dn))
        saturation_r = float(np.mean(values >= config.saturation_dn)) if len(values) else 0.0
        rr, cc = np.nonzero(local_ring)
        angles = np.arctan2(rr + sl[0].start - prop.centroid[0], cc + sl[1].start - prop.centroid[1])
        sectors = np.minimum(config.angular_sectors - 1,
                             ((angles + np.pi) / (2 * np.pi) * config.angular_sectors).astype(int))
        counts = np.bincount(sectors, minlength=config.angular_sectors)

        def signal_support(snr):
            above = values > bg + snr * noise
            fraction = float(above.mean()) if len(values) and bg_info["background_adequate"] else np.nan
            positive = np.bincount(sectors, weights=above.astype(float), minlength=config.angular_sectors).astype(float)
            fractions = np.divide(positive, counts, out=np.zeros_like(positive), where=counts > 0)
            supported = int(np.sum((counts >= config.min_sector_pixels) & (fractions >= config.min_sector_signal_fraction)))
            return fraction, supported

        signal_fraction, supported_sectors = signal_support(config.min_snr)
        geometric_reasons = []
        if not bg_info["background_adequate"]:
            geometric_reasons.append("background_unavailable")
        if not coverage:
            geometric_reasons.append("insufficient_ring_area")
        if clipped:
            geometric_reasons.append("ring_near_image_boundary")
        if max(saturation_n, saturation_r) > config.max_saturated_fraction:
            geometric_reasons.append("export_intensity_clipping")

        def reasons_at(snr):
            reasons = list(geometric_reasons)
            if bg_info["background_adequate"]:
                support, supported = signal_support(snr)
                if not nuclear_snr > snr:
                    reasons.append("nuclear_signal_below_noise_gate")
                if not ring_snr > snr:
                    reasons.append("denominator_below_noise_gate")
                if not support >= config.min_signal_fraction:
                    reasons.append("ring_low_signal_fraction")
                if supported < config.min_supported_sectors:
                    reasons.append("insufficient_angular_signal_support")
            return reasons

        # v3: the old area/angular/signal gates survive ONLY as warnings.
        # A finite ratio does not certify sample quality or cytoplasmic identity.
        warnings = reasons_at(config.min_snr)
        reasons = []
        if not sampling_valid:
            reasons.append("no_sampling_pixels" if not len(values) else "below_configured_minimum_pixels")
        if not bg_info["background_adequate"]:
            reasons.append("background_unavailable")
        elif sampling_valid:
            if not n_corr > 0:
                reasons.append("nonpositive_corrected_nuclear_signal")
            if not r_corr > 0:
                reasons.append("nonpositive_corrected_denominator")
        valid = not reasons and np.isfinite(ratio_unfiltered)
        ratio = ratio_unfiltered if valid else np.nan
        raw_ratio = _positive_ratio(n_med, r_med) if sampling_valid else np.nan
        legacy_ratio = (_positive_ratio(n_med - legacy_bg, _stat(legacy_values) - legacy_bg)
                        if len(legacy_values) >= 20 else np.nan)
        row = dict(cell_id=f"{image_id}__cell_{label:05d}", image_id=image_id, label=label)
        metrics = dict(
            algorithm_version=ALGORITHM_VERSION,
            nuclear_median_raw=n_med, perinuclear_median_raw=r_med,
            nuclear_mean_raw=n_mean, perinuclear_mean_raw=r_mean,
            sampling_valid=bool(sampling_valid),
            raw_ratio_valid=bool(np.isfinite(raw_ratio)),
            raw_nuclear_perinuclear_ratio=raw_ratio,
            raw_log2_nuclear_perinuclear_ratio=float(np.log2(raw_ratio)) if np.isfinite(raw_ratio) else np.nan,
            raw_mean_nuclear_perinuclear_ratio=_positive_ratio(n_mean, r_mean) if sampling_valid else np.nan,
            background=bg, background_method=bg_info["background_method"],
            background_adequate=bg_info["background_adequate"], background_noise_scale_dn=noise,
            nuclear_median_bg_corrected=n_corr, perinuclear_median_bg_corrected=r_corr,
            nuclear_mean_bg_corrected=n_mean-bg, perinuclear_mean_bg_corrected=r_mean-bg,
            nuclear_perinuclear_ratio=ratio,
            log2_nuclear_perinuclear_ratio=float(np.log2(ratio)) if valid else np.nan,
            mean_nuclear_perinuclear_ratio=_positive_ratio(n_mean-bg, r_mean-bg) if valid else np.nan,
            unfiltered_narrow_ratio=ratio_unfiltered,
            unfiltered_wide_ratio=_positive_ratio(n_corr, _stat(wide_values)-bg),
            legacy_v1_ratio=legacy_ratio, legacy_v1_background=legacy_bg,
            legacy_v1_percentile_background_fallback=not legacy_has_bg,
            nuclear_snr=nuclear_snr, perinuclear_snr=ring_snr,
            ring_area_px=len(values), ring_to_nucleus_area=area_fraction,
            ring_signal_fraction=signal_fraction, ring_supported_sectors=supported_sectors,
            ring_near_image_boundary=clipped,
            nuclear_saturated_fraction=saturation_n, ring_saturated_fraction=saturation_r,
            ratio_valid=valid, ring_coverage_pass=bool(coverage),
            qc_reasons=";".join(reasons) if reasons else "corrected_ratio_available_not_quality_validated",
            qc_warnings=";".join(warnings) if warnings else "none",
            strict_gate_pass_diagnostic_only=not warnings,
            nuclear_detected_ring_unresolved=bool(nuclear_snr > config.min_snr and not ring_snr > config.min_snr),
            inner_gap_px=inner, outer_distance_px=outer, wide_outer_distance_px=wide_outer,
        )
        for snr in (2, 3, 5):
            metrics[f"sensitivity_snr_{snr}_pass"] = not reasons_at(snr)
        # Background sensitivity is NOT a statistical confidence interval.
        for sign, offset in (("minus", -noise), ("plus", noise)):
            metrics[f"sensitivity_background_{sign}_noise_ratio"] = _positive_ratio(n_med-bg-offset, r_med-bg-offset)
        row.update({f"posthoc_yap_{key}": value for key, value in metrics.items()})
        rows.append(row)
    info = dict(image_id=image_id, algorithm_version=ALGORITHM_VERSION,
                **bg_info, median_nuclear_diameter_px=diameter,
                inner_gap_px=inner, outer_distance_px=outer, wide_outer_distance_px=wide_outer,
                config_json=json.dumps(asdict(config), sort_keys=True),
                sampling_pixels_overlapping_dapi_nuclei=int(np.count_nonzero((rings > 0) & (mask > 0))),
                sampling_rule="all DAPI nuclei excluded; one nearest nuclear label per pixel; any nonempty ring measured",
                limitation="perinuclear proxy; no cell-body boundary, manual validation or noise/coverage rejection")
    return pd.DataFrame(rows), rings, info


def qc_summary(result: pd.DataFrame) -> pd.DataFrame:
    groups = [col for col in ("experimental_group_label", "seeding_density_cells_per_cm2", "magnification", "image_id") if col in result]
    summary = result.groupby(groups, dropna=False).agg(
        n_dapi_cells=("cell_id", "size"), n_ratio_technical_qc_pass=("posthoc_yap_ratio_valid", "sum"),
        n_sampling_available=("posthoc_yap_sampling_valid", "sum"),
        n_raw_ratio_available=("posthoc_yap_raw_ratio_valid", "sum"),
        median_raw_uncorrected_ratio=("posthoc_yap_raw_nuclear_perinuclear_ratio", "median"),
        median_ratio=("posthoc_yap_nuclear_perinuclear_ratio", "median"),
        median_nuclear_signal=("posthoc_yap_nuclear_median_bg_corrected", "median"),
        median_ring_signal=("posthoc_yap_perinuclear_median_bg_corrected", "median"),
        n_background_available=("posthoc_yap_background_adequate", "sum"),
        n_nuclear_detected_ring_unresolved=("posthoc_yap_nuclear_detected_ring_unresolved", "sum"),
        n_snr2_pass=("posthoc_yap_sensitivity_snr_2_pass", "sum"),
        n_snr3_pass=("posthoc_yap_sensitivity_snr_3_pass", "sum"),
        n_snr5_pass=("posthoc_yap_sensitivity_snr_5_pass", "sum"),
    ).reset_index()
    summary["ratio_technical_qc_pass_fraction"] = summary.n_ratio_technical_qc_pass / summary.n_dapi_cells
    # Keep the legacy column names for readers, but expose their new precise meaning.
    summary["n_corrected_ratio_available"] = summary.n_ratio_technical_qc_pass
    summary["sampling_available_fraction"] = summary.n_sampling_available / summary.n_dapi_cells
    summary["corrected_ratio_available_fraction"] = summary.ratio_technical_qc_pass_fraction
    return summary

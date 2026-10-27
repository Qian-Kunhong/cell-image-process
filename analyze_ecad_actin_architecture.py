from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import feature, filters, measure, morphology


WORKSPACE = Path(__file__).resolve().parent
DEPENDENCY_DIR = WORKSPACE / ".codex_deps" / "oiffile"
if str(DEPENDENCY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPENDENCY_DIR))

import oiffile  # noqa: E402


ROOT = Path(
    r"F:\Naruchit data\1_Data of published manuscripts\3_Hepatocyte differentiation"
    r"\03_Raw data\Immunostaining\Confocal - E-cad F-actin"
)
OUTPUT_DIR = WORKSPACE / "analysis_outputs" / "hepatocyte_confocal_audit" / "ecad_actin"


def ensure_zcyx(array: np.ndarray, axes: str) -> np.ndarray:
    if axes != "ZCYX":
        raise ValueError((axes, array.shape))
    if array.shape[0] <= 4 and array.shape[1] > 4:
        return np.transpose(array, (1, 0, 2, 3))
    return array


def focus_slice_index(array: np.ndarray) -> int:
    dapi = array[:, 0].astype(np.float32)
    score = np.mean(np.abs(np.diff(dapi, axis=1)), axis=(1, 2))
    score += np.mean(np.abs(np.diff(dapi, axis=2)), axis=(1, 2))
    return int(np.argmax(score))


def robust_normalize(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    positive = image[image > 0]
    lo, hi = np.percentile(positive, (1, 99.5))
    return np.clip((image - lo) / max(hi - lo, 1), 0, 1)


def tissue_mask(dapi: np.ndarray, ecad: np.ndarray, actin: np.ndarray) -> np.ndarray:
    combined = np.maximum.reduce(
        [filters.gaussian(dapi, 2), filters.gaussian(ecad, 2), filters.gaussian(actin, 2)]
    )
    threshold = filters.threshold_li(combined)
    mask = combined > threshold
    mask = morphology.binary_closing(mask, morphology.disk(5))
    mask = morphology.remove_small_objects(mask, min_size=500)
    return ndi.binary_fill_holes(mask)


def ecad_network_metrics(ecad: np.ndarray, mask: np.ndarray) -> tuple[dict, np.ndarray]:
    gradient = filters.sobel(filters.gaussian(ecad, 1.2))
    candidate = gradient[mask]
    threshold = np.percentile(candidate, 82) if candidate.size else 1
    edges = (gradient >= threshold) & mask
    edges = morphology.remove_small_objects(edges, min_size=8)
    skeleton = morphology.skeletonize(edges)
    labels = measure.label(skeleton)
    sizes = np.bincount(labels.ravel())[1:]
    total = max(int(skeleton.sum()), 1)
    long_pixels = int(sizes[sizes >= 20].sum()) if sizes.size else 0
    very_long_pixels = int(sizes[sizes >= 50].sum()) if sizes.size else 0
    return (
        {
            "ecad_skeleton_pixels_per_10000_tissue_px": total / max(mask.sum(), 1) * 10000,
            "ecad_continuity_ge20_fraction": long_pixels / total,
            "ecad_continuity_ge50_fraction": very_long_pixels / total,
            "ecad_component_median_length_px": float(np.median(sizes)) if sizes.size else 0,
        },
        skeleton,
    )


def actin_orientation_metrics(actin: np.ndarray, mask: np.ndarray) -> dict:
    image = filters.gaussian(actin, 1.2)
    gy, gx = np.gradient(image)
    magnitude = np.hypot(gx, gy)
    valid_values = magnitude[mask]
    if valid_values.size == 0:
        return {"actin_orientation_entropy": np.nan, "actin_alignment_index": np.nan}
    valid = mask & (magnitude >= np.percentile(valid_values, 70))
    angles = np.mod(np.arctan2(gy[valid], gx[valid]), np.pi)
    weights = magnitude[valid]
    hist, _ = np.histogram(angles, bins=18, range=(0, np.pi), weights=weights)
    probabilities = hist / max(hist.sum(), 1e-12)
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log(probabilities)) / np.log(18)
    tensor = feature.structure_tensor(image, sigma=2, order="rc")
    eigenvalues = feature.structure_tensor_eigenvalues(tensor)
    lambda_1, lambda_2 = eigenvalues[0], eigenvalues[1]
    coherence = (lambda_1 - lambda_2) / (lambda_1 + lambda_2 + 1e-12)
    return {
        "actin_orientation_entropy": float(entropy),
        "actin_alignment_index": float(1 - entropy),
        "actin_local_coherence_median": float(np.median(coherence[valid])),
        "actin_local_coherence_mean": float(np.mean(coherence[valid])),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    previews = []
    for path in sorted(ROOT.glob("*.oib"), key=lambda p: (int(p.stem[1]), int(p.stem[2]))):
        with oiffile.OifFile(path) as handle:
            stack = ensure_zcyx(handle.asarray(), handle.axes)
        z = focus_slice_index(stack)
        dapi, ecad, actin = [robust_normalize(stack[z, index]) for index in range(3)]
        mask = tissue_mask(dapi, ecad, actin)
        ecad_metrics, skeleton = ecad_network_metrics(ecad, mask)
        actin_metrics = actin_orientation_metrics(actin, mask)
        rows.append(
            {
                "sample": path.stem,
                "focus_z_index": z,
                "tissue_coverage_fraction": float(mask.mean()),
                **ecad_metrics,
                **actin_metrics,
            }
        )
        previews.append((path.stem, ecad, actin, skeleton))

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_DIR / "ecad_actin_architecture_metrics.csv", index=False)

    order = frame["sample"].tolist()
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    axes[0].plot(x, frame["tissue_coverage_fraction"], marker="o")
    axes[0].set_ylabel("Tissue coverage fraction")
    axes[0].set_title("Occupied field area")
    axes[1].plot(x, frame["ecad_continuity_ge50_fraction"], marker="o")
    axes[1].set_ylabel("Fraction of skeleton in ≥50 px components")
    axes[1].set_title("E-cadherin continuity proxy")
    axes[2].plot(x, frame["actin_local_coherence_median"], marker="o")
    axes[2].set_ylabel("Median local structure coherence")
    axes[2].set_title("F-actin local organization proxy")
    for ax in axes:
        ax.set_xticks(x, order, rotation=45)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#dddddd", linewidth=0.7)
    plot_path = OUTPUT_DIR / "ecad_actin_architecture_summary.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    selected = [item for item in previews if item[0] in {"D30D", "D33D", "D40D", "D43D", "D50D", "D53D"}]
    fig, axes = plt.subplots(len(selected), 3, figsize=(10, 3 * len(selected)), constrained_layout=True)
    for row, (sample, ecad, actin, skeleton) in enumerate(selected):
        axes[row, 0].imshow(ecad, cmap="gray")
        axes[row, 1].imshow(actin, cmap="gray")
        axes[row, 2].imshow(ecad, cmap="gray")
        axes[row, 2].imshow(np.ma.masked_where(~skeleton, skeleton), cmap="autumn", alpha=0.9)
        axes[row, 0].set_ylabel(sample)
        for col, title in enumerate(("E-cadherin", "F-actin", "E-cad edge skeleton")):
            if row == 0:
                axes[row, col].set_title(title)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    qc_path = OUTPUT_DIR / "ecad_actin_metric_qc.png"
    fig.savefig(qc_path, dpi=150)
    plt.close(fig)

    print(frame.to_string(index=False))
    print(f"\nSaved: {plot_path}\nSaved: {qc_path}")


if __name__ == "__main__":
    main()

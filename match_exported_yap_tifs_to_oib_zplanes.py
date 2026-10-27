from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


WORKSPACE = Path(__file__).resolve().parent
DEPENDENCY_DIR = WORKSPACE / ".codex_deps" / "oiffile"
if str(DEPENDENCY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPENDENCY_DIR))

import oiffile  # noqa: E402


IMMUNO = Path(
    r"F:\Naruchit data\1_Data of published manuscripts\3_Hepatocyte differentiation"
    r"\03_Raw data\Immunostaining"
)
OIB_ROOT = IMMUNO / "Confocal - YAP"
EXPORT_ROOT = IMMUNO / "YAP quantification"
OUTPUT_DIR = WORKSPACE / "analysis_outputs" / "hepatocyte_confocal_audit" / "yap_crowding"


def ensure_zcyx(array: np.ndarray, axes: str) -> np.ndarray:
    if axes != "ZCYX":
        raise ValueError((axes, array.shape))
    if array.shape[0] <= 4 and array.shape[1] > 4:
        return np.transpose(array, (1, 0, 2, 3))
    return array


def normalized(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    mean = image.mean()
    sd = image.std()
    if sd <= 0:
        return np.zeros_like(image)
    return (image - mean) / sd


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a * b))


def main() -> None:
    rows = []
    for sample_dir in sorted(p for p in EXPORT_ROOT.iterdir() if p.is_dir() and p.name.startswith("D")):
        sample = sample_dir.name
        with oiffile.OifFile(OIB_ROOT / f"{sample}.oib") as handle:
            stack = ensure_zcyx(handle.asarray(), handle.axes)
        normalized_raw = [
            [normalized(stack[z, channel]) for z in range(stack.shape[0])]
            for channel in range(2)
        ]
        for export_path in sorted(sample_dir.glob("*-C?.tif")):
            field = int(export_path.stem.split("-")[0])
            channel = int(export_path.stem[-1]) - 1
            export = normalized(tifffile.imread(export_path))
            scores = np.asarray([correlation(export, plane) for plane in normalized_raw[channel]])
            order = np.argsort(scores)[::-1]
            rows.append(
                {
                    "sample": sample,
                    "field": field,
                    "export_channel": channel + 1,
                    "best_z_index": int(order[0]),
                    "best_pearson_r": float(scores[order[0]]),
                    "second_best_z_index": int(order[1]),
                    "second_best_pearson_r": float(scores[order[1]]),
                    "best_minus_second": float(scores[order[0]] - scores[order[1]]),
                }
            )
    output = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_DIR / "exported_tif_to_raw_zplane_matches.csv", index=False)
    print(
        output.groupby("export_channel").agg(
            n=("best_pearson_r", "size"),
            median_best_r=("best_pearson_r", "median"),
            min_best_r=("best_pearson_r", "min"),
            max_best_r=("best_pearson_r", "max"),
        ).to_string()
    )
    print("\nHigh-confidence matches (r >= 0.95):", int((output["best_pearson_r"] >= 0.95).sum()), "/", len(output))
    print("\n", output.sort_values("best_pearson_r", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()

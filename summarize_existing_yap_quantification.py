from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parent
ROOT = Path(
    r"F:\Naruchit data\1_Data of published manuscripts\3_Hepatocyte differentiation"
    r"\03_Raw data\Immunostaining\YAP quantification"
)
OUTPUT_DIR = WORKSPACE / "analysis_outputs" / "hepatocyte_confocal_audit"


def sample_order(sample: str) -> tuple[int, int]:
    return int(sample[1]), int(sample[2])


def median(series: pd.Series) -> float:
    return float(series.median())


def main() -> None:
    frames = []
    for path in ROOT.rglob("*_Nuclei.csv"):
        sample = path.name.replace("_Nuclei.csv", "")
        frame = pd.read_csv(path)
        frame.insert(0, "sample", sample)
        frames.append(frame)

    cells = pd.concat(frames, ignore_index=True)
    ratio = "Math_NucByRingYAP"
    field_summary = (
        cells.groupby(["sample", "ImageNumber"], as_index=False)
        .agg(
            n_cells=(ratio, "size"),
            ratio_median=(ratio, "median"),
            ratio_mean=(ratio, "mean"),
            ratio_q10=(ratio, lambda x: x.quantile(0.10)),
            ratio_q90=(ratio, lambda x: x.quantile(0.90)),
        )
    )

    duplicate_note = pd.Series(False, index=cells.index)
    duplicate_note |= (cells["sample"] == "D43D") & (cells["ImageNumber"] == 4)
    cells["known_exact_duplicate_field"] = duplicate_note
    deduplicated = cells.loc[~cells["known_exact_duplicate_field"]].copy()

    sample_summary = (
        deduplicated.groupby("sample", as_index=False)
        .agg(
            n_cells=(ratio, "size"),
            n_fields=("ImageNumber", "nunique"),
            ratio_median=(ratio, "median"),
            ratio_mean=(ratio, "mean"),
            ratio_sd=(ratio, "std"),
            ratio_q10=(ratio, lambda x: x.quantile(0.10)),
            ratio_q90=(ratio, lambda x: x.quantile(0.90)),
        )
    )
    sample_summary["ratio_cv"] = sample_summary["ratio_sd"] / sample_summary["ratio_mean"]
    sample_summary = sample_summary.sort_values("sample", key=lambda s: s.map(sample_order))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cell_path = OUTPUT_DIR / "existing_yap_cell_measurements_with_duplicate_flag.csv"
    field_path = OUTPUT_DIR / "existing_yap_field_summary.csv"
    sample_path = OUTPUT_DIR / "existing_yap_sample_summary_deduplicated.csv"
    cells.to_csv(cell_path, index=False)
    field_summary.to_csv(field_path, index=False)
    sample_summary.to_csv(sample_path, index=False)

    order = sample_summary["sample"].tolist()
    xpos = {sample: idx for idx, sample in enumerate(order)}
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    for _, row in field_summary.iterrows():
        duplicate = row["sample"] == "D43D" and row["ImageNumber"] == 4
        ax.scatter(
            xpos[row["sample"]],
            row["ratio_median"],
            s=45,
            marker="x" if duplicate else "o",
            color="#c44e52" if duplicate else "#4c72b0",
            alpha=0.85,
            zorder=3,
        )
    ax.plot(
        np.arange(len(order)),
        sample_summary["ratio_median"],
        color="#222222",
        linewidth=1.4,
        marker="_",
        markersize=18,
        label="cell-level median (duplicate removed)",
    )
    ax.set_xticks(np.arange(len(order)), order)
    ax.set_ylabel("YAP nucleus / perinuclear-ring ratio")
    ax.set_xlabel("Dataset condition code")
    ax.set_title("Existing YAP quantification: field medians reveal sampling structure")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.01,
        0.02,
        "Each dot = one exported field; red × = D43D field 4, exact duplicate of field 1",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )
    plot_path = OUTPUT_DIR / "existing_yap_field_medians.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print(sample_summary.to_string(index=False))
    print(f"\nSaved: {sample_path}\nSaved: {field_path}\nSaved: {plot_path}")


if __name__ == "__main__":
    main()

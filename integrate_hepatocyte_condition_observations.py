from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


WORKSPACE = Path(__file__).resolve().parent
ROOT = WORKSPACE / "analysis_outputs" / "hepatocyte_confocal_audit"
INPUT_3D = ROOT / "dapi_3d" / "yap" / "analysis_summary" / "condition_3d_summary.csv"
INPUT_JUNCTION = ROOT / "ecad_actin" / "ecad_actin_architecture_metrics.csv"
OUTPUT_DIR = ROOT / "integrated_condition_analysis"
FOV_AREA_MM2 = (140.624 / 1000) ** 2

ORDER = ["D20D", "D30D", "D31D", "D33D", "D40D", "D41D", "D43D", "D50D", "D51D", "D53D"]
CONDITIONS = {
    "D20D": (2, -3, "shared_baseline"),
    "D30D": (3, -2, "no_HA"), "D31D": (3, -2, "single_HA"), "D33D": (3, -2, "repeated_HA"),
    "D40D": (4, -1, "no_HA"), "D41D": (4, -1, "single_HA"), "D43D": (4, -1, "repeated_HA"),
    "D50D": (5, 0, "no_HA"), "D51D": (5, 0, "single_HA"), "D53D": (5, 0, "repeated_HA"),
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nuclei = pd.read_csv(INPUT_3D)
    junction = pd.read_csv(INPUT_JUNCTION)
    merged = nuclei.merge(junction, on="sample", how="left", suffixes=("", "_junction"))
    merged["accepted_nuclei_per_mm2"] = merged["n_nuclei"] / FOV_AREA_MM2
    merged["sample"] = pd.Categorical(merged["sample"], ORDER, ordered=True)
    merged = merged.sort_values("sample")
    merged.to_csv(OUTPUT_DIR / "integrated_condition_summary.csv", index=False)

    metrics = {
        "n_nuclei": "核数量/视野",
        "volume_um3_median": "核体积",
        "max_xy_area_um2_median": "最大XY核面积",
        "height_um_median": "核高度",
        "yap_nuclear_shell_ratio_median": "YAP核/周核比",
        "tissue_coverage_fraction": "E-cad/F-actin组织覆盖",
        "ecad_continuity_ge50_fraction": "E-cad连续性",
        "actin_local_coherence_median": "F-actin局部一致性",
    }
    rows = []
    for day, samples in {3: ["D30D", "D31D", "D33D"], 4: ["D40D", "D41D", "D43D"], 5: ["D50D", "D51D", "D53D"]}.items():
        block = merged.set_index("sample").loc[samples]
        control = block.loc[samples[0]]
        for sample in samples:
            row = {"sample": sample, "culture_day": day, "paper_day": CONDITIONS[sample][1], "regimen": CONDITIONS[sample][2]}
            for metric in metrics:
                row[f"{metric}_relative_to_same_day_no_HA"] = block.loc[sample, metric] / control[metric]
            rows.append(row)
    relative = pd.DataFrame(rows)
    relative.to_csv(OUTPUT_DIR / "same_day_relative_observation_matrix.csv", index=False)

    treated = relative[relative["regimen"] != "no_HA"].copy()
    columns = [f"{metric}_relative_to_same_day_no_HA" for metric in metrics]
    matrix = treated[columns].to_numpy(float)
    fig, ax = plt.subplots(figsize=(12, 5.8), constrained_layout=True)
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=0.5, vmax=1.5, aspect="auto")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_xticks(np.arange(len(metrics)), list(metrics.values()), rotation=30, ha="right")
    ax.set_yticks(np.arange(len(treated)), treated["sample"])
    ax.set_title("同一天无HA视野归一化后的观察矩阵（1.00=同日无HA）")
    fig.colorbar(image, ax=ax, label="相对同日无HA")
    fig.savefig(OUTPUT_DIR / "same_day_relative_observation_heatmap.png", dpi=200)
    plt.close(fig)

    print(merged[[
        "sample", "n_nuclei", "accepted_nuclei_per_mm2", "volume_um3_median", "max_xy_area_um2_median",
        "yap_nuclear_shell_ratio_median", "tissue_coverage_fraction", "ecad_continuity_ge50_fraction",
        "actin_local_coherence_median",
    ]].round(3).to_string(index=False))
    print("\nRelative to same-day no-HA:")
    print(relative.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

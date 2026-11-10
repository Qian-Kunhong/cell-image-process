"""04 早期核表型：均匀性 + UMAP 描述性可视化。

输入：
- 旧仓 3D 逐核表：analysis_outputs/hepatocyte_confocal_audit/dapi_3d/yap/<sample>/nucleus_3d_metrics.csv
- 条件解码：同目录 analysis_summary/condition_3d_summary.csv

输出：
- analysis_outputs/04_uniformity_umap/<ts>/uniformity_umap.png
- per_condition_uniformity.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib


def main() -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import umap

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from rp.paths import legacy_path  # noqa: E402

    YAP_ROOT_3D = legacy_path("analysis_outputs", "hepatocyte_confocal_audit", "dapi_3d", "yap")
    COND_CSV = YAP_ROOT_3D / "analysis_summary" / "condition_3d_summary.csv"
    OUT_DIR = PROJECT_ROOT / "analysis_outputs" / "04_uniformity_umap"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. 条件解码 ---
    cond = pd.read_csv(COND_CSV, encoding="utf-8-sig")
    cond = cond[["sample", "culture_day", "paper_day", "regimen"]].copy()
    regimen_cn = {"shared_baseline": "基线", "no_HA": "不暴露", "single_HA": "单次", "repeated_HA": "每日"}
    cond["regimen_cn"] = cond["regimen"].map(regimen_cn)

    # --- 2. 逐核形态 ---
    FEATURES = [
        # 最小可解释集合：尺寸、轴向厚度、三维形状各一项。
        # 不同时放 voxel_count/volume/equivalent diameter，也不重复放 bbox/PCA 三轴。
        "volume_um3",
        "height_um",
        "pca_elongation_1_over_3",
    ]
    frames = []
    for sample in cond["sample"]:
        f = YAP_ROOT_3D / sample / "nucleus_3d_metrics.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, encoding="utf-8-sig")
        frames.append(df)
    morph = pd.concat(frames, ignore_index=True).merge(cond, on="sample", how="left")
    print(f"morph rows: {len(morph)}", file=sys.stderr)

    # --- 3. 均匀性：每条件的 CV（变异系数 = SD/mean） ---
    def cv(x):
        x = x.dropna()
        return x.std() / x.mean() if x.mean() != 0 else np.nan

    uniformity_rows = []
    for (paper_day, regimen_cn, sample), g in morph.groupby(["paper_day", "regimen_cn", "sample"]):
        uniformity_rows.append({
            "sample": sample,
            "paper_day": paper_day,
            "regimen": regimen_cn,
            "n": len(g),
            "diameter_mean": g["equivalent_sphere_diameter_um"].mean(),
            "diameter_sd": g["equivalent_sphere_diameter_um"].std(),
            "diameter_cv": cv(g["equivalent_sphere_diameter_um"]),
            "volume_mean": g["volume_um3"].mean(),
            "volume_sd": g["volume_um3"].std(),
            "volume_cv": cv(g["volume_um3"]),
        })
    uni = pd.DataFrame(uniformity_rows).sort_values(["paper_day", "regimen"])
    uni.to_csv(OUT_DIR / "per_condition_uniformity.csv", index=False, encoding="utf-8-sig")
    print(uni.to_string(), file=sys.stderr)

    # --- 4. UMAP（探索性可视化，不离散成未经验证的“表型簇”） ---
    # 缺测行丢弃，不做 nan→0
    morph = morph.dropna(subset=FEATURES).reset_index(drop=True)
    print(f"morph after dropna: {len(morph)}", file=sys.stderr)
    X = morph[FEATURES].to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)

    # UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, n_components=2, random_state=42)
    emb = reducer.fit_transform(Xs)
    morph["umap1"] = emb[:, 0]
    morph["umap2"] = emb[:, 1]
    morph.to_csv(OUT_DIR / "umap_coords_descriptive.csv", index=False, encoding="utf-8-sig")

    # --- 5. 画图：连续特征的描述性展示 ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: 均匀性 CV 柱状图
    ax = axes[0]
    colors_reg = {"基线": "#999999", "不暴露": "#4C72B0", "单次": "#DD8452", "每日": "#C44E52"}
    x_pos = 0
    xticks = []
    xlabels = []
    for _, row in uni.iterrows():
        ax.bar(x_pos, row["diameter_cv"] * 100, color=colors_reg[row["regimen"]], alpha=0.8)
        xticks.append(x_pos)
        xlabels.append(f"{row['regimen']}\nday{int(row['paper_day'])}")
        x_pos += 1
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("核直径 CV (%)")
    ax.set_title("A. 均匀性：同条件内核直径变异系数（越小越均匀）")
    ax.grid(axis="y", alpha=0.3)

    # Panel B: UMAP 按方案
    ax = axes[1]
    for reg in ["不暴露", "单次", "每日"]:
        sub = morph[morph["regimen_cn"] == reg]
        ax.scatter(sub["umap1"], sub["umap2"], s=8, alpha=0.6, label=reg, color=colors_reg[reg])
    ax.set_title("B. UMAP 投影（按 HA 方案）")
    ax.legend(fontsize=8)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

    # Panel C: UMAP 按 day
    ax = axes[2]
    day_colors = {-3: "#2166ac", -2: "#4393c3", -1: "#f4a582", 0: "#b2182b"}
    for day in [-3, -2, -1, 0]:
        sub = morph[morph["paper_day"] == day]
        ax.scatter(sub["umap1"], sub["umap2"], s=8, alpha=0.6, label=f"day {day}", color=day_colors[day])
    ax.set_title("C. UMAP 投影（按培养日）")
    ax.legend(fontsize=8)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

    fig.suptitle(
        f"扩增期早期核形态：均匀性与 UMAP 描述性展示（{len(morph)} 核，10 个单视野条件）",
        fontsize=12,
    )
    fig.tight_layout()
    png_path = OUT_DIR / "uniformity_umap.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    print(f"saved: {png_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

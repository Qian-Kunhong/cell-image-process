# Feeder-free Model A trial

复合特征的公式、方向、代理解释和限制见 `COMPOSITE_FEATURES.md`。每次运行会额外输出 `raw_vs_composite_model_comparison.csv/json`，并把 K 搜索扩展到 2–12，显式报告上限命中和低置信度比例。

This directory is intentionally isolated from the legacy continuous-OCT4 and
normal/deviated workflows.

## Day 2 trial scope

- The available DNA stain is **DAPI**, not Hoechst. The code records this
  explicitly and does not relabel DAPI as Hoechst.
- Day 2 contains one 5x, one 20x, and one 40x field. The input manifest records
  all three, but the user selected the 40x field as the only trial image.
- Only the 40x field is segmented, characterized, and passed to GMM/BASC. The
  5x and 20x fields do not enter any model calculation.
- The PNG files are 8-bit pseudo-colored exports without valid micrometre per
  pixel metadata. Features in this trial therefore use pixel units.
- Day 2 has no explicit replicate/sample identifier. These metadata fields are
  retained as `not_provided`, not inferred.

## Leakage boundary

1. DAPI morphology, DAPI intensity, and existing spatial feature families are
   preprocessed and passed to PCA/GMM/UMAP.
2. GMM fitting and UMAP finish before the OCT4 table is merged.
3. OCT4 green-channel mean intensity is used only by BASC A.
4. Only `BASC_OCT4_status` is merged into the final Model A result table.

The BASC threshold core is a Python port of CRAN `Binarize` 1.3.1 BASC A.
The bootstrap quality p-value is not required for binary status assignment and
is not calculated in this trial.

For this 112-cell trial, ordinary UMAP uses its exact small-data neighbor path,
random initialization, and 50 visualization epochs. Numba JIT is disabled only
for this UMAP call because import-time compilation in the Cellpose environment
is disproportionately slow. UMAP is visualization-only and does not affect the
GMM fit or K selection. Runs with 4096 or more cells intentionally stop and
require the full `pynndescent`-backed UMAP path instead.

## Run

Open PowerShell in this directory and use the existing Cellpose environment:

```powershell
& "C:\Users\dodos\miniforge3\envs\cellpose\python.exe" ".\day2_trial.py"
```

On a rerun, existing masks can be reused:

```powershell
& "C:\Users\dodos\miniforge3\envs\cellpose\python.exe" ".\day2_trial.py" --reuse-masks
```

Alternatively run `run_day2_40x.ps1` from PowerShell. Any extra arguments are
passed to `day2_trial.py`.

For Day 4 / Sample 2 / 40x, run:

```powershell
.\run_day4_sample2_40x.ps1
```

Its outputs are written to `outputs/day4_sample2_trial/`. `Sample 2` is stored
as sample metadata; replicate remains `not_provided` because the directory name
alone is not treated as evidence of a biological or technical replicate.

For Day 4 / Sample 1 / 40x, run:

```powershell
.\run_day4_sample1_40x.ps1
```

Its outputs are written to `outputs/day4_sample1_trial/`.

For the separate Day 4 / 20x feasibility trials, run:

```powershell
.\run_day4_sample1_20x.ps1
.\run_day4_sample2_20x.ps1
```

Their outputs are isolated under `outputs/day4_sample1_20x_trial/` and
`outputs/day4_sample2_20x_trial/`; they do not overwrite the 40x results.

Phenotype 编号只在同一倍率、同一整套拟合模型内有效；20× 的 `Phenotype 1` 不等于 40× 的同名编号。跨图片叠图使用按全局 phenotype rank 固定的 `tab20` 调色板，图例放在绘图区外。

The phenotype-membership comparison panels use one blue hue. Posterior
probability is encoded only by transparency: higher probability is more opaque,
and lower probability is more transparent.

Outputs are written under `outputs/day2_trial/` beside this code.

The main image-space validation figure is
`outputs/day2_trial/figures/phenotype_overlay_on_merge.png`. It overlays each
QC-kept nucleus with its dominant GMM phenotype color on the 40x Merge image;
white crosses flag cells whose maximum posterior probability is below 0.80.
The Merge/OCT4 pixels are display-only and never enter preprocessing, UMAP, or
GMM fitting.

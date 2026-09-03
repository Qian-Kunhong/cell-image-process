# YAP/DAPI Model A 1.0

共享 DAPI 复合特征的公式和边界见 `../Hochest_OCT4_1.0/COMPOSITE_FEATURES.md`。运行会同时比较原始模型和复合特征模型，输出 `raw_vs_composite_model_comparison.csv/json`；不会预设复合特征改善结果。

This workflow is intentionally separate from the OCT4/BASC workflow.

## Scientific boundary

- Only DAPI-derived nuclear morphology, DAPI intensity, and spatial features
  enter preprocessing, PCA, UMAP, or GMM.
- YAP/AF488 pixels are not opened until GMM fitting and UMAP are complete.
- YAP is retained as a continuous, background-corrected nuclear/perinuclear
  enrichment ratio. No YAP-positive/negative threshold is invented.
- The perinuclear ring is a proxy because these images have no membrane or
  cytoplasmic marker. It must not be described as a true whole-cell cytoplasm.
- YAP background is the median of a sufficiently large cell-free region. Dense
  fields without such a region use an explicitly flagged first-percentile
  estimate; no cross-image exposure correction is applied.
- Ratios require at least 20 non-overlapping perinuclear pixels. A separate
  high-coverage flag records whether the ring area is at least 20% of nuclear
  area, so dense-colony measurements remain available without hiding their
  lower spatial support.
- The supplied experimental design defines `Ctrl` as Control (No HA), `HA1` as
  HA-1 (72 h, 2.5 nM), and `HA2` as HA-2 (48 h, 5 nM).
- Folder codes `2.5`, `5`, `7.5`, and `10` denote seeding densities of
  2.5×10³, 5×10³, 7.5×10³, and 1×10⁴ cells/cm², respectively.
- All four seeding densities are compared separately inside each HA experimental
  group, yielding 12 distinct combinations per magnification. Each combination currently has
  one image field, so outputs are descriptive and do not use cell-level p-values
  as if cells were biological replicates.
- HA experimental-group and seeding-density metadata are post-hoc comparison
  variables only. They do not enter preprocessing, PCA, UMAP, or GMM.
- HA1 and HA2 differ in both HA concentration and exposure time. Their contrast
  cannot identify an isolated concentration effect or an isolated time effect.

## Inputs

- `*_DAPI_ORG.png`: grayscale DAPI input for segmentation and morphology.
- `*_AF488_ORG.png`: grayscale YAP/AF488 input for post-hoc localization.
- merged RGB PNG: display only.

The files are 8-bit PNG exports rather than microscope-native raw data.

## Run the 40x trial

```powershell
.\run_all_40x.ps1
```

On reruns, reuse the saved Cellpose masks:

```powershell
.\run_all_40x.ps1 --reuse-masks
```

## Run the independent 20x trial

```powershell
.\run_all_20x.ps1
```

On reruns:

```powershell
.\run_all_20x.ps1 --reuse-masks
```

若只验证 GMM、后验、表格和 YAP 后表征而不等待大样本 UMAP，可显式传入
`--skip-umap`；此时图中坐标为 PCA1/2 占位，`run_info.json` 会明确记录，不能称为 UMAP 结果。默认运行不跳过。

The 20x and 40x images are fitted separately; their pixel-scale morphology
features are never pooled into the same GMM.
相同编号的 phenotype 也不跨倍率对应。每个倍率整套模型采用固定的全局 phenotype 调色板，各图片缺少某一类时不会导致其余颜色漂移，图例位于绘图区之外。

Outputs are written to `outputs/all_40x_trial/` or `outputs/all_20x_trial/`.
Posterior probability columns
are dynamic and retain all selected GMM components. Image, HA group, exposure
time, HA concentration, seeding density, magnification, export-index, and
source-path metadata are retained for each cell.

The explicit mapping is also exported as `tables/experimental_design.csv`.

Key within-group figures are
`umap_posthoc_seeding_density_within_ha_group.png`,
`phenotype_composition_by_ha_group_and_seeding_density.png`, and
`yap_localization_by_ha_group_and_seeding_density.png`.

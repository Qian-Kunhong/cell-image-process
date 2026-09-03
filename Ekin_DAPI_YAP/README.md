# Ekin_DAPI_YAP

目录名只表示来源与通道。PyCharm 中直接运行 `run_baseline.py`（基线特征）或 `run_composite.py`（增加生物学相关复合特征）；公共实现位于 `../shared/dapi_model_a.py`，公式见 `../Ekin_DAPI_OCT4/COMPOSITE_FEATURES.md`。生物学相关量是形态与邻域代理，不是经过验证的生物学状态标签。两入口均保留原始/增强模型对照指标，但最终 phenotype、UMAP 和后表征采用入口指定的模型。

入口顶部 `MAGNIFICATION` 默认 `40x`，可改为 `20x`；也可设置运行参数 `--fit-magnification 20x`、`--data-root`、`--reuse-masks`。默认结果分别位于 `outputs/baseline/<倍率>/all_fields/` 和 `outputs/composite/<倍率>/all_fields/`；同配置重跑更新同一路径。不同数据集请指定独立 `--output-root`。`run_info.json` 记录最终 `feature_set`。

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

## 可直接比较组别差距的统计表

每次运行会自动生成：

- `field_descriptive_statistics.csv`：每个倍率 × HA 组 × 接种密度单视野的中位数、四分位数、均值和标准差；
- `same_density_group_contrasts.csv`：相同接种密度下 HA1−Ctrl、HA2−Ctrl、HA2−HA1 的中位数差和相对变化百分比；
- `phenotype_composition.csv`：各单视野 phenotype 细胞比例；
- `same_density_phenotype_composition_contrasts.csv`：相同密度下 phenotype 比例的组间百分点差。

这些是单视野描述统计，不提供 p 值，也不把单细胞当作生物学重复。HA1 与 HA2 的差异仍是“处理方案差异”，不能拆成单独时间效应或浓度效应。

## 表型图与边界 QC

表型图分成灰度 DAPI 定位图和不透明纯色 mask 图。右图颜色直接对应整套模型图例，数字为 phenotype 编号；左图黑描边白叉表示 `Pmax < 0.50`（等于 0.50 不标记），不是另一个表型。此显示阈值不改变 GMM 后验、分类或 QC；模型诊断中既有的 `fraction_max_posterior_below_0_8` 仍表示低于 0.8 的比例，不能与图中标记数量混用。

直接接触图像第一/最后一行或一列的分割核会被标记为 `touches_border`。旧规则漏掉了 Ctrl_5_40 图中标签 222/223 这两个离底边仅 1 像素的截断轮廓。现在默认 `--edge-buffer-px 2`，仅保留 bbox 距四边最小距离大于 2 像素的核，安全带内对象记录为 `within_2px_image_edge`。这是明确记录的保守边缘容差，不是经验证的完整性分类器，也不是凭肉眼估计“露出几分之几”。距离安全带更远的分割错误仍需另行 QC。

绘图会额外导出 `segmentation/masks_qc_keep/`，只保留最终结果中的核，标签 ID 不重排。原始 `segmentation/masks/` 不变，核周环测量仍使用完整分割来避免把已排除核误当成背景/可用核周区域。重新运行主流程时，QC 在邻域特征之前执行，截断核不再向完整核提供邻域形态值；因此需重拟合 GMM，旧 phenotype 编号不可与新结果直接等同。kNN 邻域在视野边缘仍有截断偏差，此修改不等于完全的空间边界校正。

只重画已有结果可运行 `replot_phenotypes.py --output-root <结果目录>`。默认不创建旧图备份；仅显式传入 `--backup-existing` 时，旧图才保存在 `figures/phenotype_overlays_before_display_fix/`。`tables/phenotype_display_audit.json` 记录显示阈值以及原始 mask 和单细胞结果的 SHA256 不变校验。

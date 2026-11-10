# 当前分析状态

更新时间：2026-11-10。

## 可以保留的口径

- 统计单位是视野；当前每条件只有一个 OIB。
- 逐核数据只用于描述单个视野内的分布，不能把核当作条件重复。
- 主流程使用连续形态特征（分位数、离散度、CV）和描述性 UMAP。
- UMAP 输入限于 `volume_um3`、`height_um`、`pca_elongation_1_over_3`，避免确定性重复变量反复加权核大小。

## 已停用的口径

- 不预设 K，不报告 P1/P2/P3、Cluster 1–K 或簇比例。
- 不把 BIC 在逐核技术子样本上的最优 K 当成生物学表型数。
- 分割 QC 中的实例颜色只用于区分相邻标签，没有生物学含义。

## 旧产物状态

`analysis_outputs/04_uniformity_umap/` 和 `analysis_outputs/05_replicate_ekin/` 中现有图片由旧代码生成，
其中仍可能含 GMM、三簇、簇比例或按簇上色。重新运行净化后的 04/05/05b/05c/08/09/10/12 前，
这些图片一律视为过期，不得用于报告或结论。

重新运行后必须再执行：

```powershell
python scripts\verify_count_ledger.py
python scripts\audit_pipeline.py
```


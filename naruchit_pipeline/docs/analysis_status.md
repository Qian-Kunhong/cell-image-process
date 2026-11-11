# 当前分析状态

更新时间：2026-11-11。

## 2026-11-11 完整复跑

- 数据资产盘点：1393 个文件、22 类。
- `cells.csv`：973 核 × 23 列。
- `population_features.csv`：10 个视野群体 × 200 列；未启用 cluster 字段。
- 掩膜标签与逐核表均为 973，逐样本全部一致。
- 旧汇总口径仍为 991，与有效掩膜相差 +18；该差异没有被覆盖或混用。
- 03–12 共 12 个连续形态、YAP 描述和分割 QC 脚本全部成功。
- 核数台账六项通过：1573 / 973 / 991 / 2821 / 868 / 1190。
- `audit_pipeline.py` 无 FAIL、无 WARN；Naruchit 11 项单测通过。

最新时间戳运行目录：

```text
analysis_outputs/02_inventory/20261111-172534_full_rerun_20261111/
analysis_outputs/00_cells/20261111-172534_full_rerun_20261111/
analysis_outputs/01_population/20261111-172535_full_rerun_20261111/
analysis_outputs/verify/20261111-172537/
```

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

2026-11-11 之前的 `analysis_outputs/04_uniformity_umap/` 和
`analysis_outputs/05_replicate_ekin/` 图片可能含旧 GMM/三簇口径，仍视为失效。
当前固定输出目录已由净化后的 03–12 脚本全部重新生成；实例颜色只用于分割 QC。

重新运行后必须再执行：

```powershell
python scripts\verify_count_ledger.py
python scripts\audit_pipeline.py
```


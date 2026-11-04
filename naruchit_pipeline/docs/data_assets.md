# 数据资产登记（Naruchit data）

数据根：`F:\Naruchit data\1_Data of published manuscripts\3_Hepatocyte differentiation`

规则：本表只记录“数据在哪、单位是什么、有多少、硬限制是什么”。**不记录分析结论**（结论在 `analysis_outputs/` 的每次运行里）。
每次新增数据集追加一行，不改历史行。

## 当前在用的数据

| 名称 | 相对数据根路径 | 单位 | 实有数量 | 用途 | 硬限制 |
|---|---|---:|---|---|---|
| 共聚焦 DAPI+YAP 三维原始 | `03_Raw data\Immunostaining\Confocal - YAP\*.oib` | 条件（样本） | 10 个 OIB，D20D–D53D | 3D 核形态、YAP 核/周核比 | 每条件**恰好 1 个视野**；PMT 不完全一致；无孔号 |
| 共聚焦 DAPI+E-cad+F-actin 三维原始 | `03_Raw data\Immunostaining\Confocal - E-cad F-actin\*.oib` | 条件 | 10 个 OIB | E-cad 连续性、F-actin 方向性 | 与 YAP 面板是**不同视野**，不能逐细胞对应 |
| D0 四剂量 DAPI（IN Cell 10x） | `03_Raw data\Immunostaining\IN Cell\...\D0\HA00/01/05/10_DAPI.tif` | 图像 | 4 张 | 剂量-形态组成 | 每剂量 1 张图，无重复 |
| 流式 ALB/AFP | `03_Raw data\Flow cytometry\Differentiation_AFP-FITC, ALB-PE\D17`、`\D21` | 孔 | 80 + 75 个 `.fcs` | 终点标签 y | 按 run×10 复孔组织，与图像无共同键 |
| 白蛋白 / 尿素 / 细胞密度 | `03_Raw data\Plate reader\...`、`22.07_...xlsx` | 孔 | 每 run×条件 10 复孔 | 终点标签 y 与协变量 | 同上 |

## 旧仓产物（只读引用）

旧仓：`C:\Users\dodos\PyCharmMiscProject\surrogate model\cell image`（**本模块只读它的产物，不改动仓内其他目录**）

| 产物 | 相对旧仓路径 | 单位 | 规模 |
|---|---|---:|---|
| 逐核 3D 指标表 | `analysis_outputs\hepatocyte_confocal_audit\dapi_3d\yap\<样本>\nucleus_3d_metrics.csv` | 细胞核 | 10 个表，**973 行** |
| 3D 掩膜 | `...\dapi_3d\yap\<样本>\dapi_3d_masks.npz` | — | 含 `labels` / `raw_labels` |
| 条件级汇总 | `...\dapi_3d\yap\analysis_summary\condition_3d_summary.csv` | 条件 | 10 行，`n_nuclei` 合计 **991** |

## 2026-11-04 核数核验结论（重要）

三个本该相等的数字对不上，逐样本比对结果：

| 样本 | 掩膜标签数 | 逐核表行数 | 旧汇总 n_nuclei | 判定 |
|---|---:|---:|---:|---|
| D20D | 8 | 8 | 8 | 一致 |
| D30D | 48 | 48 | 51 | 汇总偏大 3 |
| D31D | 46 | 46 | 48 | 汇总偏大 2 |
| D33D | 32 | 32 | 32 | 一致 |
| D40D | 146 | 146 | 146 | 一致 |
| D41D | 123 | 123 | 122 | 汇总偏小 1 |
| D43D | 69 | 69 | 69 | 一致 |
| D50D | 190 | 190 | 192 | 汇总偏大 2 |
| D51D | 197 | 197 | 196 | 汇总偏小 1 |
| D53D | 114 | 114 | 127 | 汇总偏大 13 |
| **合计** | **973** | **973** | **991** | 差 +18 |

**结论**：

1. **掩膜与逐核表完全自洽（10/10 一致，973 = 973）**，二者是同一版本，可作为当前细胞级事实来源。
2. **`condition_3d_summary.csv` 的 `n_nuclei` 与掩膜不同步**：6 个样本对不上，合计多 18 个核。但两套的核体积中位数在 7 个样本上逐位相同（D20D 1380.7892875271、D40D 561.1032792398、D33D 984.8081715769），说明**核集合高度重叠，差异来自计数规则（哪些核算“有效”），不是重新分割过**。
3. 因此“991”和“973”是**两个计数口径**，不能互相替代：引用 991 必须注明来自旧汇总表、且未逐核对齐；本项目统一用 **973**（掩膜口径）。
4. 由于核集合基本重叠，旧汇总里的体积中位数等统计量**仍可作为参考**，但凡是写进正文的数字，都应以本项目重算的 `population_features.csv` 为准，避免同一张表里两种口径混用。

复现命令：

```powershell
& 'C:\Users\dodos\miniforge3\envs\cellpose\python.exe' .\scripts\verify_nucleus_counts.py
```

## 还不存在的东西（不要当成已有）

- 早期图像与终点标签之间的**孔级配对键**：没有。图像侧只有条件代码（`D41D`），标签侧只有 `run × 复孔`，没有共同字段。
- 任何跨孔的生物学重复：没有。每条件 1 个视野，细胞核是技术子样本。
- `run_id`：当前数据完全没有，`config/project.json` 里保留该字段是给未来前瞻实验用的。

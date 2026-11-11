# Naruchit data — 早期核表型与干细胞群体分化品质

用培养早期的活细胞核表型预测群体最终分化品质的项目仓库。

| 项 | 值 |
|---|---|
| 数据根 | `F:\Naruchit data\1_Data of published manuscripts\3_Hepatocyte differentiation` |
| 所属仓库（本模块位于其中，不改动仓内其他目录） | `C:\Users\dodos\PyCharmMiscProject\surrogate model\cell image` |
| 解释器 | `C:\Users\dodos\miniforge3\envs\cellpose\python.exe`（Py 3.10，已装 cellpose；**无 openpyxl**） |
| 统计单位 | 视野（`field`）。每条件仅 1 个 OIB，细胞核是技术子样本 |

所有路径都从 `config/project.json` 读，代码里不写盘符。

## 当前状态（2026-11-11 完整复跑）

```
旧仓 per-sample 逐核表 (10 个 CSV)
        │  scripts/00_collect_legacy_nuclei.py
        ▼
   cells.csv  973 核 × 23 列
        │  scripts/01_build_population_features.py
        ▼
population_features.csv  10 群体 × 200 列
```

最新产物位置：

- `analysis_outputs\02_inventory\20261111-172534_full_rerun_20261111\`
- `analysis_outputs\00_cells\20261111-172534_full_rerun_20261111\`
- `analysis_outputs\01_population\20261111-172535_full_rerun_20261111\`
- `analysis_outputs\verify\20261111-172537\`
- `analysis_outputs\03_early_phenotype\`
- `analysis_outputs\04_uniformity_umap\`
- `analysis_outputs\05_replicate_ekin\`

本次 00–12 全流程均成功；计数台账、科学口径审计和 11 项单测全部通过。

每个运行目录内都有 `run_info.json`（脚本名、参数、解释器、时间），可直接复现。

## 目录

| 目录 | 内容 |
|---|---|
| `config/project.json` | 数据根、旧仓路径、统计单位、成像物理尺度 |
| `data/README.md` | 数据在哪的说明（**不存数据**） |
| `docs/data_assets.md` | 数据资产登记 + 核数核验结论（先读这个） |
| `docs/feature_dictionary.md` | 特征定义、单位、共线性、是否进模型 |
| `docs/analysis_status.md` | 当前可用/停用口径与旧产物失效状态 |
| `docs/metadata_schema.md` | 前瞻实验的孔级主表字段（未来配对用） |
| `src/rp/paths.py` | 路径解析、运行目录、`run_info.json` |
| `src/rp/population.py` | 细胞级 → 群体级聚合（均值/分位数/离散度/亚群熵） |
| `scripts/00_...` | 收拢旧仓逐核表为 `cells.csv` |
| `scripts/01_...` | 聚合到群体级特征表 |
| `scripts/02_inventory_data_assets.py` | 盘点数据资产：文件级清单 + 单位级计数（不改数据） |
| `scripts/03_...`–`05_...` | 连续核形态总览、均匀性 / UMAP 与描述性效应量复算；不生成未经验证的离散表型 |
| `scripts/06_...`–`12_...` | 最终森林图、YAP 箱线图、分割叠图与条件网格图 |
| `scripts/verify_count_ledger.py` | 核数台账自动核验（1573 / 973 / 991 / 2821 / 868） |
| `scripts/verify_nucleus_counts.py` | 掩膜 / 逐核表 / 旧汇总三方核数校验 |
| `analysis_outputs/` | 00/01/02/verify 使用新时间戳目录；03–12 更新各自固定的派生图目录 |
| `tests/` | 11 项单测，只测纯函数与路径 |

## 命令

```powershell
$py = "C:\Users\dodos\miniforge3\envs\cellpose\python.exe"
Set-Location 'C:\Users\dodos\PyCharmMiscProject\surrogate model\cell image\naruchit_pipeline'

# 0. 先盘点数据资产（1393 个文件 / 22 类，产物在 analysis_outputs/02_inventory/<时间戳>）
& $py .\scripts\02_inventory_data_assets.py

# 1. 收拢旧仓逐核表（只读旧仓，不修改）
& $py .\scripts\00_collect_legacy_nuclei.py

# 2. 聚合到群体级（--input 指上一步的 cells.csv）
& $py .\scripts\01_build_population_features.py `
      --input .\analysis_outputs\00_cells\<时间戳_legacy>\cells.csv `
      --group-col sample_code --tag legacy

# 3. 核数一致性校验（发现口径漂移时先跑这个）
& $py .\scripts\verify_nucleus_counts.py

# 4. 测试
& $py -m unittest discover -s tests -v
```

## 三条硬边界（写论文前先看）

1. **统计单位是视野，不是核。** 每条件只有 1 个 OIB，973 个核是技术子样本。任何组间差异只能写“单视野描述性”，不能给 p 值。
2. **早期图像与终点标签之间没有配对键。** 图像侧只有 `D41D` 这类条件代码，标签侧是 `WO1/HA1 × 10 复孔`，二者无共同字段。因此**禁止报告任何预测性能**，直到前瞻配对实验（`docs/metadata_schema.md`）拿到数据。
3. **计数只用一套口径。** 当前掩膜口径 **973 核**；旧汇总表写的是 991 核，两者差 18，来源见 `docs/data_assets.md`。同一张表里不得混用。

## 目前能做到哪

| 能做 | 不能做 |
|---|---|
| 10 个条件的核形态群体表示（均值/分位数/离散度） | 预测分化品质（无配对数据） |
| 同日 HA 方案的条件级描述性对比 | 对任何组间差异做显著性检验 |
| 核数/口径一致性自动校验 | 跨倍率、跨面板特征比较 |
| 把旧仓资产接进可复现的流水线 | 引用旧文档的 991 核作为基数 |
| 剂量级（4 档）早期↔后期对齐：IN Cell D0/D3 ↔ D17/D21，RT-PCR D3 ↔ D17/D21（前提是 IN Cell 的 DAPI 分割可用） | 孔级/视野级纵向追踪：免疫染色需固定，不同天数必然换孔；唯一同孔连续的是明场时程，无核通道 |

## 下一步（对齐交接文档 §8）

| 任务 | 本仓库落点 | 前提 |
|---|---|---|
| T5 群体表示模块 | 已有 `src/rp/population.py`，待补直方图/推土距离/Voronoi | 无 |
| T2 特征字典 | `docs/feature_dictionary.md` 已起草，待按真实列名补全 | 无 |
| T3 照明与批次归一 | 新建 `scripts/03_normalize_illumination.py` | T2 |
| T6 基线模型 | 新建 `scripts/04_baseline_models.py` | **必须等配对数据** |

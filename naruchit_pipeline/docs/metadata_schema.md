# 元数据主表规范

一行 = 一个孔 × 一个时间点。**这是整个项目里唯一允许被 join 的表。**

图像文件名只编码条件与时间（如 `D41D`），标签按 `run × 复孔` 组织（如 `HA1 × B03`）——
两套编号之间没有共同字段，这就是预测类课题卡死的根因。所以主表必须在取数据之前就定好，
而不是事后把表拼起来。

## 字段

| 字段 | 类型 | 例 | 必填 | 说明 |
|---|---|---|---|---|
| `experiment_id` | str | EXP01 | Y | 一次完整实验 |
| `run_id` | str | R1 | Y | 独立生物学重复，划分 train/test 的最小单位 |
| `batch_id` | str | B1 | Y | 试剂/细胞批次，用于批次效应溯源 |
| `plate_id` | str | P1 | Y | 板 |
| `well_id` | str | B03 | Y | 配对键的核心，跨表 join 只用它 |
| `sample_code` | str | D41D | | 沿用历史编号，便于与旧图像对话 |
| `culture_day` | int | 4 | Y | 接种后第几天 |
| `paper_day` | int | -1 | | 论文口径的天数，与上一列并存防混淆 |
| `ha_regimen` | str | no_HA | Y | 方案，不是“已处理次数” |
| `ha_dose_nM` | float | 10 | | |
| `ha_lot` | str | L2301 | | 批次效应 |
| `seeding_density` | float | 15.5 | Y | 混杂主因，必须记 |
| `image_id` | str | IMG0042 | | 可回溯到文件 |
| `field_id` | str | F2 | | 视野编号，每孔至少 3 个 |
| `z_plane` | int | 12 | | |
| `hoechst_dose` | float | 0.5 | | 成像条件 |
| `exposure_ms` | float | 20 | | 成像条件 |
| `pmt` | float | 300 | | PMT 不一致会直接毁掉强度特征 |
| `endpoint_day` | int | 17 | | |
| `fcm_ALBpAFPp` | float | 0.41 | | y：四象限比例 |
| `albumin_ng_ml_24h` | float | 211.9 | | y：分泌功能 |
| `urea` | float | | | y：次级功能 |
| `cell_density` | float | | | y 或协变量 |
| `protocol_version` | str | v2 | Y | 协议一改就加版本，只增不改 |

## 落盘

`analysis_outputs/paired/metadata_master.csv`，UTF-8-SIG 编码，只增不改。

## 自检（数据到手先跑这几条）

```python
import pandas as pd
m = pd.read_csv(r"...\metadata_master.csv", encoding="utf-8-sig")

# 1. 每个 run x 方案 x 时间点至少 3 孔
print(m.groupby(["run_id", "ha_regimen", "culture_day"]).size().min())

# 2. well_id 不得跨 run 复用（复用就是跨孔污染）
print(m.groupby("well_id")["run_id"].nunique().max(), "应为 1")

# 3. 配对键唯一
print(m.duplicated(subset=["well_id", "culture_day"]).sum(), "应为 0")
```

三条任一不满足，都不要开始建模，先回去补数据。

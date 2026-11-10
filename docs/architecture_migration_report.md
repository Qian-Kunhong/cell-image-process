# 架构迁移与验证报告

更新时间：2026-11-10。

## 结论

第一条公共计算链已经实际接入，不再只是目录规划：

```text
图像读取/通道转换
→ Cellpose 输入与标签后处理
→ 图像边缘 QC
→ 2D 核形态
→ 自适应邻域/复合描述量
→ 群体级汇总与运行记录
```

公共实现位于 `src/cell_image_core/`。项目目录只提供数据发现、实验元数据、
特征清单、模型选择、标记通道解释和展示文字。

## 已迁移

| 能力 | 唯一实现 | 已接入项目 |
|---|---|---|
| 灰度/RGB 通道读取、2D/3 通道转换、百分位归一化 | `cell_image_core.io` | Ekin OCT4、Ekin YAP、Suzui DAPI |
| Cellpose 模型加载、标签接触分离、四邻域边界 | `cell_image_core.segmentation` | Ekin OCT4、Ekin YAP、Suzui DAPI |
| 边缘距离与保留原因 | `cell_image_core.qc` | Ekin YAP |
| 2D 核形态、DAPI 强度、邻域、复合描述量 | `cell_image_core.features` | Ekin OCT4、Ekin YAP；Suzui 已接入公共基础公式 |
| 群体级连续汇总 | `cell_image_core.population` | Naruchit（旧参数名由薄包装兼容） |
| 独立运行目录与 `run_info.json` | `cell_image_core.provenance` | Naruchit |
| 数据对象约束 | `cell_image_core.contracts` | 新代码可直接使用；旧入口逐步适配 |

`shared/dapi_model_a.py` 现在是 Ekin OCT4 的兼容/模型层，不再是新的公共
基础函数归属地。Ekin YAP 的项目元数据和标记解释仍保留在项目包中。

## 验证结果

### 合成黄金数组

- 公共内核合成数组、参数透传与项目适配器测试：9/9 通过。
- 根目录架构与入口测试：20/20 通过。
- Ekin YAP 回归测试：51/51 通过。
- Naruchit 路径与群体汇总测试：11/11 通过。
- Ekin OCT4 回归测试：6/6 通过。
- 四组共 88 项测试全部通过；全仓 652 个 Python 文件通过 AST 语法解析。

### 真实视野迁移比较

使用现有 Ekin YAP 40× 图像与既有 mask，对 12 个视野分别运行迁移前实现
和新内核适配器。12/12 视野的行数、列集合和全部公共字段一致，失败字段为 0。
验证完成后，迁移专用的 `_legacy_*` 参考函数已从生产模块删除，避免形成第二实现源。

机器可读结果位于（该目录被 Git 忽略）：

```text
validation_runs/core_migration/summary.json
validation_runs/core_migration/field_comparisons.csv
```

### 隔离全流程工程回归

复用既有 mask，在 `validation_runs/ekin_yap_full_regression_20261110/` 完成了
一次独立 40× 全流程运行，没有覆盖正式结果：

- 12 个视野；
- 3075 个分割对象；
- 2481 个边缘 QC 后对象；
- 2441 个可采样原始 YAP 比值；
- 1127 个背景校正后数学上可用的 YAP 比值；
- 表格和图均成功写出，总耗时约 104 秒。

这只是工程连通性验证，不构成新的科学结论。该次现有模型动态选择了 `K=7`；
这不是预设 K，也不表示七种已验证细胞类型。该运行同时显示增强特征模型的
平均最大后验和种子稳定性低于原始特征模型，因此不能因为流程跑通就把增强
模型写成更科学或更可靠。模型层继续保留为项目级实验实现，不提升为公共默认。

### Naruchit 派生图与入口审计

净化后的 12 个出图脚本均已成功执行，其中审计跟踪的 8 个主要派生图已重新
生成；计数账本的 6 个检查项全部一致。这 12 个旧式脚本也已统一加入 `main()`
守卫，导入模块不再自动读数据、建目录或覆盖结果。`audit_pipeline.py` 最终没有
FAIL 或 WARN。

## 有意没有合并

- 不统一 Ekin、Suzui、Naruchit 的 K、模型、特征清单或生物学解释。
- 不把 YAP、OCT4、处理组、时间、浓度或密度送入 DAPI 形态模型。
- 不把 2D/3D、20×/40×、BASC/连续强度当成同一种口径。
- 不移动或删除历史结果；验证输出放入独立且被忽略的目录。
- `Ekin_DAPI_YAP - Doubao/` 暂按历史快照保留，因为其输出仍被
  `writing_assets/` 清单引用；它不是正式运行入口。

## 剩余边界

`shared/dapi_model_a.py`、Ekin YAP 的主流程和 Suzui 的聚类/推理脚本仍然较长。
其中包含项目科学选择，不能机械拆成一个“万能模型”。后续拆分只能按以下接口：

1. 项目显式提交特征 manifest；
2. 拟合函数只返回模型 artifact、坐标和诊断；
3. 绘图只读取保存的结果，不重新拟合；
4. 任何离散类别必须标明算法来源、训练数据、倍率和验证状态。


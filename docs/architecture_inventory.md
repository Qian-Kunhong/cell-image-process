# 代码架构资产清单

更新时间：2026-11-10。

本文件只盘点代码归属与重复实现，不改变任何分析算法、数据或输出。

迁移执行情况和真实数据回归结果见 `architecture_migration_report.md`。

## 1. 结论

当前仓库是“项目目录 + 少量共享代码 + 多套复制流水线”，还不是“多个项目调用同一套计算内核”。

主要问题：

1. 初次盘点时，`Ekin_DAPI_YAP/` 与 `Ekin_DAPI_YAP - Doubao/` 有 22 个同路径 Python 文件，其中 20 个逐字节相同；公共内核迁移后为 19 个相同、3 个分叉。
2. `shared/dapi_model_a.py` 同时承担 IO、分割、特征、预处理、GMM、UMAP、绘图和完整入口，共约 1433 行。
3. 初次盘点时 Ekin YAP 的 `model_a_pipeline.py` 约 1042 行，也同时包含计算与呈现；正式目录现已接入公共内核，历史快照不再与其相同。
4. Suzui 通过 `runpy`、`spec_from_file_location` 和修改被导入模块的全局变量复用脚本，接口不稳定。
5. Naruchit 的 `src/rp/paths.py`、`src/rp/population.py` 已接近纯计算模块，但 03–12 仍是导入即执行的脚本。
6. 仓库没有统一 `pyproject.toml`；大量入口依赖 `sys.path.insert()`。
7. 输出同时使用 `outputs/`、`analysis_outputs/`、项目内部目录和固定覆盖路径。

## 2. 当前项目与职责

| 目录 | 当前职责 | 可保留的项目特有部分 | 主要架构问题 |
|---|---|---|---|
| `shared/` | Ekin OCT4 公共实现、入口参数 | 暂无；目标应是通用内核 | `dapi_model_a.py` 是 1433 行单体，且默认输出指向具体项目 |
| `Ekin_DAPI_OCT4/` | DAPI 形态模型 + OCT4 后表征 | OCT4/BASC 规则、样本清单 | 通过 `import *` 调用共享单体；配置仍嵌在入口 |
| `Ekin_DAPI_YAP/` | 40× DAPI 形态 + YAP 后表征 | YAP 核/核周测量、实验组元数据 | 1042 行主流程混合分割、训练、汇总和绘图 |
| `Ekin_DAPI_YAP - Doubao/` | Ekin YAP 历史/副本 | 仅用于旧资产追溯，不再开发 | 19 个 Python 文件仍与正式目录重复；3 个已分叉 |
| `Suzui/DAPI/` | 分割、特征、QC、聚类、推理、SA/SHAP | Suzui 数据布局、历史模型兼容 | 多个 1000–2200 行脚本，配置与算法绑定 |
| `Suzui/DAPI_OCT4/` | 配对 DAPI/OCT4 流程 | 双通道配对与 OCT4 对照 | 靠动态导入和修改 `Suzui/DAPI` 模块全局量复用 |
| `naruchit_pipeline/` | 视野级连续形态描述与数据盘点 | Naruchit 条件解码、统计边界 | `rp` 内核较干净，但多数绘图脚本仍有顶层副作用 |
| 根目录分析脚本 | 肝细胞 2D/3D/YAP 辅助分析 | 3D OIB 与特定历史资产适配 | 路径、读取、计算、绘图分散，未注册为正式项目 |

## 3. 当前能力实现矩阵

| 能力 | 当前实现 | 建议唯一真源 | 项目适配器保留内容 |
|---|---|---|---|
| 图像/通道读取 | `shared/dapi_model_a.py`、Ekin YAP、Suzui 各自实现 | `cell_image_core.io` | 文件命名、通道映射、数据根 |
| Cellpose 分割 | shared、Ekin YAP、Suzui DAPI、Suzui 双染各一套 | `cell_image_core.segmentation` | 模型名、直径、通道、边缘策略配置 |
| 2D 形态特征 | shared、Ekin YAP、Suzui 多套实现 | `cell_image_core.features.morphology_2d` | 项目允许的特征集合 |
| 3D 形态特征 | 根目录/Naruchit 历史分析 | `cell_image_core.features.morphology_3d` | OIB 轴顺序、物理尺度 |
| 邻域特征 | shared、Ekin YAP、Suzui 重复实现 | `cell_image_core.features.neighborhood` | 邻域半径及是否启用 |
| QC/边缘核 | shared、Ekin YAP、Suzui 各自实现 | `cell_image_core.qc` | 项目阈值与允许规则 |
| 群体级汇总 | `naruchit_pipeline/src/rp/population.py` | `cell_image_core.population` | 统计单位与分组键 |
| OCT4 后表征 | shared/Ekin OCT4、Suzui 双染 | 通用采样函数 + 项目 OCT4 适配器 | BASC 或连续强度规则不能混用 |
| YAP 核/核周比 | `Ekin_DAPI_YAP/ekin_dapi_yap/yap_ratio_qc.py` | `cell_image_core.markers.yap` | YAP 字段与实验组映射 |
| 预处理/特征选择 | shared、Ekin YAP、Suzui 各自实现 | `cell_image_core.preprocessing` | 项目批准的 feature manifest |
| 模型拟合 | shared、Ekin YAP、Suzui 各自实现 | 先定义 artifact 接口，再决定可复用算法 | 不能强制不同项目共用 K、特征或模型 |
| UMAP/PCA | shared、Ekin YAP、Suzui、Naruchit | `cell_image_core.representation` | 只保存坐标与参数；展示层不得重拟合 |
| 绘图 | 混在各流水线，另有多份 display/PPT 文件 | `cell_image_core.visualization` | 项目文字、颜色规范、版式 |
| 运行记录 | Naruchit `rp.paths`、Ekin `run_info.json`、Suzui 各自写 | `cell_image_core.provenance` | 项目名、数据清单和科学限制 |

## 4. 唯一真源候选

这些是迁移起点，不表示原文件可以原样搬进公共包。

| 能力 | 候选来源 | 原因 | 迁移前必须处理 |
|---|---|---|---|
| 运行目录与 `run_info` | `naruchit_pipeline/src/rp/paths.py` | 纯函数比例最高，已有测试 | 去掉 Naruchit 专用配置键 |
| 群体级连续汇总 | `naruchit_pipeline/src/rp/population.py` | 明确区分群体与技术子样本 | 把可选 cluster 字段标成外部验证后才能启用 |
| YAP 测量 | `Ekin_DAPI_YAP/ekin_dapi_yap/yap_ratio_qc.py` | 有独立配置和较完整测试 | 与文件读取、绘图完全分离 |
| YAP 显示 | `ekin_dapi_yap/yap_ratio_display.py` | 已接近单用途模块 | 只接收标准结果对象，不接项目路径 |
| DAPI 形态基础公式 | `shared/dapi_model_a.py` 与 Ekin YAP 的交集 | 两项目均使用，已有部分测试 | 从单体中逐函数抽取并做黄金样本比较 |
| Suzui 双通道配对 | `Suzui/DAPI_OCT4/double_work_utils.py` | 领域特有，不应塞进通用分割层 | 保留为 Suzui adapter，调用公共 IO/特征内核 |

## 5. 不应错误合并的内容

- 2D 与 3D 形态特征不能因为列名相似就放进同一模型。
- 20× 与 40× 模型不能共享拟合参数或表型编号。
- Ekin OCT4 的 BASC 二元规则不能用于 Suzui 的连续 OCT4 对照。
- YAP、OCT4 与实验组元数据只能后表征，不能混入 DAPI 形态模型。
- Naruchit 当前没有可验证的离散表型，也没有早期图像到终点标签的配对键。
- “共用内核”指共用确定的计算定义，不代表共用同一训练数据、K、模型或科学解释。

## 6. 目标模块边界

```text
src/cell_image_core/
├── io/                 # 读取、通道与样本清单
├── segmentation/       # Cellpose 与标签后处理
├── features/           # 2D/3D 形态、邻域、强度采样
├── qc/                 # 质量控制，不负责绘图
├── population/         # 视野/孔级聚合
├── preprocessing/      # 缺失值、缩放、特征 manifest
├── representation/     # PCA/UMAP，只返回坐标与参数
├── models/             # fit/transform/predict 与 artifact
├── results/            # 标准表结构
├── visualization/      # 只消费结果，不训练模型
└── provenance/         # run_info、版本、输入清单

projects/
├── ekin_oct4/          # 配置、样本清单、BASC adapter、入口
├── ekin_yap/           # 配置、YAP adapter、入口
├── suzui/              # 数据布局与历史兼容 adapter
└── naruchit/           # 条件解码、统计边界、入口
```

## 7. 第一条迁移切片

先迁移不涉及聚类或生物学标签的公共链路：

```text
DAPI 读取
→ Cellpose 分割
→ 边缘核过滤
→ 2D 形态特征
→ QC 表和 QC 叠图
```

第一阶段只抽函数，不改变参数与结果。建议以一个 Ekin YAP 视野和一个 Suzui 视野作为黄金样本。

验收条件：

1. 新旧 mask 的形状、标签数与前景像素完全一致，或差异有明确原因。
2. 面积、周长、圆度、质心等公共字段逐列比较。
3. 项目入口只传 `SampleRecord` 和配置，不再复制计算函数。
4. 新内核没有项目路径常量，也不在 import 时创建目录。
5. 旧入口暂时保留为 wrapper，结果一致后再删除重复实现。

## 8. 执行顺序

1. 把当前科学净化改动单独形成检查点，避免与架构迁移混合。
2. 新建 `pyproject.toml` 与 `src/cell_image_core/` 空骨架。
3. 定义 `SampleRecord`、`SegmentationResult`、`FeatureTable`、`RunArtifact`。
4. 为第一条迁移切片建立黄金样本/合成数组测试。
5. 从 shared 和 Ekin YAP 抽取读取、分割、2D 特征与 QC。
6. 用 wrapper 接回 Ekin OCT4、Ekin YAP，再接 Suzui。
7. 结果一致后隔离 `Ekin_DAPI_YAP - Doubao/` 的重复文件。
8. 最后才拆模型、UMAP 和绘图，避免同时改变科学结果与文件结构。

## 9. 当前禁止动作

- 不直接移动或删除项目目录。
- 不在抽公共函数时顺便改特征、阈值、K 或模型选择。
- 不把显示函数当训练内核，也不让绘图脚本重新拟合。
- 不用“文件名相同”判断可合并，必须比较输入、单位、过滤规则和输出模式。


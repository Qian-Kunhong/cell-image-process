# Suzui 数据分析

原根目录的分析脚本与原 `double_work_build` 都以 `F:\Suzui` 为默认数据根目录，现统一归入本目录。来源相同不等于图像批次或分析配置相同：DAPI 流程包含训练数据、单通道/时序图像；配对染色流程默认分析 `paper_Oct-4\Tic-SNL,Rac1,Oct-4x`。

## 目录与运行

- `DAPI/`：选焦 → Cellpose 分割 → 特征提取/QC → 聚类 → XGBoost 推理；保留自适应聚类、相对批次归一化、SA、SHAP、GIF 脚本。
- `DAPI_OCT4/`：DAPI 分割并复用同一掩膜测量配对 OCT4；特征提取、QC、聚类及推理复用 `DAPI/` 的实现，通过 `double_work_utils.py` 定位，不复制核心算法。

在 PyCharm 中运行相应子目录里的原编号脚本，文件名与运行顺序不变。先核对脚本顶部的 `mode`、`DATASET_NAME`、`TRAINING_SET_NAME` 和路径。已有运行配置如仍指向根目录旧路径，应改为 `Suzui/DAPI/<脚本名>`。

## 两条流程的边界

配对染色流程将 DAPI 形态/邻域特征与 OCT4 强度表分开；OCT4 用于强度对照，DAPI 灰度强度不进入该双染流程的聚类/推理。这不是 Ekin 的 BASC 二元表征流程，不能互换标记解释。常规 DAPI 流程继续保持现有配置规则，不因目录整合自动变成双染模式。

默认外部数据、训练集和 `analysis_out` 路径保持不变，`masks_double`、`features_double` 等双染输出命名也不变。本次只整合源码归属与依赖，不合并数据表、不重训模型、不重算图像、不删除已有数据。

详见 [双染运行说明](DAPI_OCT4/README_double_work.md)。Ekin 的两个 Model A 流程及其 `baseline`/`composite` 入口仍位于仓库根目录的 `Ekin_DAPI_OCT4/` 和 `Ekin_DAPI_YAP/`。

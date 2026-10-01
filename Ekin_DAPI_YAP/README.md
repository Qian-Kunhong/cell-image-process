# Ekin DAPI–YAP 40× analysis

本项目现在只保留 **40×** 流程。10×/20×结果、入口和说明均已移除；原始显微图像不受影响。

## 运行入口

在 PyCharm 中直接运行：

```text
run_40x_analysis.py
```

这是唯一的完整分析入口。它会：

1. 用 DAPI 分割细胞核并提取单细胞核形态与邻域特征；
2. 进行预处理、PCA、动态 K 搜索、GMM 和仅用于展示的 UMAP；
3. 保存全部 GMM 后验概率和主导表型；
4. 聚类完成后才测量 YAP 核内/核周比值。

默认结果目录：

```text
outputs/composite_no_dapi_intensity/40x/all_fields/
```

重复运行时可在 PyCharm 参数中加入 `--reuse-masks`，复用已有 Cellpose mask。

## 特征开关

所有进入预处理、PCA、UMAP 和 GMM 的特征开关集中在
`run_40x_analysis.py` 的 `FEATURE_SWITCHES`。

- `True`：进入形态模型；
- `False`：不进入预处理、PCA、UMAP 或 GMM；
- 当前关闭所有直接或派生的 DAPI 强度特征；
- DAPI 图像仍用于核分割；
- YAP 永远不进入形态模型。

内部主流程集中在 `ekin_dapi_yap/` 包内，一般不需要直接运行或修改。

## 生成展示图

已有分析结果时，运行：

```text
make_ppt_figures.py
```

该脚本只读取现有 CSV，不重新分割、不重拟合 GMM、不重算 UMAP、也不重测 YAP。
默认从 40×结果生成 PNG 和 SVG。图中只保留简洁标题；较长说明写入 SVG 元数据。

## 只重新测量 YAP

如需保留形态模型、UMAP、表型和全部后验概率，只更新 YAP 测量，运行：

```text
refresh_yap_posthoc.py
```

它默认读取当前 40×结果，并写入新的带时间戳子目录，不覆盖原模型。

## 科学边界

- GMM 表型仅由 DAPI 核形态和空间邻域特征得到，不是已验证的细胞类型或细胞周期标签。
- YAP/AF488 只用于聚类后的连续生物学表征。
- YAP 指标是核内信号与 DAPI 排除核周环信号的比值；核周环是细胞质代理，不是真正的全细胞质分割。
- Ctrl、HA-1、HA-2和播种密度只用于聚类后的组间比较，不进入形态模型。
- HA-1 与 HA-2同时改变处理时长和浓度，不能把二者差异单独解释成时间效应或浓度效应。
- 每个处理组合目前只有一个图像视野，因此统计结果是描述性的，不能把单细胞当作生物学重复。
- 输入是 8-bit PNG 导出图，不是显微镜原始文件。

YAP 算法和字段说明见 `YAP_RATIO_METHOD.md`。

## 主要文件

| 文件 | 作用 |
|---|---|
| `run_40x_analysis.py` | 唯一完整运行入口与特征开关 |
| `ekin_dapi_yap/` | 分割、特征、GMM、UMAP、YAP 测量与显示的内部实现 |
| `tests/` | 回归测试；不参与正常分析运行 |
| `make_ppt_figures.py` | 从既有结果生成展示图 |
| `refresh_yap_posthoc.py` | 只重新测量 YAP，不改变形态模型 |

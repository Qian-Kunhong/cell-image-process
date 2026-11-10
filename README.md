# Cell Image Analysis

完整中文文件结构、数据来源、运行入口、统计输出和科学限制已集中到：

**[README.txt — 项目总说明](README.txt)**

- `Suzui/DAPI/`：常规 DAPI、训练集与时序分析。
- `Suzui/DAPI_OCT4/`：Suzui 配对染色对照（原 `double_work_build`）。
- `Ekin_DAPI_OCT4/`：Ekin OCT4 Model A。
- `Ekin_DAPI_YAP/`：Ekin YAP Model A。
- `shared/`：Ekin OCT4 模型兼容层；`tests/`：入口、架构与目录测试。
- `src/cell_image_core/`：跨项目共用的确定性计算内核；项目科学口径不放在这里。
- `writing_assets/`：供论文/报告项目调用的代码与图片路径清单，可选导出自包含图片包。
- `docs/architecture_inventory.md`：共享内核、重复实现、唯一真源候选与迁移顺序。
- `docs/architecture_migration_report.md`：已完成迁移、测试和真实数据回归记录。

开发环境可在仓库根目录执行 `pip install -e .`；旧入口仍可直接运行，
其兼容层会定位 `src/`，不要求先安装包。

`Ekin_DAPI_OCT4/` 保留 `run_baseline.py` 和 `run_composite.py`；
`Ekin_DAPI_YAP/` 的当前主入口是 `run_40x_analysis.py`，并将内部实现集中在
`ekin_dapi_yap/` 包中。详细操作、输出位置及 Suzui 运行方式请看总说明。

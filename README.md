# Cell Image Analysis

完整中文文件结构、数据来源、运行入口、统计输出和科学限制已集中到：

**[README.txt — 项目总说明](README.txt)**

- `Suzui/DAPI/`：常规 DAPI、训练集与时序分析。
- `Suzui/DAPI_OCT4/`：Suzui 配对染色对照（原 `double_work_build`）。
- `Ekin_DAPI_OCT4/`：Ekin OCT4 Model A。
- `Ekin_DAPI_YAP/`：Ekin YAP Model A。
- `shared/`：Ekin 公共实现；`tests/`：入口与目录测试。

Ekin 两个目录均提供可在 PyCharm 直接运行的 `run_baseline.py` 和
`run_composite.py`。详细操作、输出位置及 Suzui 运行方式请看总说明。

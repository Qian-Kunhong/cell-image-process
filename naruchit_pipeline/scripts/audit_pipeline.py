"""把已经踩过的坑变成可执行的检查，而不是写在文档里的形容词。

用法：python scripts\audit_pipeline.py
退出码：0 = 没有 FAIL（WARN 只是提醒）；1 = 有 FAIL。

检查项都对应一次真实事故：
1. nan_to_num / GaussianMixture —— 曾把缺测当零并生成未经验证的离散“表型”；
2. P1/P2/P3 —— 项目没有经过验证的三个表型；
3. 已知核级 p 值列 —— 把技术子样本当独立样本；
4. report_build 里的图没有 SIMULATED 标记 —— 假数据图会被当成实验结果；
5. 脚本 mtime 比它自己的产物新 —— 改了代码没重跑，图上其实还是旧结果（12 号脚本踩过）。
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent

# 这些文件本身含有规则文本（检查脚本/违禁词清单），不参与自我检查
SKIP = {"audit_pipeline.py", "draft_audit.py"}

SOURCE_RULES = [
    ("缺测填零", re.compile(r"np\.nan_to_num\("), "缺测行应 dropna，不许 nan→0"),
    ("未经验证的 GMM", re.compile(r"GaussianMixture\("), "主流程只报告连续特征，不生成离散表型"),
    ("P1/P2/P3", re.compile(r'「?P[123]」?/P「?|"P1/P2/P3"'), "项目没有经过验证的三个表型"),
]

# 图脚本 → 它应该新写出的产物；代码比产物新就说明改了没跑
FIGURE_OUTPUTS = {
    "04_uniformity_umap.py": "analysis_outputs/04_uniformity_umap/uniformity_umap.png",
    "05_replicate_ekin.py": "analysis_outputs/05_replicate_ekin/ha_effect_forest.png",
    "05b_fix_forest.py": "analysis_outputs/05_replicate_ekin/ha_effect_forest_fixed.png",
    "05c_english_labels.py": "analysis_outputs/05_replicate_ekin/yap_boxplot_en.png",
    "08_3d_segmentation_overlay.py": "analysis_outputs/05_replicate_ekin/segmentation_3d_overlay.png",
    "09_2d_segmentation_overlay.py": "analysis_outputs/05_replicate_ekin/segmentation_2d_overlay.png",
    "10_ekin_style_overlay.py": "analysis_outputs/05_replicate_ekin/ekin_style_overlay.png",
    "12_phenotype_grid.py": "analysis_outputs/05_replicate_ekin/segmentation_grid.png",
}

fails: list[str] = []
warns: list[str] = []


def scan_sources() -> None:
    targets = sorted((ROOT / "scripts").glob("*.py")) + sorted(REPO.glob("*.py"))
    for path in targets:
        if path.name in SKIP:
            continue
        text = path.read_text(encoding="utf-8-sig")
        for label, pattern, why in SOURCE_RULES:
            if pattern.search(text):
                fails.append(f"[源码] {path.relative_to(REPO)}：{label} —— {why}")


def scan_known_nuclear_p_columns() -> None:
    invalid = {"spearman_p", "pearson_p", "pooled_p", "within_field_permutation_p"}
    roots = [ROOT / "analysis_outputs", REPO / "reports"]
    for base in roots:
        if not base.exists():
            continue
        for path in base.rglob("*.csv"):
            with open(path, encoding="utf-8-sig", newline="") as fh:
                header = next(csv.reader(fh), [])
            bad = [h for h in header if h.strip() in invalid]
            if bad:
                fails.append(f"[表头] {path.relative_to(REPO)}：{bad} —— 这些列把核当成独立样本")


def scan_simulated() -> None:
    build = REPO / "report_build"
    if not build.exists():
        return
    for path in sorted(build.glob("*.py")):
        text = path.read_text(encoding="utf-8-sig")
        if "savefig" in text and "SIMULATED" not in text:
            fails.append(f"[假数据] report_build/{path.name}：出图但没说 SIMULATED")


def scan_stale_outputs() -> None:
    for script, rel in FIGURE_OUTPUTS.items():
        code = ROOT / "scripts" / script
        out = ROOT / rel
        if not code.exists():
            continue
        if not out.exists():
            fails.append(f"[未跑] {script}：产物 {rel} 不存在")
        elif code.stat().st_mtime > out.stat().st_mtime + 1:
            fails.append(f"[未跑] {script}：代码比 {out.name} 新，改了没重跑")


def scan_entrypoints() -> None:
    missing = []
    for path in sorted((ROOT / "scripts").glob("*.py")) + sorted(REPO.glob("*.py")):
        if path.name in SKIP:
            continue
        text = path.read_text(encoding="utf-8-sig")
        if "savefig" in text or "to_csv" in text:
            if '__name__ == "__main__"' not in text:
                missing.append(path.name)
    if missing:
        warns.append(f"[入口] 导入即执行、没有 __main__ 守卫：{len(missing)} 个（{', '.join(missing[:6])}…）")


def main() -> int:
    scan_sources()
    scan_known_nuclear_p_columns()
    scan_simulated()
    scan_stale_outputs()
    scan_entrypoints()

    for item in fails:
        print("FAIL", item)
    for item in warns:
        print("WARN", item)
    if not fails and not warns:
        print("PASS：没有命中任何已知坑")
    elif not fails:
        print(f"PASS（{len(warns)} 条 WARN 待处理）")
    else:
        print(f"FAIL {len(fails)} 条，WARN {len(warns)} 条")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

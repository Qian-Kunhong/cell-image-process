"""核对 docs/counting_ledger.md 里六个核数口径是否与原始文件一致。

用法：python scripts\verify_count_ledger.py
台账里的数字由本脚本从 markdown 读出来现算比对；台账写错、行被删掉都会 FAIL。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LEDGER = Path(__file__).resolve().parent.parent / "docs" / "counting_ledger.md"

# 台账表格里的中文标签 → 内部口径名。改台账措辞时要同步改这里。
LABEL_TO_KEY = {
    "3D 原始检测": "raw_3d",
    "3D 有效核": "valid_3d",
    "旧汇总口径": "legacy_sum",
    "2D 切片上的核": "yap_2d",
    "2D–3D 匹配核": "matched",
    "最佳焦平面 2D 掩膜核": "mask_2d",
}
ROW = re.compile(r"^\|\s*\*\*(\d+)\*\*\s*\|\s*(.+?)\s*\|", re.MULTILINE)


def read_ledger(path: Path = LEDGER) -> dict[str, int]:
    """从台账表格抽出六个数字；缺项直接报错，避免脚本在没有台账时静默 PASS。"""
    text = path.read_text(encoding="utf-8-sig")
    found: dict[str, int] = {}
    for number, label in ROW.findall(text):
        key = LABEL_TO_KEY.get(label)
        if key:
            found[key] = int(number)
    missing = sorted(set(LABEL_TO_KEY.values()) - set(found))
    if missing:
        raise RuntimeError(f"台账里缺这些口径的数字：{missing}（{path}）")
    return found


def load_paths():
    cfg = json.loads((Path(__file__).resolve().parent.parent / "config" / "project.json").read_text(encoding="utf-8-sig"))
    legacy = Path(cfg["legacy_code_root"])
    yap3d = legacy / "analysis_outputs" / "hepatocyte_confocal_audit" / "dapi_3d" / "yap"
    yap2d = Path(cfg["data_root"]) / "03_Raw data" / "Immunostaining" / "YAP quantification"
    return yap3d, yap2d


def count_3d(yap3d: Path):
    raw = valid = rows = 0
    for sample_dir in sorted(p for p in yap3d.iterdir() if p.is_dir()):
        masks = sample_dir / "dapi_3d_masks.npz"
        if masks.exists():
            z = np.load(masks, allow_pickle=True)
            if "raw_labels" in z.files:
                raw += len(np.unique(z["raw_labels"])) - 1
            if "labels" in z.files:
                valid += len(np.unique(z["labels"])) - 1
        metrics = sample_dir / "nucleus_3d_metrics.csv"
        if metrics.exists():
            rows += len(pd.read_csv(metrics, encoding="utf-8-sig"))
    return raw, valid, rows


def count_legacy(yap3d: Path) -> int:
    f = yap3d / "analysis_summary" / "condition_3d_summary.csv"
    return int(pd.read_csv(f, encoding="utf-8-sig")["n_nuclei"].sum())


def count_yap2d(yap2d: Path) -> int:
    total = 0
    for f in yap2d.glob("*/Ana_*/*_Nuclei.csv"):
        total += len(pd.read_csv(f, encoding="utf-8-sig"))
    return total


def count_matched(yap3d: Path) -> int:
    f = yap3d / "best_focus_2d_vs_3d" / "matched_best_focus_2d_to_3d_nuclei.csv"
    return len(pd.read_csv(f, encoding="utf-8-sig"))


def count_mask_2d(yap3d: Path) -> int:
    """最佳焦平面 2D 掩膜里的核数合计（图 08/09/10/12 的口径：9 样本为 1182）。"""
    total = 0
    for f in sorted((yap3d / "best_focus_2d_vs_3d" / "masks_2d").glob("*.npz")):
        labels = np.load(f)["labels"]
        total += len(np.unique(labels)) - 1
    return total


def main() -> int:
    expected = read_ledger()
    yap3d, yap2d = load_paths()
    raw, valid, rows = count_3d(yap3d)
    actual = {
        "raw_3d": raw,
        "valid_3d": valid,
        "legacy_sum": count_legacy(yap3d),
        "yap_2d": count_yap2d(yap2d),
        "matched": count_matched(yap3d),
        "mask_2d": count_mask_2d(yap3d),
    }
    print(f"{'口径':<12}{'台账':>6}{'实测':>6}  状态")
    bad = []
    for k, exp in expected.items():
        act = actual[k]
        ok = act == exp
        print(f"{k:<12}{exp:>6}{act:>6}  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            bad.append(k)
    # 交叉自洽：掩膜 labels 唯一值必须等于逐核表行数
    if valid != rows:
        print(f"INTERNAL: labels 唯一值 {valid} != 逐核表行数 {rows}")
        bad.append("internal")
    if bad:
        print("FAIL:", ", ".join(bad))
        return 1
    print("PASS: 六个口径与台账一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""按 PROJECT_RULES.md 第4节对已交付文档做一次机械自检，输出问题清单。

用法：python draft_audit.py [docx 路径 ...]
不给路径时读环境变量 DRAFT_DIR（默认 E:\\Kino-oka Lab\\2026 discussion）。
"""
import re, sys, os
from pathlib import Path
from docx import Document

DEFAULT_DIR = Path(os.environ.get("DRAFT_DIR", r"E:\Kino-oka Lab\2026 discussion"))
DOC_NAMES = [
    "Original paper story - hepatocyte differentiation (Nov 2026).docx",
    "Future plan - hepatocyte differentiation (Nov 2026).docx",
]
DOCS = [str(p) for p in (Path(a) for a in sys.argv[1:])] or [str(DEFAULT_DIR / n) for n in DOC_NAMES]

META = ["需要强调", "这句话", "换句话", "推理的桥梁", "仅支持", "值得注意的是",
        "也就是说", "如前所述", "不言而喻", "换句话说", "简单来说", "可以说"]
FABRICATED = ["Table 1", "Table1", "表1", "P1/P2/P3", "P1", "P2", "P3"]
JARGON = ["孔级", "构成量", "表型轴", "R² 增量", "R2 增量", "well-level", "孔键", "构成事件"]

def audit(path):
    doc = Document(path)
    print("\n" + "=" * 70)
    print(os.path.basename(path))
    hits = {"meta": [], "fabricated": [], "jargon": [], "question": []}
    paras = [p for p in doc.paragraphs if p.text.strip()]
    for i, p in enumerate(paras):
        t = p.text.strip()
        style = (p.style.name or "")
        for k in META:
            if k in t:
                hits["meta"].append((i, k, t[:90]))
        for k in FABRICATED:
            if re.search(r"\b" + re.escape(k) + r"\b", t) or (k in ("表1",) and k in t):
                hits["fabricated"].append((i, k, t[:90]))
        for k in JARGON:
            if k in t:
                # 同段或上一段有没有括号/“即”解释
                ctx = t + (paras[i - 1].text if i else "")
                if not ("（" in t or "即" in ctx or "(" in t):
                    hits["jargon"].append((i, k, t[:90]))
        if ("Heading" in style or "标题" in style or len(t) < 30) and t.endswith("?") or t.endswith("？"):
            hits["question"].append((i, t[:90]))
    for cat, label in [("meta", "元话语/车轱辘"), ("fabricated", "可疑编造结构"),
                       ("jargon", "未定义自造术语"), ("question", "问句标题")]:
        print(f"\n[{label}] {len(hits[cat])} 处")
        seen = set()
        for i, k, t in hits[cat]:
            if (i, k) in seen:
                continue
            seen.add((i, k))
            print(f"  #{i} <{k}> {t}")
    return hits

total = 0
for d in DOCS:
    if os.path.exists(d):
        h = audit(d)
        total += sum(len(v) for v in h.values())
    else:
        print("missing:", d)
print("\nTOTAL_HITS", total)

from pathlib import Path
import re
import numpy as np
import tifffile as tiff
import cv2
import pandas as pd

SUZUI_ROOT = Path(r"F:\Suzui")
ANALYSIS_ROOT = SUZUI_ROOT / "analysis_out"
DATASET_NAME = "A-1-1 timelapse"

# SRC = SUZUI_ROOT / "BF and Dapi"
SRC = SUZUI_ROOT / DATASET_NAME

# 新文件夹，避免覆盖旧结果
OUT_DAPI = ANALYSIS_ROOT / DATASET_NAME
OUT_BF   = ANALYSIS_ROOT / f"{DATASET_NAME}_bf"
OUT_DAPI.mkdir(parents=True, exist_ok=True)
OUT_BF.mkdir(parents=True, exist_ok=True)

REPORT = OUT_DAPI / "focus_report_filtered.csv"

# ---------- 参数 ----------
# 绝对清晰度：保留全体 best_score 的前多少比例
KEEP_TOP_FRACTION = 0.50

# 组内优势：best 必须比 second 至少高这么多
# 例如 1.08 = 至少高 8%
MIN_SCORE_RATIO = 1.08

# 也可以同时要求 best-second 的绝对差值足够大
MIN_SCORE_DIFF = 0.0005

# True = 每组至少保留 1 张 best focus。
# 这样不会因为全局阈值导致某些组整组被丢掉。
KEEP_AT_LEAST_ONE_PER_GROUP = True

# None = 全量跑
MAX_GROUPS = None

# DAPI 文件
pat_dapi = re.compile(
    r"^(?P<container>[A-Z])\s*-\s*(?P<pos>\d+)\(fld\s*(?P<fld>\d+)\s*wv\s*DAPI.*time\s*(?P<time>\d+).*(?:wix\s*(?P<wix>\d+))\)\.tif$",
    re.IGNORECASE
)

# BF 文件
pat_bf = re.compile(
    r"^(?P<container>[A-Z])\s*-\s*(?P<pos>\d+)\(fld\s*(?P<fld>\d+)\s*wv\s*TL-Brightfield.*time\s*(?P<time>\d+).*\)\.tif$",
    re.IGNORECASE
)

def tenengrad_focus(img_u16, sigma=1.5, crop=0.10):
    """保留原版评分逻辑"""
    img = img_u16.astype(np.float32)
    p1, p99 = np.percentile(img, [1, 99])
    img = np.clip((img - p1) / (p99 - p1 + 1e-6), 0, 1)
    img = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)

    h, w = img.shape
    c = img[int(h * crop):int(h * (1 - crop)), int(w * crop):int(w * (1 - crop))]

    gx = cv2.Sobel(c, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(c, cv2.CV_64F, 0, 1, ksize=3)
    return float((gx * gx + gy * gy).mean())

# ---------- 收集文件 ----------
dapi_groups = {}
bf_map = {}

for p in SRC.glob("*.tif"):
    name = p.name

    m = pat_dapi.match(name)
    if m:
        key = (
            m.group("container"),
            int(m.group("pos")),
            int(m.group("fld")),
            int(m.group("time"))
        )
        wix = int(m.group("wix"))
        dapi_groups.setdefault(key, {})[wix] = p
        continue

    m = pat_bf.match(name)
    if m:
        key = (
            m.group("container"),
            int(m.group("pos")),
            int(m.group("fld")),
            int(m.group("time"))
        )
        bf_map[key] = p
        continue

print("DAPI groups:", len(dapi_groups))
print("BF images :", len(bf_map))

# ---------- 第一遍：算每组分数 ----------
all_rows = []

items = list(dapi_groups.items())
if MAX_GROUPS is not None:
    items = items[:MAX_GROUPS]

for key, wix_dict in items:
    container, pos, fld, time = key

    scores = {}
    for wix, path in wix_dict.items():
        img = tiff.imread(path)
        scores[wix] = tenengrad_focus(img, sigma=1.5, crop=0.10)

    # 从高到低排序
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    best_wix, best_score = sorted_items[0]
    if len(sorted_items) >= 2:
        second_wix, second_score = sorted_items[1]
    else:
        second_wix, second_score = np.nan, 0.0

    score_ratio = best_score / (second_score + 1e-12)
    score_diff = best_score - second_score
    best_path = wix_dict[best_wix]

    row = {
        "container": container,
        "pos": pos,
        "fld": fld,
        "time": time,
        "best_wix": best_wix,
        "second_wix": second_wix,
        "best_score": best_score,
        "second_score": second_score,
        "score_ratio": score_ratio,
        "score_diff": score_diff,
        "score_wix1": scores.get(1, np.nan),
        "score_wix2": scores.get(2, np.nan),
        "score_wix3": scores.get(3, np.nan),
        "best_file": str(best_path),
    }
    all_rows.append(row)

df = pd.DataFrame(all_rows)

if len(df) == 0:
    raise RuntimeError("No DAPI groups found.")

# ---------- 第二遍：全局绝对阈值 ----------
abs_threshold = df["best_score"].quantile(1.0 - KEEP_TOP_FRACTION)

# ---------- 第三遍：综合判断 ----------
df["pass_absolute"] = df["best_score"] >= abs_threshold
df["pass_ratio"] = df["score_ratio"] >= MIN_SCORE_RATIO
df["pass_diff"] = df["score_diff"] >= MIN_SCORE_DIFF

# 同时满足才保留
df["keep_by_qc"] = df["pass_absolute"] & df["pass_ratio"] & df["pass_diff"]
df["keep"] = df["keep_by_qc"]

if KEEP_AT_LEAST_ONE_PER_GROUP:
    df["keep"] = True

print(f"best_score threshold = {abs_threshold:.6f}")
print(f"MIN_SCORE_RATIO = {MIN_SCORE_RATIO}")
print(f"MIN_SCORE_DIFF  = {MIN_SCORE_DIFF}")
print(f"QC-pass groups = {int(df['keep_by_qc'].sum())} / {len(df)}")
if KEEP_AT_LEAST_ONE_PER_GROUP:
    print("KEEP_AT_LEAST_ONE_PER_GROUP = True")
    print(f"Will keep {int(df['keep'].sum())} / {len(df)} groups (best image from every group)")
else:
    print(f"Will keep {int(df['keep'].sum())} / {len(df)} groups")

# ---------- 第四遍：只保存 keep=True ----------
picked = 0
skipped = 0

for _, row in df.iterrows():
    key = (row["container"], int(row["pos"]), int(row["fld"]), int(row["time"]))
    best_wix = int(row["best_wix"])
    best_path = Path(row["best_file"])

    if not bool(row["keep"]):
        skipped += 1
        continue

    out_dapi_name = (
        f"{row['container']}-{int(row['pos'])}"
        f"_fld{int(row['fld']):03d}_time{int(row['time']):03d}"
        f"_DAPI_bestwix{best_wix}.tif"
    )
    (OUT_DAPI / out_dapi_name).write_bytes(best_path.read_bytes())

    if key in bf_map:
        out_bf_name = (
            f"{row['container']}-{int(row['pos'])}"
            f"_fld{int(row['fld']):03d}_time{int(row['time']):03d}_BF.tif"
        )
        (OUT_BF / out_bf_name).write_bytes(bf_map[key].read_bytes())

    picked += 1

df.to_csv(REPORT, index=False)

print("Picked best-focus DAPI    :", picked)
print("QC-pass groups           :", int(df["keep_by_qc"].sum()))
print("QC-flagged groups        :", int((~df["keep_by_qc"]).sum()))
print("Skipped groups           :", skipped)
print("Saved DAPI to:", OUT_DAPI)
print("Saved BF  to:", OUT_BF)
print("Saved report:", REPORT)

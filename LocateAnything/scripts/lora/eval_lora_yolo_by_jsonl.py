#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_lora_yolo_by_folder.py

功能：
1. 单模型评测：
   使用 val_quick100.jsonl / val_full.jsonl / test.jsonl 中的 GT box，
   对模型输出的 YOLO label 进行 IoU 评测。

2. 自动输出：
   - 每张图明细 CSV
   - 总体 summary JSON
   - 按 GT 面积分组统计 CSV

3. 双模型对比：
   对比 Base 和 LoRA 的评测 CSV，输出 delta_iou、面积比变化、按 GT 面积分组统计。

适配目录结构：
pred_root/
  ├── 1416/
  │   ├── labels/
  │   │   ├── 14160002.txt
  │   ├── json/
  │   ├── vis/

也兼容：
pred_root/
  ├── 1416/
  │   ├── 14160002.txt
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


BOX_RE = re.compile(
    r"<box>\s*<\s*([0-9.]+)\s*>\s*<\s*([0-9.]+)\s*>\s*<\s*([0-9.]+)\s*>\s*<\s*([0-9.]+)\s*>\s*</box>"
)


IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def normalize_box(box):
    """
    输入 xyxy，输出裁剪到 0~1 的 xyxy。
    """
    if box is None:
        return None

    x1, y1, x2, y2 = map(float, box)

    xa, xb = sorted([x1, x2])
    ya, yb = sorted([y1, y2])

    xa = max(0.0, min(1.0, xa))
    ya = max(0.0, min(1.0, ya))
    xb = max(0.0, min(1.0, xb))
    yb = max(0.0, min(1.0, yb))

    if xb <= xa or yb <= ya:
        return None

    return [xa, ya, xb, yb]


def parse_gt_box_from_answer(answer):
    """
    解析 LocateAnything 标准答案：
    <box><x1><y1><x2><y2></box>

    坐标范围是 0~1000，转换到 0~1。
    """
    if not answer:
        return None

    m = BOX_RE.search(str(answer))
    if not m:
        return None

    x1, y1, x2, y2 = map(float, m.groups())

    box = [
        x1 / 1000.0,
        y1 / 1000.0,
        x2 / 1000.0,
        y2 / 1000.0,
    ]

    return normalize_box(box)


def get_answer_from_item(item):
    """
    从 conversations 中找到 gpt / assistant 的回答。
    """
    convs = item.get("conversations", [])

    for c in convs:
        role = c.get("from", "")
        if role in ["gpt", "assistant"]:
            return c.get("value", "")

    # 兼容非 conversations 格式
    if "answer" in item:
        return item["answer"]

    return ""


def yolo_to_xyxy(line):
    """
    YOLO:
    class_id xc yc w h

    返回归一化 xyxy。
    """
    parts = line.strip().split()

    if len(parts) < 5:
        return None

    try:
        _, xc, yc, w, h = parts[:5]
        xc = float(xc)
        yc = float(yc)
        w = float(w)
        h = float(h)
    except Exception:
        return None

    if w <= 0 or h <= 0:
        return None

    box = [
        xc - w / 2.0,
        yc - h / 2.0,
        xc + w / 2.0,
        yc + h / 2.0,
    ]

    return normalize_box(box)


def box_area(box):
    if box is None:
        return 0.0

    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(a, b):
    if a is None or b is None:
        return 0.0

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter

    if union <= 0:
        return 0.0

    return inter / union


def intersection_area(a, b):
    if a is None or b is None:
        return 0.0

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def area_bin(a):
    """
    按 GT 面积分组。
    a 是归一化面积，范围 0~1。
    """
    if a < 0.3:
        return "small_gt(<0.3)"
    elif a < 0.6:
        return "mid_gt(0.3-0.6)"
    elif a < 0.85:
        return "large_gt(0.6-0.85)"
    else:
        return "huge_gt(>=0.85)"


def find_pred_label(pred_root, folder, image_name):
    """
    支持两种预测目录结构：
    1. pred_root/folder/labels/stem.txt
    2. pred_root/folder/stem.txt
    """
    pred_root = Path(pred_root)
    stem = Path(image_name).stem

    candidates = [
        pred_root / str(folder) / "labels" / f"{stem}.txt",
        pred_root / str(folder) / f"{stem}.txt",
    ]

    for p in candidates:
        if p.exists():
            return p

    return candidates[0]


def read_pred_box(label_path):
    """
    读取预测 YOLO label。
    如果多行，取面积最大的框。
    """
    label_path = Path(label_path)

    if not label_path.exists():
        return None

    boxes = []

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        box = yolo_to_xyxy(line)
        if box is not None:
            boxes.append(box)

    if not boxes:
        return None

    boxes.sort(key=box_area, reverse=True)
    return boxes[0]


def load_jsonl(path):
    items = []

    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                items.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] bad jsonl line {line_id}: {e}")

    return items


def get_image_info(item):
    """
    从 jsonl item 中获取 folder 和 image_name。
    """
    image_path = Path(item.get("image", ""))

    folder = item.get("folder")
    if folder is None or str(folder).strip() == "":
        folder = image_path.parent.name

    image_name = item.get("image_name")
    if image_name is None or str(image_name).strip() == "":
        image_name = image_path.name

    return str(folder), str(image_name), str(image_path)


def evaluate_one_model(
    split_jsonl,
    pred_root,
    out_csv,
    full_image_thr=0.95,
    large_box_thr=0.85,
    small_box_thr=0.01,
):
    split_jsonl = Path(split_jsonl)
    pred_root = Path(pred_root)
    out_csv = Path(out_csv)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    items = load_jsonl(split_jsonl)

    rows = []

    for idx, item in enumerate(items):
        folder, image_name, image_path = get_image_info(item)

        answer = get_answer_from_item(item)
        gt_box = parse_gt_box_from_answer(answer)

        pred_label = find_pred_label(pred_root, folder, image_name)
        pred_box = read_pred_box(pred_label)

        iou = box_iou(gt_box, pred_box)

        gt_area = box_area(gt_box)
        pred_area = box_area(pred_box)

        inter = intersection_area(gt_box, pred_box)

        coverage = inter / (gt_area + 1e-8)
        purity = inter / (pred_area + 1e-8)
        area_ratio = pred_area / (gt_area + 1e-8)

        row = {
            "idx": idx,
            "folder": folder,
            "image_name": image_name,
            "image_path": image_path,
            "pred_label": str(pred_label),
            "pred_exists": int(pred_label.exists()),
            "iou": iou,
            "hit50": int(iou >= 0.5),
            "hit75": int(iou >= 0.75),
            "missing_gt": int(gt_box is None),
            "missing_pred": int(pred_box is None),
            "gt_area": gt_area,
            "pred_area": pred_area,
            "area_ratio": area_ratio,
            "coverage_inter_over_gt": coverage,
            "purity_inter_over_pred": purity,
            "gt_bin": area_bin(gt_area),
            "full_image": int(pred_area >= full_image_thr),
            "large_box": int(pred_area >= large_box_thr),
            "small_box": int(0.0 < pred_area < small_box_thr),
        }

        if gt_box is not None:
            row.update({
                "gt_x1": gt_box[0],
                "gt_y1": gt_box[1],
                "gt_x2": gt_box[2],
                "gt_y2": gt_box[3],
            })
        else:
            row.update({
                "gt_x1": "",
                "gt_y1": "",
                "gt_x2": "",
                "gt_y2": "",
            })

        if pred_box is not None:
            row.update({
                "pred_x1": pred_box[0],
                "pred_y1": pred_box[1],
                "pred_x2": pred_box[2],
                "pred_y2": pred_box[3],
            })
        else:
            row.update({
                "pred_x1": "",
                "pred_y1": "",
                "pred_x2": "",
                "pred_y2": "",
            })

        rows.append(row)

    save_detail_csv(rows, out_csv)

    summary = make_summary(
        rows=rows,
        split_jsonl=split_jsonl,
        pred_root=pred_root,
        full_image_thr=full_image_thr,
        large_box_thr=large_box_thr,
        small_box_thr=small_box_thr,
    )

    summary_json = out_csv.with_suffix(".summary.json")
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    group_csv = out_csv.with_name(out_csv.stem + "_by_gt_area.csv")
    save_group_summary(rows, group_csv)

    print_eval_summary(summary, out_csv, summary_json, group_csv)

    return rows, summary


def save_detail_csv(rows, out_csv):
    if not rows:
        print("[WARN] no rows to save")
        return

    fieldnames = list(rows[0].keys())

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("[Saved detail csv]", out_csv)


def make_summary(
    rows,
    split_jsonl,
    pred_root,
    full_image_thr,
    large_box_thr,
    small_box_thr,
):
    total = len(rows)

    if total == 0:
        return {
            "split_jsonl": str(split_jsonl),
            "pred_root": str(pred_root),
            "total": 0,
        }

    ious = [safe_float(r["iou"]) for r in rows]

    summary = {
        "split_jsonl": str(split_jsonl),
        "pred_root": str(pred_root),
        "total": total,

        "mean_iou": mean(ious),
        "median_iou": median(ious),
        "acc50": mean([safe_float(r["hit50"]) for r in rows]),
        "acc75": mean([safe_float(r["hit75"]) for r in rows]),

        "missing_gt_rate": mean([safe_float(r["missing_gt"]) for r in rows]),
        "missing_pred_rate": mean([safe_float(r["missing_pred"]) for r in rows]),

        "full_image_rate": mean([safe_float(r["full_image"]) for r in rows]),
        "large_box_rate": mean([safe_float(r["large_box"]) for r in rows]),
        "small_box_rate": mean([safe_float(r["small_box"]) for r in rows]),

        "mean_gt_area": mean([safe_float(r["gt_area"]) for r in rows]),
        "mean_pred_area": mean([safe_float(r["pred_area"]) for r in rows]),
        "mean_area_ratio_pred_over_gt": mean([safe_float(r["area_ratio"]) for r in rows]),
        "mean_coverage_inter_over_gt": mean([safe_float(r["coverage_inter_over_gt"]) for r in rows]),
        "mean_purity_inter_over_pred": mean([safe_float(r["purity_inter_over_pred"]) for r in rows]),

        "full_image_threshold": full_image_thr,
        "large_box_threshold": large_box_thr,
        "small_box_threshold": small_box_thr,

        "missing_gt": sum(int(r["missing_gt"]) for r in rows),
        "missing_pred": sum(int(r["missing_pred"]) for r in rows),
        "full_image": sum(int(r["full_image"]) for r in rows),
        "large_box": sum(int(r["large_box"]) for r in rows),
        "small_box": sum(int(r["small_box"]) for r in rows),
    }

    return summary


def save_group_summary(rows, out_csv):
    groups = defaultdict(list)

    for r in rows:
        groups[r["gt_bin"]].append(r)

    summary_rows = []

    order = [
        "small_gt(<0.3)",
        "mid_gt(0.3-0.6)",
        "large_gt(0.6-0.85)",
        "huge_gt(>=0.85)",
    ]

    for name in order:
        items = groups.get(name, [])

        if not items:
            continue

        count = len(items)

        summary_rows.append({
            "gt_bin": name,
            "count": count,
            "mean_iou": mean([safe_float(x["iou"]) for x in items]),
            "median_iou": median([safe_float(x["iou"]) for x in items]),
            "acc50": mean([safe_float(x["hit50"]) for x in items]),
            "acc75": mean([safe_float(x["hit75"]) for x in items]),
            "mean_gt_area": mean([safe_float(x["gt_area"]) for x in items]),
            "mean_pred_area": mean([safe_float(x["pred_area"]) for x in items]),
            "mean_area_ratio_pred_over_gt": mean([safe_float(x["area_ratio"]) for x in items]),
            "mean_coverage_inter_over_gt": mean([safe_float(x["coverage_inter_over_gt"]) for x in items]),
            "mean_purity_inter_over_pred": mean([safe_float(x["purity_inter_over_pred"]) for x in items]),
            "missing_pred_rate": mean([safe_float(x["missing_pred"]) for x in items]),
            "full_image_rate": mean([safe_float(x["full_image"]) for x in items]),
            "large_box_rate": mean([safe_float(x["large_box"]) for x in items]),
            "small_box_rate": mean([safe_float(x["small_box"]) for x in items]),
        })

    if not summary_rows:
        return

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(summary_rows[0].keys())

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("[Saved group summary]", out_csv)


def print_eval_summary(summary, out_csv, summary_json, group_csv):
    print("=" * 80)
    print("Evaluation Result")
    print("=" * 80)
    print("Total:", summary.get("total"))
    print("Mean IoU:", summary.get("mean_iou"))
    print("Median IoU:", summary.get("median_iou"))
    print("Acc@0.5:", summary.get("acc50"))
    print("Acc@0.75:", summary.get("acc75"))
    print("Missing pred rate:", summary.get("missing_pred_rate"))
    print("Full image rate:", summary.get("full_image_rate"))
    print("Large box rate:", summary.get("large_box_rate"))
    print("Mean GT area:", summary.get("mean_gt_area"))
    print("Mean Pred area:", summary.get("mean_pred_area"))
    print("Mean pred/gt area ratio:", summary.get("mean_area_ratio_pred_over_gt"))
    print("Mean coverage inter/gt:", summary.get("mean_coverage_inter_over_gt"))
    print("Mean purity inter/pred:", summary.get("mean_purity_inter_over_pred"))
    print("-" * 80)
    print("Saved detail:", out_csv)
    print("Saved summary:", summary_json)
    print("Saved group:", group_csv)
    print("=" * 80)


def load_csv_as_dict(path):
    rows = []

    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    return rows


def compare_two_models(base_csv, lora_csv, out_csv):
    base_csv = Path(base_csv)
    lora_csv = Path(lora_csv)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    base_rows = load_csv_as_dict(base_csv)
    lora_rows = load_csv_as_dict(lora_csv)

    def make_key(r):
        return (str(r.get("folder")), str(r.get("image_name")))

    base_map = {make_key(r): r for r in base_rows}
    lora_map = {make_key(r): r for r in lora_rows}

    keys = sorted(set(base_map.keys()) & set(lora_map.keys()))

    rows = []

    for folder, image_name in keys:
        b = base_map[(folder, image_name)]
        l = lora_map[(folder, image_name)]

        iou_base = safe_float(b.get("iou"))
        iou_lora = safe_float(l.get("iou"))

        gt_area = safe_float(b.get("gt_area"))
        pred_area_base = safe_float(b.get("pred_area"))
        pred_area_lora = safe_float(l.get("pred_area"))

        area_ratio_base = pred_area_base / (gt_area + 1e-8)
        area_ratio_lora = pred_area_lora / (gt_area + 1e-8)

        row = {
            "folder": folder,
            "image_name": image_name,
            "gt_bin": b.get("gt_bin"),
            "gt_area": gt_area,

            "iou_base": iou_base,
            "iou_lora": iou_lora,
            "delta_iou": iou_lora - iou_base,

            "pred_area_base": pred_area_base,
            "pred_area_lora": pred_area_lora,

            "area_ratio_base": area_ratio_base,
            "area_ratio_lora": area_ratio_lora,
            "delta_area_ratio": area_ratio_lora - area_ratio_base,

            "coverage_base": safe_float(b.get("coverage_inter_over_gt")),
            "coverage_lora": safe_float(l.get("coverage_inter_over_gt")),
            "delta_coverage": safe_float(l.get("coverage_inter_over_gt")) - safe_float(b.get("coverage_inter_over_gt")),

            "purity_base": safe_float(b.get("purity_inter_over_pred")),
            "purity_lora": safe_float(l.get("purity_inter_over_pred")),
            "delta_purity": safe_float(l.get("purity_inter_over_pred")) - safe_float(b.get("purity_inter_over_pred")),

            "base_better": int(iou_base > iou_lora),
            "lora_better": int(iou_lora > iou_base),
            "same_iou": int(abs(iou_lora - iou_base) < 1e-8),
        }

        rows.append(row)

    if not rows:
        print("[WARN] no matched rows between base and lora csv")
        return

    fieldnames = list(rows[0].keys())

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("[Saved compare detail]", out_csv)

    group_csv = out_csv.with_name(out_csv.stem + "_by_gt_area.csv")
    save_compare_group_summary(rows, group_csv)

    summary_json = out_csv.with_suffix(".summary.json")
    summary = make_compare_summary(rows, base_csv, lora_csv)
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print_compare_summary(summary, summary_json, group_csv)


def make_compare_summary(rows, base_csv, lora_csv):
    total = len(rows)

    return {
        "base_csv": str(base_csv),
        "lora_csv": str(lora_csv),
        "total": total,

        "base_mean_iou": mean([safe_float(r["iou_base"]) for r in rows]),
        "lora_mean_iou": mean([safe_float(r["iou_lora"]) for r in rows]),
        "mean_delta_iou": mean([safe_float(r["delta_iou"]) for r in rows]),

        "base_median_iou": median([safe_float(r["iou_base"]) for r in rows]),
        "lora_median_iou": median([safe_float(r["iou_lora"]) for r in rows]),

        "lora_better_count": sum(int(r["lora_better"]) for r in rows),
        "base_better_count": sum(int(r["base_better"]) for r in rows),
        "same_iou_count": sum(int(r["same_iou"]) for r in rows),

        "base_mean_area_ratio": mean([safe_float(r["area_ratio_base"]) for r in rows]),
        "lora_mean_area_ratio": mean([safe_float(r["area_ratio_lora"]) for r in rows]),
        "mean_delta_area_ratio": mean([safe_float(r["delta_area_ratio"]) for r in rows]),

        "base_mean_coverage": mean([safe_float(r["coverage_base"]) for r in rows]),
        "lora_mean_coverage": mean([safe_float(r["coverage_lora"]) for r in rows]),
        "mean_delta_coverage": mean([safe_float(r["delta_coverage"]) for r in rows]),

        "base_mean_purity": mean([safe_float(r["purity_base"]) for r in rows]),
        "lora_mean_purity": mean([safe_float(r["purity_lora"]) for r in rows]),
        "mean_delta_purity": mean([safe_float(r["delta_purity"]) for r in rows]),
    }


def save_compare_group_summary(rows, out_csv):
    groups = defaultdict(list)

    for r in rows:
        groups[r["gt_bin"]].append(r)

    summary_rows = []

    order = [
        "small_gt(<0.3)",
        "mid_gt(0.3-0.6)",
        "large_gt(0.6-0.85)",
        "huge_gt(>=0.85)",
    ]

    for name in order:
        items = groups.get(name, [])

        if not items:
            continue

        summary_rows.append({
            "gt_bin": name,
            "count": len(items),

            "base_iou": mean([safe_float(x["iou_base"]) for x in items]),
            "lora_iou": mean([safe_float(x["iou_lora"]) for x in items]),
            "delta_iou": mean([safe_float(x["delta_iou"]) for x in items]),

            "base_area_ratio": mean([safe_float(x["area_ratio_base"]) for x in items]),
            "lora_area_ratio": mean([safe_float(x["area_ratio_lora"]) for x in items]),
            "delta_area_ratio": mean([safe_float(x["delta_area_ratio"]) for x in items]),

            "base_coverage": mean([safe_float(x["coverage_base"]) for x in items]),
            "lora_coverage": mean([safe_float(x["coverage_lora"]) for x in items]),
            "delta_coverage": mean([safe_float(x["delta_coverage"]) for x in items]),

            "base_purity": mean([safe_float(x["purity_base"]) for x in items]),
            "lora_purity": mean([safe_float(x["purity_lora"]) for x in items]),
            "delta_purity": mean([safe_float(x["delta_purity"]) for x in items]),

            "lora_better_count": sum(int(x["lora_better"]) for x in items),
            "base_better_count": sum(int(x["base_better"]) for x in items),
            "same_iou_count": sum(int(x["same_iou"]) for x in items),
        })

    if not summary_rows:
        return

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("[Saved compare group]", out_csv)


def print_compare_summary(summary, summary_json, group_csv):
    print("=" * 80)
    print("Base vs LoRA Compare Result")
    print("=" * 80)

    print("Total:", summary.get("total"))

    print("Base mean IoU:", summary.get("base_mean_iou"))
    print("LoRA mean IoU:", summary.get("lora_mean_iou"))
    print("Mean delta IoU:", summary.get("mean_delta_iou"))

    print("Base median IoU:", summary.get("base_median_iou"))
    print("LoRA median IoU:", summary.get("lora_median_iou"))

    print("LoRA better count:", summary.get("lora_better_count"))
    print("Base better count:", summary.get("base_better_count"))
    print("Same IoU count:", summary.get("same_iou_count"))

    print("Base mean area ratio:", summary.get("base_mean_area_ratio"))
    print("LoRA mean area ratio:", summary.get("lora_mean_area_ratio"))
    print("Mean delta area ratio:", summary.get("mean_delta_area_ratio"))

    print("Base mean coverage:", summary.get("base_mean_coverage"))
    print("LoRA mean coverage:", summary.get("lora_mean_coverage"))
    print("Mean delta coverage:", summary.get("mean_delta_coverage"))

    print("Base mean purity:", summary.get("base_mean_purity"))
    print("LoRA mean purity:", summary.get("lora_mean_purity"))
    print("Mean delta purity:", summary.get("mean_delta_purity"))

    print("-" * 80)
    print("Saved summary:", summary_json)
    print("Saved group:", group_csv)
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare base CSV and lora CSV.",
    )

    parser.add_argument(
        "--split_jsonl",
        default=None,
        help="GT jsonl, e.g. val_quick100.jsonl / val_full.jsonl / test_conversations.jsonl.",
    )

    parser.add_argument(
        "--pred_root",
        default=None,
        help="Prediction root, e.g. outputs_eval/base_val100.",
    )

    parser.add_argument(
        "--out_csv",
        required=True,
        help="Output detail CSV.",
    )

    parser.add_argument(
        "--base_csv",
        default=None,
        help="Base evaluation CSV for compare mode.",
    )

    parser.add_argument(
        "--lora_csv",
        default=None,
        help="LoRA evaluation CSV for compare mode.",
    )

    parser.add_argument(
        "--full_image_thr",
        type=float,
        default=0.95,
        help="Pred area >= this threshold is counted as full image.",
    )

    parser.add_argument(
        "--large_box_thr",
        type=float,
        default=0.85,
        help="Pred area >= this threshold is counted as large box.",
    )

    parser.add_argument(
        "--small_box_thr",
        type=float,
        default=0.01,
        help="0 < pred area < this threshold is counted as small box.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.compare:
        if not args.base_csv or not args.lora_csv:
            raise ValueError("--compare requires --base_csv and --lora_csv")

        compare_two_models(
            base_csv=args.base_csv,
            lora_csv=args.lora_csv,
            out_csv=args.out_csv,
        )

    else:
        if not args.split_jsonl or not args.pred_root:
            raise ValueError("eval mode requires --split_jsonl and --pred_root")

        evaluate_one_model(
            split_jsonl=args.split_jsonl,
            pred_root=args.pred_root,
            out_csv=args.out_csv,
            full_image_thr=args.full_image_thr,
            large_box_thr=args.large_box_thr,
            small_box_thr=args.small_box_thr,
        )


if __name__ == "__main__":
    main()
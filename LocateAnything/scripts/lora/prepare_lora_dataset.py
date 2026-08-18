#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prepare_lora_dataset_all_in_one.py

一键完成 LocateAnything-3B LoRA 数据准备：
1. 从 images/<folder>/<image> 和 labels/<folder>/<image>.txt 读取数据
2. 按文件夹划分 train / val / test
3. 将 YOLO label 转成 LocateAnything <box><x1><y1><x2><y2></box>
4. 生成普通格式 train/val/test.json 和 train/val/test.jsonl
5. 生成 Eagle 推荐 conversations 格式 train/val/test_conversations.jsonl
6. 生成 meta_train_only_jsonl.json 和 meta_jsonl.json
7. 生成 split_info.json、skipped_samples.json、summary_splits.json
"""

import json
import random
import re
from pathlib import Path


# =========================================================
# 1. 配置区
# =========================================================

PROJECT_ROOT = Path("/data/ljy/locate_anything_project")

IMAGE_ROOT = PROJECT_ROOT / "images"
LABEL_ROOT = PROJECT_ROOT / "labels"
OUT_DIR = PROJECT_ROOT / "lora_food_plate_data"

START_FOLDER_NAME = "1331"
END_FOLDER_NAME = "1663"

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

QUESTION = (
    "Locate a single instance that matches the following description: "
    "A whole plate of food including the visible plate rim, excluding table, "
    "background, text, watermark, and other plates."
)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# 当前任务一般每张图只训练一个框
ALLOW_MULTI_LABEL = False

# 面积过滤。你之前已经确认“大框不要过滤”，所以 MAX_AREA 默认 None。
FILTER_AREA = True
MIN_AREA = 0.01
MAX_AREA = None

# 是否额外保存普通 question/answer 格式的 json/jsonl。
SAVE_PLAIN_JSON = True
SAVE_PLAIN_JSONL = True


# =========================================================
# 2. 工具函数
# =========================================================

def natural_key(name: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", name)]


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def list_range_folders(root: Path):
    if not root.exists():
        raise FileNotFoundError(f"IMAGE_ROOT not found: {root}")

    folders = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: natural_key(p.name))
    names = [p.name for p in folders]

    if START_FOLDER_NAME not in names:
        raise FileNotFoundError(f"START_FOLDER_NAME not found: {START_FOLDER_NAME}")
    if END_FOLDER_NAME not in names:
        raise FileNotFoundError(f"END_FOLDER_NAME not found: {END_FOLDER_NAME}")

    s = names.index(START_FOLDER_NAME)
    e = names.index(END_FOLDER_NAME)
    if s > e:
        raise ValueError("START_FOLDER_NAME is after END_FOLDER_NAME.")
    return folders[s:e + 1]


def split_folders(folders):
    folders = list(folders)
    random.seed(RANDOM_SEED)
    random.shuffle(folders)

    n = len(folders)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_folders = folders[:n_train]
    val_folders = folders[n_train:n_train + n_val]
    test_folders = folders[n_train + n_val:]
    return train_folders, val_folders, test_folders


def find_image(image_folder: Path, stem: str):
    for ext in IMAGE_EXTS:
        p = image_folder / f"{stem}{ext}"
        if p.exists():
            return p
        p_upper = image_folder / f"{stem}{ext.upper()}"
        if p_upper.exists():
            return p_upper
    return None


def read_yolo_label(label_path: Path):
    if not label_path.exists():
        return []

    lines = [x.strip() for x in label_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    boxes = []

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:5])
        except Exception:
            continue

        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 <= bw <= 1 and 0 <= bh <= 1):
            continue
        if bw <= 0 or bh <= 0:
            continue

        area = bw * bh
        if FILTER_AREA:
            if MIN_AREA is not None and area < MIN_AREA:
                continue
            if MAX_AREA is not None and area > MAX_AREA:
                continue

        boxes.append((class_id, xc, yc, bw, bh))

    return boxes


def yolo_to_la_box(xc, yc, bw, bh):
    x1 = int(round(max(0, min(1000, (xc - bw / 2.0) * 1000))))
    y1 = int(round(max(0, min(1000, (yc - bh / 2.0) * 1000))))
    x2 = int(round(max(0, min(1000, (xc + bw / 2.0) * 1000))))
    y2 = int(round(max(0, min(1000, (yc + bh / 2.0) * 1000))))

    xa, xb = sorted([x1, x2])
    ya, yb = sorted([y1, y2])
    if xb <= xa or yb <= ya:
        return None
    return f"<box><{xa}><{ya}><{xb}><{yb}></box>"


def make_plain_sample(image_path: Path, label_path: Path):
    boxes = read_yolo_label(label_path)
    if not boxes:
        return None, "invalid_or_filtered_label"

    if (not ALLOW_MULTI_LABEL) and len(boxes) > 1:
        return None, "multi_label_skipped"

    answers = []
    for _, xc, yc, bw, bh in boxes:
        answer = yolo_to_la_box(xc, yc, bw, bh)
        if answer is not None:
            answers.append(answer)

    if not answers:
        return None, "invalid_or_filtered_label"

    return {
        "image": str(image_path),
        "question": QUESTION,
        "answer": "".join(answers),
        "folder": image_path.parent.name,
        "image_name": image_path.name,
        "label": str(label_path),
    }, None


def plain_to_conversation(item):
    return {
        "image": item["image"],
        "conversations": [
            {"from": "human", "value": item["question"]},
            {"from": "gpt", "value": item["answer"]},
        ],
        "folder": item.get("folder", ""),
        "image_name": item.get("image_name", ""),
        "label": item.get("label", ""),
    }


def collect_samples_from_folders(folders):
    samples = []
    skipped = []

    for image_folder in folders:
        folder_name = image_folder.name
        label_folder = LABEL_ROOT / folder_name

        if not label_folder.exists():
            skipped.append({"folder": folder_name, "reason": "label_folder_not_found"})
            continue

        label_files = sorted(label_folder.glob("*.txt"), key=lambda p: natural_key(p.name))

        for label_path in label_files:
            stem = label_path.stem
            image_path = find_image(image_folder, stem)

            if image_path is None:
                skipped.append({
                    "folder": folder_name,
                    "stem": stem,
                    "reason": "image_not_found",
                    "label": str(label_path),
                })
                continue

            sample, reason = make_plain_sample(image_path, label_path)
            if sample is None:
                skipped.append({
                    "folder": folder_name,
                    "stem": stem,
                    "reason": reason or "invalid_sample",
                    "image": str(image_path),
                    "label": str(label_path),
                })
                continue

            samples.append(sample)

    return samples, skipped


def save_split_outputs(split_name: str, plain_samples):
    conv_samples = [plain_to_conversation(x) for x in plain_samples]

    if SAVE_PLAIN_JSON:
        save_json(OUT_DIR / f"{split_name}.json", plain_samples)
    if SAVE_PLAIN_JSONL:
        write_jsonl(OUT_DIR / f"{split_name}.jsonl", plain_samples)

    conv_path = OUT_DIR / f"{split_name}_conversations.jsonl"
    write_jsonl(conv_path, conv_samples)
    return conv_path, len(conv_samples)


def make_meta(train_path: Path, train_len: int, val_path: Path, val_len: int):
    meta_train_only = {
        "food_plate_train": {
            "annotation": str(train_path),
            "root": "/",
            "repeat_time": 1,
            "length": train_len,
            "visual_prompt": False,
        }
    }

    meta_with_val = {
        "food_plate_train": {
            "annotation": str(train_path),
            "root": "/",
            "repeat_time": 1,
            "length": train_len,
            "visual_prompt": False,
        },
        "food_plate_val": {
            "annotation": str(val_path),
            "root": "/",
            "repeat_time": 1,
            "length": val_len,
            "visual_prompt": False,
        },
    }

    save_json(OUT_DIR / "meta_train_only_jsonl.json", meta_train_only)
    save_json(OUT_DIR / "meta_jsonl.json", meta_with_val)


def summarize_skipped(skipped_list):
    counts = {}
    for item in skipped_list:
        reason = item.get("reason", "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


# =========================================================
# 3. 主函数
# =========================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    folders = list_range_folders(IMAGE_ROOT)
    train_folders, val_folders, test_folders = split_folders(folders)

    train_samples, train_skipped = collect_samples_from_folders(train_folders)
    val_samples, val_skipped = collect_samples_from_folders(val_folders)
    test_samples, test_skipped = collect_samples_from_folders(test_folders)

    train_conv_path, train_len = save_split_outputs("train", train_samples)
    val_conv_path, val_len = save_split_outputs("val", val_samples)
    test_conv_path, test_len = save_split_outputs("test", test_samples)

    make_meta(train_conv_path, train_len, val_conv_path, val_len)

    split_info = {
        "start_folder": START_FOLDER_NAME,
        "end_folder": END_FOLDER_NAME,
        "random_seed": RANDOM_SEED,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "train_folders": [p.name for p in train_folders],
        "val_folders": [p.name for p in val_folders],
        "test_folders": [p.name for p in test_folders],
        "num_total_folders": len(folders),
        "num_train_folders": len(train_folders),
        "num_val_folders": len(val_folders),
        "num_test_folders": len(test_folders),
        "num_train_samples": train_len,
        "num_val_samples": val_len,
        "num_test_samples": test_len,
        "question": QUESTION,
        "allow_multi_label": ALLOW_MULTI_LABEL,
        "filter_area": FILTER_AREA,
        "min_area": MIN_AREA,
        "max_area": MAX_AREA,
    }

    skipped = {
        "train_skipped": train_skipped,
        "val_skipped": val_skipped,
        "test_skipped": test_skipped,
    }

    summary = {
        "train": {"folders": len(train_folders), "samples": train_len, "skipped": summarize_skipped(train_skipped)},
        "val": {"folders": len(val_folders), "samples": val_len, "skipped": summarize_skipped(val_skipped)},
        "test": {"folders": len(test_folders), "samples": test_len, "skipped": summarize_skipped(test_skipped)},
    }

    save_json(OUT_DIR / "split_info.json", split_info)
    save_json(OUT_DIR / "skipped_samples.json", skipped)
    save_json(OUT_DIR / "summary_splits.json", summary)

    print("=" * 80)
    print("LocateAnything LoRA dataset prepared")
    print("=" * 80)
    print(f"Image root: {IMAGE_ROOT}")
    print(f"Label root: {LABEL_ROOT}")
    print(f"Output dir: {OUT_DIR}")
    print()
    print(f"Total folders: {len(folders)}")
    print(f"Train folders: {len(train_folders)}, samples: {train_len}")
    print(f"Val folders:   {len(val_folders)}, samples: {val_len}")
    print(f"Test folders:  {len(test_folders)}, samples: {test_len}")
    print()
    print("Generated:")
    for name in [
        "train_conversations.jsonl",
        "val_conversations.jsonl",
        "test_conversations.jsonl",
        "meta_train_only_jsonl.json",
        "meta_jsonl.json",
        "split_info.json",
        "skipped_samples.json",
        "summary_splits.json",
    ]:
        print(f"  {OUT_DIR / name}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

DATA_ROOT = Path("/data/ljy/locate_anything_project/lora_food_plate_data")

TRAIN_JSONL = DATA_ROOT / "train_conversations.jsonl"
VAL_JSONL = DATA_ROOT / "val_conversations.jsonl"

META_TRAIN_ONLY = DATA_ROOT / "meta_train_only_jsonl.json"
META_JSON = DATA_ROOT / "meta_jsonl.json"


def count_lines(path: Path):
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def main():
    train_len = count_lines(TRAIN_JSONL)
    val_len = count_lines(VAL_JSONL)

    meta_train_only = {
        "food_plate_train": {
            "annotation": str(TRAIN_JSONL),
            "root": "/",
            "repeat_time": 1,
            "length": train_len,
            "visual_prompt": False
        }
    }

    meta_with_val = {
        "food_plate_train": {
            "annotation": str(TRAIN_JSONL),
            "root": "/",
            "repeat_time": 1,
            "length": train_len,
            "visual_prompt": False
        },
        "food_plate_val": {
            "annotation": str(VAL_JSONL),
            "root": "/",
            "repeat_time": 1,
            "length": val_len,
            "visual_prompt": False
        }
    }

    with open(META_TRAIN_ONLY, "w", encoding="utf-8") as f:
        json.dump(meta_train_only, f, ensure_ascii=False, indent=2)

    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta_with_val, f, ensure_ascii=False, indent=2)

    print("Saved:", META_TRAIN_ONLY)
    print("Saved:", META_JSON)
    print("train length:", train_len)
    print("val length:", val_len)


if __name__ == "__main__":
    main()
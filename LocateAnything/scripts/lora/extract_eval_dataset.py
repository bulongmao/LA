#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import shutil
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_jsonl", required=True)
    parser.add_argument("--dst_root", required=True)
    parser.add_argument("--dst_jsonl", required=True)
    parser.add_argument("--copy", action="store_true", help="copy images instead of symlink")
    args = parser.parse_args()

    src_jsonl = Path(args.src_jsonl)
    dst_root = Path(args.dst_root)
    dst_jsonl = Path(args.dst_jsonl)

    dst_root.mkdir(parents=True, exist_ok=True)
    dst_jsonl.parent.mkdir(parents=True, exist_ok=True)

    new_items = []
    folders = set()

    with open(src_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)
            src_img = Path(item["image"])

            folder = item.get("folder") or src_img.parent.name
            image_name = item.get("image_name") or src_img.name

            folders.add(folder)

            dst_dir = dst_root / folder
            dst_dir.mkdir(parents=True, exist_ok=True)

            dst_img = dst_dir / image_name

            if not dst_img.exists():
                if args.copy:
                    shutil.copy2(src_img, dst_img)
                else:
                    try:
                        os.symlink(src_img, dst_img)
                    except Exception:
                        shutil.copy2(src_img, dst_img)

            item["image"] = str(dst_img)
            item["folder"] = folder
            item["image_name"] = image_name
            new_items.append(item)

    with open(dst_jsonl, "w", encoding="utf-8") as f:
        for item in new_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    folders = sorted(folders, key=lambda x: int(x) if str(x).isdigit() else x)

    print("Saved image root:", dst_root)
    print("Saved jsonl:", dst_jsonl)
    print("Num images:", len(new_items))
    print("Num folders:", len(folders))
    print("START_FOLDER_NAME =", folders[0])
    print("END_FOLDER_NAME   =", folders[-1])


if __name__ == "__main__":
    main()

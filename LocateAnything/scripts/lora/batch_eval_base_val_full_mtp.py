#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_la_food_plate_mtp_batch.py

Native MTP batched version of batch_la_food_plate.py.

Purpose:
- Replace per-image official Worker inference with locateanything_batch.generate_batch().
- Keep existing output format: outputs/<folder>/json, labels, vis.
- Keep one-box rule: multiple boxes -> largest; no valid box -> full-image fallback.
- Dispatch work by image batches instead of folders, so one large folder can also use multiple GPUs.

Prerequisite:
- The server environment must be able to import locateanything_batch.
- The model path is passed to the backend through LA3B_MODEL.
"""

import os
import re
import csv
import json
import time
import queue
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch.multiprocessing as mp
from PIL import Image, ImageDraw, ImageFont


# =========================================================
# 0. User configuration
# =========================================================

INPUT_ROOT = Path("/data/ljy/locate_anything_project/eval_images/val_full")
OUTPUT_ROOT = Path("/data/ljy/locate_anything_project/outputs_eval/base_val_full")

MODEL_PATH = "/data/ljy/locate_anything_project/models/LocateAnything-3B"

START_FOLDER_NAME = "1339"
END_FOLDER_NAME = "1647"

# Multi-GPU example: GPU_IDS = [0, 1, 2, 3]
GPU_IDS = [0, 1, 2, 3]

# Per-GPU batch size. Start from 2 or 4; increase after confirming VRAM is stable.
BATCH_SIZE = 4

OVERWRITE = True

PHRASE = "A whole plate of food including the visible plate rim, excluding table, background, text, watermark, and other plates"

# ground_single / ground_multi
GROUNDING_MODE = "ground_single"

CLASS_ID = 0
CLASS_NAME = "target_dish"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# For short grounding outputs, 512 is usually enough. Increase if raw_answer is truncated.
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.0

FULL_IMAGE_FALLBACK = True


# =========================================================
# 1. File utilities
# =========================================================

def natural_key(name: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", name)]


def list_target_folders() -> List[Path]:
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"INPUT_ROOT not found: {INPUT_ROOT}")

    folders = sorted([p for p in INPUT_ROOT.iterdir() if p.is_dir()], key=lambda p: natural_key(p.name))
    name_to_idx = {p.name: i for i, p in enumerate(folders)}

    if START_FOLDER_NAME not in name_to_idx:
        raise FileNotFoundError(f"START_FOLDER_NAME not found: {START_FOLDER_NAME}")
    if END_FOLDER_NAME not in name_to_idx:
        raise FileNotFoundError(f"END_FOLDER_NAME not found: {END_FOLDER_NAME}")

    s = name_to_idx[START_FOLDER_NAME]
    e = name_to_idx[END_FOLDER_NAME]
    if s > e:
        raise ValueError(f"START_FOLDER_NAME after END_FOLDER_NAME: {START_FOLDER_NAME} > {END_FOLDER_NAME}")

    return folders[s:e + 1]


def list_images(folder: Path) -> List[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: natural_key(p.name),
    )


def collect_images(folders: List[Path]) -> List[Path]:
    images: List[Path] = []
    for folder in folders:
        images.extend(list_images(folder))
    return images


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def build_query(phrase: str) -> str:
    if GROUNDING_MODE == "ground_multi":
        return f"Locate all the instances that match the following description: {phrase}."
    return f"Locate a single instance that matches the following description: {phrase}."


# =========================================================
# 2. Output paths
# =========================================================

def ensure_dirs():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_ROOT / "classes.txt", "w", encoding="utf-8") as f:
        f.write(CLASS_NAME + "\n")


def clean_old_logs():
    log_dir = OUTPUT_ROOT / "logs"
    if not log_dir.exists():
        return
    for p in log_dir.glob("summary_worker_*.csv"):
        try:
            p.unlink()
        except Exception:
            pass


def output_paths(image_path: Path) -> Tuple[Path, Path, Path]:
    folder_name = image_path.parent.name
    stem = image_path.stem
    folder_root = OUTPUT_ROOT / folder_name
    json_dir = folder_root / "json"
    label_dir = folder_root / "labels"
    vis_dir = folder_root / "vis"
    json_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    return json_dir / f"{stem}.json", label_dir / f"{stem}.txt", vis_dir / f"{stem}.jpg"


def output_complete(image_path: Path) -> bool:
    json_path, label_path, vis_path = output_paths(image_path)
    if not (json_path.exists() and label_path.exists() and vis_path.exists()):
        return False
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return "selected_box" in data
    except Exception:
        return False


# =========================================================
# 3. Box parsing and selection
# =========================================================

def parse_boxes(answer: str, image_width: int, image_height: int) -> List[Dict[str, float]]:
    text = str(answer)
    boxes: List[Dict[str, float]] = []

    # LocateAnything standard: <box><x1><y1><x2><y2></box>
    pattern_tag = r"<box>\s*<([\d.]+)>\s*<([\d.]+)>\s*<([\d.]+)>\s*<([\d.]+)>\s*</box>"
    for m in re.finditer(pattern_tag, text):
        box = coords_to_box([float(x) for x in m.groups()], image_width, image_height)
        if box is not None:
            boxes.append(box)

    # Compatibility: [x1, y1, x2, y2] or (x1, y1, x2, y2)
    if not boxes:
        pattern_list = r"[\[\(]\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*[\]\)]"
        for m in re.finditer(pattern_list, text):
            box = coords_to_box([float(x) for x in m.groups()], image_width, image_height)
            if box is not None:
                boxes.append(box)

    return deduplicate_boxes(boxes)


def coords_to_box(coords: List[float], image_width: int, image_height: int):
    if len(coords) != 4:
        return None

    x1, y1, x2, y2 = [float(v) for v in coords]
    max_coord = max(abs(x1), abs(y1), abs(x2), abs(y2))

    if max_coord <= 1.5:  # 0-1 normalized
        x1 *= image_width
        x2 *= image_width
        y1 *= image_height
        y2 *= image_height
    elif max_coord <= 1000:  # LocateAnything 0-1000 coordinates
        x1 = x1 / 1000.0 * image_width
        x2 = x2 / 1000.0 * image_width
        y1 = y1 / 1000.0 * image_height
        y2 = y2 / 1000.0 * image_height
    # else: already pixel coordinates

    xa, xb = sorted([x1, x2])
    ya, yb = sorted([y1, y2])

    xa = max(0.0, min(float(image_width), xa))
    xb = max(0.0, min(float(image_width), xb))
    ya = max(0.0, min(float(image_height), ya))
    yb = max(0.0, min(float(image_height), yb))

    w = xb - xa
    h = yb - ya
    area = w * h
    if w <= 1.0 or h <= 1.0 or area <= 1.0:
        return None

    return {
        "x1": xa,
        "y1": ya,
        "x2": xb,
        "y2": yb,
        "width": w,
        "height": h,
        "area": area,
        "area_ratio": area / float(image_width * image_height),
    }


def compute_iou(a: Dict[str, float], b: Dict[str, float]) -> float:
    ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a["area"] + b["area"] - inter
    return 0.0 if union <= 0 else inter / union


def deduplicate_boxes(boxes: List[Dict[str, float]], iou_threshold: float = 0.98) -> List[Dict[str, float]]:
    kept: List[Dict[str, float]] = []
    for box in boxes:
        if not any(compute_iou(box, old) >= iou_threshold for old in kept):
            kept.append(box)
    return kept


def select_one_box_or_full(boxes: List[Dict[str, float]], w: int, h: int) -> Tuple[Dict[str, float], str]:
    if boxes:
        box = dict(max(boxes, key=lambda b: b["area"]))
        box["source"] = "largest_model_box"
        return box, "largest_model_box"

    if not FULL_IMAGE_FALLBACK:
        raise RuntimeError("No valid boxes and FULL_IMAGE_FALLBACK=False")

    box = {
        "x1": 0.0,
        "y1": 0.0,
        "x2": float(w),
        "y2": float(h),
        "width": float(w),
        "height": float(h),
        "area": float(w * h),
        "area_ratio": 1.0,
        "source": "full_image_fallback",
    }
    return box, "full_image_fallback"


# =========================================================
# 4. Save json / label / visualization
# =========================================================

def save_json(path: Path, data: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_yolo(path: Path, box: Dict[str, float], w: int, h: int):
    xc = ((box["x1"] + box["x2"]) / 2.0) / w
    yc = ((box["y1"] + box["y2"]) / 2.0) / h
    bw = (box["x2"] - box["x1"]) / w
    bh = (box["y2"] - box["y1"]) / h
    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    bw = max(0.0, min(1.0, bw))
    bh = max(0.0, min(1.0, bh))
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


def draw_vis(image: Image.Image, box: Dict[str, float], path: Path, selected_rule: str):
    vis = image.copy().convert("RGB")
    draw = ImageDraw.Draw(vis)
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    line_width = max(2, int(min(image.size) * 0.004))
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=line_width)

    text = CLASS_NAME + (" [fallback]" if selected_rule == "full_image_fallback" else "")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(14, int(min(image.size) * 0.03)))
    except Exception:
        font = ImageFont.load_default()

    tb = draw.textbbox((x1, y1), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    bg = [x1, max(0, y1 - th - 6), min(image.size[0], x1 + tw + 8), y1]
    draw.rectangle(bg, fill=(255, 0, 0))
    draw.text((bg[0] + 4, bg[1] + 2), text, fill=(255, 255, 255), font=font)
    vis.save(path, quality=95)


# =========================================================
# 5. Native MTP batched inference
# =========================================================

def normalize_answers(raw_answers: Any) -> List[str]:
    if isinstance(raw_answers, list):
        return [str(x) for x in raw_answers]
    if isinstance(raw_answers, tuple):
        if len(raw_answers) > 0 and isinstance(raw_answers[0], list):
            return [str(x) for x in raw_answers[0]]
        return [str(x) for x in raw_answers]
    return [str(raw_answers)]


def run_generate_batch(pairs: List[Tuple[Image.Image, str]]) -> List[str]:
    from locateanything_batch import generate_batch
    try:
        raw_answers = generate_batch(pairs, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)
    except TypeError:
        raw_answers = generate_batch(pairs)
    return normalize_answers(raw_answers)


def process_batch(batch_paths: List[Path], rank: int, physical_gpu_id: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    query = build_query(PHRASE)

    valid_paths: List[Path] = []
    images: List[Image.Image] = []

    for image_path in batch_paths:
        json_path, label_path, vis_path = output_paths(image_path)
        base_row = {
            "image_path": str(image_path),
            "folder": image_path.parent.name,
            "image": image_path.name,
            "status": "",
            "phrase": PHRASE,
            "query": query,
            "grounding_mode": GROUNDING_MODE,
            "num_model_boxes": 0,
            "selected_rule": "",
            "runtime_sec": 0.0,
            "json_path": str(json_path),
            "label_path": str(label_path),
            "vis_path": str(vis_path),
            "worker_rank": rank,
            "gpu_id": physical_gpu_id,
            "batch_size": len(batch_paths),
            "error": "",
        }

        if (not OVERWRITE) and output_complete(image_path):
            base_row["status"] = "skipped_complete"
            rows.append(base_row)
            continue

        try:
            images.append(Image.open(image_path).convert("RGB"))
            valid_paths.append(image_path)
        except Exception:
            base_row["status"] = "failed_open_image"
            base_row["error"] = traceback.format_exc().replace("\n", "\\n")[:2000]
            rows.append(base_row)

    if not valid_paths:
        return rows

    try:
        pairs = [(img, query) for img in images]
        answers = run_generate_batch(pairs)
        if len(answers) != len(valid_paths):
            raise RuntimeError(f"generate_batch returned {len(answers)} answers, but input has {len(valid_paths)} images")
    except Exception:
        err = traceback.format_exc().replace("\n", "\\n")[:3000]
        for image_path in valid_paths:
            json_path, label_path, vis_path = output_paths(image_path)
            rows.append({
                "image_path": str(image_path),
                "folder": image_path.parent.name,
                "image": image_path.name,
                "status": "failed_generate_batch",
                "phrase": PHRASE,
                "query": query,
                "grounding_mode": GROUNDING_MODE,
                "num_model_boxes": 0,
                "selected_rule": "",
                "runtime_sec": 0.0,
                "json_path": str(json_path),
                "label_path": str(label_path),
                "vis_path": str(vis_path),
                "worker_rank": rank,
                "gpu_id": physical_gpu_id,
                "batch_size": len(batch_paths),
                "error": err,
            })
        return rows

    for image_path, image, answer in zip(valid_paths, images, answers):
        t0 = time.time()
        json_path, label_path, vis_path = output_paths(image_path)
        row = {
            "image_path": str(image_path),
            "folder": image_path.parent.name,
            "image": image_path.name,
            "status": "",
            "phrase": PHRASE,
            "query": query,
            "grounding_mode": GROUNDING_MODE,
            "num_model_boxes": 0,
            "selected_rule": "",
            "runtime_sec": 0.0,
            "json_path": str(json_path),
            "label_path": str(label_path),
            "vis_path": str(vis_path),
            "worker_rank": rank,
            "gpu_id": physical_gpu_id,
            "batch_size": len(batch_paths),
            "error": "",
        }
        try:
            w, h = image.size
            boxes = parse_boxes(answer, w, h)
            selected_box, selected_rule = select_one_box_or_full(boxes, w, h)

            json_data = {
                "image_path": str(image_path),
                "width": w,
                "height": h,
                "phrase": PHRASE,
                "query": query,
                "grounding_mode": GROUNDING_MODE,
                "raw_answer": answer,
                "num_model_boxes": len(boxes),
                "all_model_boxes": boxes,
                "selected_box": selected_box,
                "selected_rule": selected_rule,
                "worker_rank": rank,
                "gpu_id": physical_gpu_id,
                "batch_size": len(batch_paths),
                "runtime_sec": round(time.time() - t0, 4),
                "error": "",
            }

            save_json(json_path, json_data)
            save_yolo(label_path, selected_box, w, h)
            draw_vis(image, selected_box, vis_path, selected_rule)

            row["status"] = "ok"
            row["num_model_boxes"] = len(boxes)
            row["selected_rule"] = selected_rule
            row["runtime_sec"] = round(time.time() - t0, 4)
        except Exception:
            row["status"] = "failed_postprocess"
            row["error"] = traceback.format_exc().replace("\n", "\\n")[:2000]
            row["runtime_sec"] = round(time.time() - t0, 4)

        rows.append(row)

    return rows


# =========================================================
# 6. Multi-GPU workers
# =========================================================

def write_csv(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted(set().union(*(r.keys() for r in rows)))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def gpu_worker_main(rank: int, physical_gpu_id: int, task_queue: mp.Queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
    os.environ["LA3B_MODEL"] = str(MODEL_PATH)

    print(
        f"[Worker {rank}] start, physical_gpu={physical_gpu_id}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
        f"LA3B_MODEL={os.environ.get('LA3B_MODEL')}",
        flush=True,
    )

    try:
        from locateanything_batch import load
        load()
        print(f"[Worker {rank}] model loaded on GPU {physical_gpu_id}", flush=True)
    except Exception:
        err = traceback.format_exc().replace("\n", "\\n")[:3000]
        summary_path = OUTPUT_ROOT / "logs" / f"summary_worker_{rank}_gpu{physical_gpu_id}.csv"
        write_csv(summary_path, [{"worker_rank": rank, "gpu_id": physical_gpu_id, "status": "failed_load_model", "error": err}])
        print(f"[Worker {rank}] failed_load_model: {err}", flush=True)
        return

    rows: List[Dict[str, Any]] = []

    while True:
        try:
            batch_strs = task_queue.get_nowait()
        except queue.Empty:
            break

        batch_paths = [Path(p) for p in batch_strs]
        if not batch_paths:
            continue

        print(
            f"[Worker {rank} | GPU {physical_gpu_id}] batch={len(batch_paths)} "
            f"first={batch_paths[0].parent.name}/{batch_paths[0].name}",
            flush=True,
        )
        rows.extend(process_batch(batch_paths, rank, physical_gpu_id))

    summary_path = OUTPUT_ROOT / "logs" / f"summary_worker_{rank}_gpu{physical_gpu_id}.csv"
    write_csv(summary_path, rows)
    print(f"[Worker {rank}] done, rows={len(rows)}", flush=True)


def merge_summaries():
    all_rows: List[Dict[str, Any]] = []
    for p in sorted((OUTPUT_ROOT / "logs").glob("summary_worker_*.csv")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                all_rows.extend(list(csv.DictReader(f)))
        except Exception:
            pass
    write_csv(OUTPUT_ROOT / "summary.csv", all_rows)


def audit_missing(folders: List[Path]):
    rows: List[Dict[str, Any]] = []
    for folder in folders:
        for img in list_images(folder):
            json_path, label_path, vis_path = output_paths(img)
            missing = []
            if not json_path.exists():
                missing.append("json")
            if not label_path.exists():
                missing.append("label")
            if not vis_path.exists():
                missing.append("vis")
            if missing:
                rows.append({"folder": folder.name, "image": img.name, "missing": "|".join(missing), "image_path": str(img)})
    write_csv(OUTPUT_ROOT / "missing_outputs.csv", rows)
    return rows


# =========================================================
# 7. Main
# =========================================================

def main():
    ensure_dirs()
    clean_old_logs()

    folders = list_target_folders()
    all_images = collect_images(folders)
    pending_images = all_images if OVERWRITE else [p for p in all_images if not output_complete(p)]
    batches = chunk_list(pending_images, BATCH_SIZE)

    print("=" * 100)
    print(f"INPUT_ROOT       : {INPUT_ROOT}")
    print(f"OUTPUT_ROOT      : {OUTPUT_ROOT}")
    print(f"MODEL_PATH       : {MODEL_PATH}")
    print(f"FOLDER_RANGE     : {START_FOLDER_NAME} -> {END_FOLDER_NAME}")
    print(f"NUM_FOLDERS      : {len(folders)}")
    print(f"NUM_IMAGES       : {len(all_images)}")
    print(f"PENDING_IMAGES   : {len(pending_images)}")
    print(f"NUM_BATCHES      : {len(batches)}")
    print(f"GPU_IDS          : {GPU_IDS}")
    print(f"BATCH_SIZE       : {BATCH_SIZE}")
    print(f"PHRASE           : {PHRASE}")
    print(f"GROUNDING_MODE   : {GROUNDING_MODE}")
    print(f"MAX_NEW_TOKENS   : {MAX_NEW_TOKENS}")
    print(f"TEMPERATURE      : {TEMPERATURE}")
    print("=" * 100)

    if not pending_images:
        print("[Exit] No pending images.", flush=True)
        return

    task_queue = mp.Queue()
    for batch in batches:
        task_queue.put([str(p) for p in batch])

    processes = []
    for rank, gpu_id in enumerate(GPU_IDS):
        p = mp.Process(target=gpu_worker_main, args=(rank, gpu_id, task_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    merge_summaries()
    missing = audit_missing(folders)

    if missing:
        print(f"[Audit] Missing outputs: {len(missing)}. See {OUTPUT_ROOT / 'missing_outputs.csv'}")
    else:
        print("[Audit] No missing outputs.")

    print(f"[Done] Summary: {OUTPUT_ROOT / 'summary.csv'}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

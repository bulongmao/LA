import os
import re
import csv
import json
import time
import queue
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.multiprocessing as mp
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoTokenizer, AutoProcessor


# =========================================================
# 0. 只需要改这里
# =========================================================

INPUT_ROOT = Path("/data/ljy/locate_anything_project/images")
OUTPUT_ROOT = Path("/data/ljy/locate_anything_project/outputs_eval")

MODEL_PATH = "/data/ljy/locate_anything_project/models/LocateAnything-3B"

START_FOLDER_NAME = "1416"
END_FOLDER_NAME = "1416"

GPU_IDS = [0,1,2,3]

OVERWRITE = True

# 官网 demo 输入框里等价填写的内容
PHRASE = "A whole plate of food including the visible plate rim, excluding table, background, text, watermark, and other plates"

# 推荐先用 ground_single，最符合“只要一份整体”的需求
# 可选：ground_single / ground_multi
GROUNDING_MODE = "ground_single"

CLASS_ID = 0
CLASS_NAME = "target_dish"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 官方建议 hybrid，平衡速度与准确性
GENERATION_MODE = "hybrid"

# 官方建议最大生成 token 足够大，避免输出被截断。
# 如果确认每张图只需 1 个框，也可以改成 2048 提速。
MAX_NEW_TOKENS = 8192

TORCH_DTYPE = "bfloat16"   # RTX 4090 一般可用；报错再改 float16


# =========================================================
# 1. 官方风格 Worker
# =========================================================

class LocateAnythingWorker:
    """
    基本照官方推荐 Worker 写法。
    关键点：
    1. 模型只加载一次；
    2. 使用官方 ground_single / ground_multi prompt template；
    3. 使用官方 parse_boxes 解析 <box><x1><y1><x2><y2></box>。
    """

    def __init__(self, model_path: str, device: str = "cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype

        print(f"[Model] Loading: {model_path} on {device}", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device).eval()

        print(f"[Model] Loaded on {device}", flush=True)

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        question: str,
        generation_mode: str = GENERATION_MODE,
        max_new_tokens: int = MAX_NEW_TOKENS,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = self.processor.py_apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        images, videos = self.processor.process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            return_tensors="pt",
        ).to(self.device)

        pixel_values = inputs["pixel_values"].to(self.dtype)
        input_ids = inputs["input_ids"]
        image_grid_hws = inputs.get("image_grid_hws", None)

        response = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            image_grid_hws=image_grid_hws,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode=generation_mode,

            # 尽量贴近官方 worker 默认采样参数。
            # 如果想完全确定性，可把 do_sample=False，并删除 temperature/top_p/repetition_penalty。
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=verbose,
        )

        answer = response[0] if isinstance(response, tuple) else response

        result = {"answer": str(answer)}

        if isinstance(response, tuple) and len(response) >= 3:
            result["history"] = response[1]
            result["stats"] = response[2]

        return result

    def ground_single(self, image: Image.Image, phrase: str) -> Dict[str, Any]:
        prompt = f"Locate a single instance that matches the following description: {phrase}."
        return self.predict(image, prompt)

    def ground_multi(self, image: Image.Image, phrase: str) -> Dict[str, Any]:
        prompt = f"Locate all the instances that match the following description: {phrase}."
        return self.predict(image, prompt)

    @staticmethod
    def parse_boxes(answer: str, image_width: int, image_height: int) -> List[Dict[str, float]]:
        """
        官方输出坐标是 [0, 1000] 归一化整数：
        <box><x1><y1><x2><y2></box>
        """
        boxes = []

        for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", str(answer)):
            x1, y1, x2, y2 = [int(g) for g in m.groups()]

            px1 = x1 / 1000.0 * image_width
            py1 = y1 / 1000.0 * image_height
            px2 = x2 / 1000.0 * image_width
            py2 = y2 / 1000.0 * image_height

            xa, xb = sorted([px1, px2])
            ya, yb = sorted([py1, py2])

            xa = max(0.0, min(float(image_width), xa))
            xb = max(0.0, min(float(image_width), xb))
            ya = max(0.0, min(float(image_height), ya))
            yb = max(0.0, min(float(image_height), yb))

            w = xb - xa
            h = yb - ya
            area = w * h

            if w > 1 and h > 1 and area > 1:
                boxes.append(
                    {
                        "x1": xa,
                        "y1": ya,
                        "x2": xb,
                        "y2": yb,
                        "width": w,
                        "height": h,
                        "area": area,
                        "area_ratio": area / float(image_width * image_height),
                    }
                )

        return boxes


# =========================================================
# 2. 工具函数
# =========================================================

def get_torch_dtype(name: str):
    name = name.lower()
    if name in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if name in ["fp16", "float16", "half"]:
        return torch.float16
    if name in ["fp32", "float32"]:
        return torch.float32
    raise ValueError(f"Unsupported TORCH_DTYPE: {name}")


def natural_key(name: str):
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", name)
    ]


def list_target_folders() -> List[Path]:
    folders = [p for p in INPUT_ROOT.iterdir() if p.is_dir()]
    folders = sorted(folders, key=lambda p: natural_key(p.name))

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
        [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ],
        key=lambda p: natural_key(p.name),
    )


def ensure_dirs():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "logs").mkdir(parents=True, exist_ok=True)

    # 全局 classes.txt 仍然保留一份
    with open(OUTPUT_ROOT / "classes.txt", "w", encoding="utf-8") as f:
        f.write(CLASS_NAME + "\n")


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

    json_path = json_dir / f"{stem}.json"
    label_path = label_dir / f"{stem}.txt"
    vis_path = vis_dir / f"{stem}.jpg"

    return json_path, label_path, vis_path


def output_complete(image_path: Path) -> bool:
    json_path, label_path, vis_path = output_paths(image_path)
    return json_path.exists() and label_path.exists() and vis_path.exists()


def select_one_box_or_full(boxes: List[Dict[str, float]], w: int, h: int) -> Tuple[Dict[str, float], str]:
    if boxes:
        box = max(boxes, key=lambda b: b["area"])
        box = dict(box)
        box["source"] = "largest_model_box"
        return box, "largest_model_box"

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


def save_json(path: Path, data: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_yolo(path: Path, box: Dict[str, float], w: int, h: int):
    xc = ((box["x1"] + box["x2"]) / 2.0) / w
    yc = ((box["y1"] + box["y2"]) / 2.0) / h
    bw = (box["x2"] - box["x1"]) / w
    bh = (box["y2"] - box["y1"]) / h

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


def draw_vis(image: Image.Image, box: Dict[str, float], path: Path, selected_rule: str):
    vis = image.copy().convert("RGB")
    draw = ImageDraw.Draw(vis)

    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    line_width = max(2, int(min(image.size) * 0.004))

    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=line_width)

    text = CLASS_NAME
    if selected_rule == "full_image_fallback":
        text += " [fallback]"

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


def write_csv(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = sorted(set().union(*(r.keys() for r in rows)))

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =========================================================
# 3. 单图 / 单文件夹
# =========================================================

def process_image(worker: LocateAnythingWorker, image_path: Path) -> Dict[str, Any]:
    t0 = time.time()

    json_path, label_path, vis_path = output_paths(image_path)

    rec = {
        "image_path": str(image_path),
        "folder": image_path.parent.name,
        "image": image_path.name,
        "status": "",
        "phrase": PHRASE,
        "grounding_mode": GROUNDING_MODE,
        "num_model_boxes": 0,
        "selected_rule": "",
        "runtime_sec": 0.0,
        "json_path": str(json_path),
        "label_path": str(label_path),
        "vis_path": str(vis_path),
        "error": "",
    }

    if (not OVERWRITE) and output_complete(image_path):
        rec["status"] = "skipped_complete"
        return rec

    try:
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        if GROUNDING_MODE == "ground_multi":
            result = worker.ground_multi(image, PHRASE)
        else:
            result = worker.ground_single(image, PHRASE)

        answer = result["answer"]
        boxes = worker.parse_boxes(answer, w, h)

        selected_box, selected_rule = select_one_box_or_full(boxes, w, h)

        json_data = {
            "image_path": str(image_path),
            "width": w,
            "height": h,
            "phrase": PHRASE,
            "grounding_mode": GROUNDING_MODE,
            "raw_answer": answer,
            "num_model_boxes": len(boxes),
            "all_model_boxes": boxes,
            "selected_box": selected_box,
            "selected_rule": selected_rule,
            "runtime_sec": round(time.time() - t0, 4),
            "error": "",
        }

        save_json(json_path, json_data)
        save_yolo(label_path, selected_box, w, h)
        draw_vis(image, selected_box, vis_path, selected_rule)

        rec["status"] = "ok"
        rec["num_model_boxes"] = len(boxes)
        rec["selected_rule"] = selected_rule
        rec["runtime_sec"] = round(time.time() - t0, 4)

    except Exception:
        rec["status"] = "failed"
        rec["error"] = traceback.format_exc().replace("\n", "\\n")[:2000]
        rec["runtime_sec"] = round(time.time() - t0, 4)

    return rec


def process_folder(worker: LocateAnythingWorker, folder: Path) -> List[Dict[str, Any]]:
    rows = []
    imgs = list_images(folder)

    if not imgs:
        return [{"folder": folder.name, "image": "", "status": "no_images"}]

    for img_path in imgs:
        rows.append(process_image(worker, img_path))

    return rows


# =========================================================
# 4. 多 GPU
# =========================================================

def gpu_worker_main(rank: int, physical_gpu_id: int, task_queue: mp.Queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)

    dtype = get_torch_dtype(TORCH_DTYPE)
    worker = LocateAnythingWorker(
        MODEL_PATH,
        device="cuda:0",
        dtype=dtype,
    )

    rows = []

    while True:
        try:
            folder_str = task_queue.get_nowait()
        except queue.Empty:
            break

        folder = Path(folder_str)
        print(f"[Worker {rank} | GPU {physical_gpu_id}] {folder.name}", flush=True)

        try:
            rows.extend(process_folder(worker, folder))
        except Exception:
            rows.append(
                {
                    "folder": folder.name,
                    "image": "",
                    "status": "folder_failed",
                    "error": traceback.format_exc().replace("\n", "\\n")[:2000],
                }
            )

    summary_path = OUTPUT_ROOT / "logs" / f"summary_worker_{rank}_gpu{physical_gpu_id}.csv"
    write_csv(summary_path, rows)

    print(f"[Worker {rank}] done, rows={len(rows)}", flush=True)


def merge_summaries():
    all_rows = []
    for p in sorted((OUTPUT_ROOT / "logs").glob("summary_worker_*.csv")):
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            all_rows.extend(list(reader))

    write_csv(OUTPUT_ROOT / "summary.csv", all_rows)


def audit_missing(folders: List[Path]):
    rows = []
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
                rows.append(
                    {
                        "folder": folder.name,
                        "image": img.name,
                        "missing": "|".join(missing),
                        "image_path": str(img),
                    }
                )
    write_csv(OUTPUT_ROOT / "missing_outputs.csv", rows)
    return rows


# =========================================================
# 5. main
# =========================================================

def main():
    ensure_dirs()
    folders = list_target_folders()

    print("=" * 100)
    print(f"INPUT_ROOT       : {INPUT_ROOT}")
    print(f"OUTPUT_ROOT      : {OUTPUT_ROOT}")
    print(f"MODEL_PATH       : {MODEL_PATH}")
    print(f"FOLDER_RANGE     : {START_FOLDER_NAME} -> {END_FOLDER_NAME}")
    print(f"NUM_FOLDERS      : {len(folders)}")
    print(f"GPU_IDS          : {GPU_IDS}")
    print(f"PHRASE           : {PHRASE}")
    print(f"GROUNDING_MODE   : {GROUNDING_MODE}")
    print(f"GENERATION_MODE  : {GENERATION_MODE}")
    print(f"MAX_NEW_TOKENS   : {MAX_NEW_TOKENS}")
    print("=" * 100)

    task_queue = mp.Queue()
    for folder in folders:
        task_queue.put(str(folder))

    processes = []
    for rank, gpu_id in enumerate(GPU_IDS):
        p = mp.Process(
            target=gpu_worker_main,
            args=(rank, gpu_id, task_queue),
        )
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

import os

# ============================================================
# LocateAnything：food + container 直接并集版本
#
# 功能：
# 1. 第一次推理：检测目标食物 food
# 2. 第二次推理：检测直接承载目标食物的容器 container
# 3. 直接对 food 和 container 做并集，得到最终 merged_box
#
# 规则：
# - 若同时有 food 和 container：最终框 = union(food, container)
# - 若只有 food：最终框 = food
# - 若只有 container：最终框 = container
# - 若都没有：不输出框
#
# 输出：
# - json/      每张图完整调试结果
# - vis/       可视化图（蓝色 food，绿色 container，红色 merged_box）
# - labels/    YOLO 标签，仅输出最终 merged_box
# - classes.txt
# - summary.csv
# ============================================================

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import re
import csv
import json
import time
import traceback
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoTokenizer, AutoProcessor


# ============================================================
# 1. 路径配置
# ============================================================
MODEL_PATH = "/data/ljy/locate_anything_project/models/LocateAnything-3B"

# single：只处理 SINGLE_INPUT_DIR
# all：递归处理 IMAGE_ROOT 下所有含图片的文件夹
RUN_MODE = "single"

SINGLE_INPUT_DIR = "/data/ljy/locate_anything_project/images/1609"
IMAGE_ROOT = "/data/ljy/locate_anything_project/images"

SINGLE_OUTPUT_ROOT = "/data/ljy/locate_anything_project/outputs/1609"
ALL_OUTPUT_ROOT = "/data/ljy/locate_anything_project/outputs"

# ============================================================
# 多卡并行配置
# 每个进程只处理自己负责的一部分图片
# ============================================================
WORKER_ID = int(os.environ.get("WORKER_ID", "0"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "1"))


# ============================================================
# 2. 目标类别配置
# ============================================================
TARGET_DISH_NAME = "matcha mille crepe cake"

TARGET_DISH_ALIASES = (
    "matcha mille crepe cake, "
    "matcha crepe cake, "
    "green tea mille crepe cake, "
    "matcha layered crepe cake, "
    "slice of matcha mille crepe cake, "
    "green matcha crepe cake slice with cream layers, "
    "layered green tea crepe cake with visible cream, "
    "one or more servings of matcha mille crepe cake in the image, "
    "all visible matcha mille crepe cake servings, including the cake slice, cream layers,"
)

# 最终 merged_box 的 YOLO 类别
YOLO_CLASS_ID = 0
YOLO_CLASS_NAME = TARGET_DISH_NAME

FOOD_LABEL = "food"
CONTAINER_LABEL = "container"
FINAL_LABEL = TARGET_DISH_NAME


# ============================================================
# 3. 推理配置
# ============================================================
DEVICE = "cuda:0"
DTYPE = torch.bfloat16

GENERATION_MODE = "hybrid"
MAX_NEW_TOKENS = 128
DO_SAMPLE = False

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# 是否在可视化图里同时显示 food / container 调试框
DRAW_DEBUG_BOXES = False


# ============================================================
# 4. Prompt：分别检测食物与容器
# ============================================================
FOOD_PROMPT = (
    f"Locate the visible target food: {TARGET_DISH_ALIASES}. "
    f"The bounding box should tightly cover only the target food, "
    f"and should exclude visible text, numbers, logos, labels, menus, printed paper, and background areas. "
    f"Return exactly one bounding box around the main visible {TARGET_DISH_NAME} serving. "
    f"The box must contain visible {TARGET_DISH_NAME}. "
    f"Return only one bounding box."
)

CONTAINER_PROMPT = (
    f"Locate the plate, bowl, tray, box, or container that directly holds the target food: {TARGET_DISH_ALIASES}. "
    f"Return exactly one bounding box around the directly supporting container. "
    f"If the target food is on a flat plate or shallow tray, include the visible outer edge of that plate or tray, even if the rim is thin or partly covered by the food. "
    f"Do not search for containers independently from the target food. "
    f"And should exclude visible text, numbers, logos, labels, menus, printed paper, and background areas. "
    f"Return only one bounding box."
)


# ============================================================
# 5. LocateAnything Worker
# ============================================================
class LocateAnythingWorker:
    def __init__(self, model_path: str, device: str = "cuda:0", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")

        print("[INFO] CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("[INFO] torch visible gpu count =", torch.cuda.device_count())
        print("[INFO] using logical device =", self.device)
        print("[INFO] logical cuda:0 name =", torch.cuda.get_device_name(0))
        print("[INFO] dtype =", self.dtype)

        print("[INFO] loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        print("[INFO] loading processor...")
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        print("[INFO] loading model...")
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device).eval()

        print("[INFO] model loaded successfully.")

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        question: str,
        generation_mode: str = "hybrid",
        max_new_tokens: int = 96,
        verbose: bool = False,
    ) -> str:
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
            add_generation_prompt=True
        )

        images, videos = self.processor.process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            return_tensors="pt"
        ).to(self.device)

        pixel_values = inputs["pixel_values"].to(self.dtype)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        image_grid_hws = inputs.get("image_grid_hws", None)

        response = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_hws=image_grid_hws,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode=generation_mode,
            do_sample=DO_SAMPLE,
            repetition_penalty=1.2,
            verbose=verbose,
        )

        answer = response[0] if isinstance(response, tuple) else response
        return str(answer)


# ============================================================
# 6. 框解析：支持 <box> 和 JSON
# ============================================================
def add_box_from_xyxy(
    boxes: list,
    x1,
    y1,
    x2,
    y2,
    image_width: int,
    image_height: int,
    box_id: int = 0,
):
    try:
        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)
    except Exception:
        return

    # 0~1 归一化坐标
    if 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1:
        px1 = x1 * image_width
        py1 = y1 * image_height
        px2 = x2 * image_width
        py2 = y2 * image_height

    # 0~1000 LocateAnything 坐标
    elif 0 <= x1 <= 1000 and 0 <= y1 <= 1000 and 0 <= x2 <= 1000 and 0 <= y2 <= 1000:
        px1 = x1 / 1000 * image_width
        py1 = y1 / 1000 * image_height
        px2 = x2 / 1000 * image_width
        py2 = y2 / 1000 * image_height

    # 像素坐标
    else:
        px1, py1, px2, py2 = x1, y1, x2, y2

    px1 = max(0, min(px1, image_width))
    py1 = max(0, min(py1, image_height))
    px2 = max(0, min(px2, image_width))
    py2 = max(0, min(py2, image_height))

    if px2 <= px1 or py2 <= py1:
        return

    bw = px2 - px1
    bh = py2 - py1

    boxes.append({
        "box_id": box_id,
        "x1": round(px1, 2),
        "y1": round(py1, 2),
        "x2": round(px2, 2),
        "y2": round(py2, 2),
        "width": round(bw, 2),
        "height": round(bh, 2),
        "area": round(bw * bh, 2),
        "aspect_ratio": round(bw / bh, 3),
    })


def parse_boxes(answer: str, image_width: int, image_height: int) -> list[dict]:
    boxes = []

    # 1. LocateAnything 原生格式：<box><x1><y1><x2><y2></box>
    pattern = r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>"
    for idx, m in enumerate(re.finditer(pattern, answer)):
        x1, y1, x2, y2 = m.groups()
        add_box_from_xyxy(
            boxes,
            x1,
            y1,
            x2,
            y2,
            image_width,
            image_height,
            box_id=idx,
        )

    if boxes:
        return boxes

    # 2. JSON 数组格式：[{"label":"...","x1":0.1,"y1":0.2,"x2":0.8,"y2":0.9}]
    text = answer.replace("<|im_end|>", "").strip()

    try:
        data = json.loads(text)
    except Exception:
        match = re.search(r"\[\s*\{.*?\}\s*\]", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                data = []
        else:
            data = []

    if isinstance(data, list):
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue

            if all(k in item for k in ["x1", "y1", "x2", "y2"]):
                add_box_from_xyxy(
                    boxes,
                    item["x1"],
                    item["y1"],
                    item["x2"],
                    item["y2"],
                    image_width,
                    image_height,
                    box_id=idx,
                )

    return boxes


# ============================================================
# 7. 选择框与并集逻辑
# ============================================================
def select_first_box(boxes: list[dict], role: str):
    if not boxes:
        return None

    b = dict(boxes[0])
    b["role"] = role
    return b


def box_area(box: dict) -> float:
    return max(0.0, box["x2"] - box["x1"]) * max(0.0, box["y2"] - box["y1"])


def box_area_ratio_to_image(box: dict, image_width: int, image_height: int) -> float:
    return box_area(box) / (image_width * image_height + 1e-6)


def union_box(a: dict, b: dict) -> dict:
    x1 = min(a["x1"], b["x1"])
    y1 = min(a["y1"], b["y1"])
    x2 = max(a["x2"], b["x2"])
    y2 = max(a["y2"], b["y2"])

    bw = x2 - x1
    bh = y2 - y1

    return {
        "box_id": 0,
        "x1": round(x1, 2),
        "y1": round(y1, 2),
        "x2": round(x2, 2),
        "y2": round(y2, 2),
        "width": round(bw, 2),
        "height": round(bh, 2),
        "area": round(bw * bh, 2),
        "aspect_ratio": round(bw / bh, 3) if bh > 0 else 0,
    }

def create_full_image_box(image_width: int, image_height: int) -> dict:
    """
    当 food 和 container 都没有有效框时，兜底输出整张图作为目标框。
    """
    x1 = 0.0
    y1 = 0.0
    x2 = float(image_width)
    y2 = float(image_height)

    bw = x2 - x1
    bh = y2 - y1

    return {
        "box_id": 0,
        "x1": round(x1, 2),
        "y1": round(y1, 2),
        "x2": round(x2, 2),
        "y2": round(y2, 2),
        "width": round(bw, 2),
        "height": round(bh, 2),
        "area": round(bw * bh, 2),
        "aspect_ratio": round(bw / bh, 3) if bh > 0 else 0,
        "post_mode": "full_image_fallback",
        "box_status": "full_image_fallback_box",
        "area_ratio": 1.0,
    }

def build_final_merged_box(food_box, container_box, image_width: int, image_height: int):
    """
    直接合并规则：
    - 两者都有：并集
    - 仅 food：返回 food
    - 仅 container：返回 container
    - 两者都没有：返回全图框
    """
    if food_box is not None and container_box is not None:
        final_box = union_box(food_box, container_box)
        final_box["post_mode"] = "direct_union_food_container"
        final_box["box_status"] = "food_container_union_box"
        final_box["area_ratio"] = round(box_area_ratio_to_image(final_box, image_width, image_height), 4)
        return final_box, "food_container_union_box"

    if food_box is not None:
        final_box = dict(food_box)
        final_box["post_mode"] = "food_only_fallback"
        final_box["box_status"] = "food_only_box"
        final_box["area_ratio"] = round(box_area_ratio_to_image(final_box, image_width, image_height), 4)
        return final_box, "food_only_box"

    if container_box is not None:
        final_box = dict(container_box)
        final_box["post_mode"] = "container_only_fallback"
        final_box["box_status"] = "container_only_box"
        final_box["area_ratio"] = round(box_area_ratio_to_image(final_box, image_width, image_height), 4)
        return final_box, "container_only_box"

    # 关键修改：food 和 container 都没有时，兜底输出整张图
    final_box = create_full_image_box(image_width, image_height)
    return final_box, "full_image_fallback_box"


# ============================================================
# 8. YOLO 输出
# ============================================================
def convert_xyxy_to_yolo(box: dict, image_width: int, image_height: int):
    x1 = float(box["x1"])
    y1 = float(box["y1"])
    x2 = float(box["x2"])
    y2 = float(box["y2"])

    x1 = max(0.0, min(x1, image_width))
    y1 = max(0.0, min(y1, image_height))
    x2 = max(0.0, min(x2, image_width))
    y2 = max(0.0, min(y2, image_height))

    bw = x2 - x1
    bh = y2 - y1

    if bw <= 0 or bh <= 0:
        return None

    x_center = ((x1 + x2) / 2.0) / image_width
    y_center = ((y1 + y2) / 2.0) / image_height
    box_width = bw / image_width
    box_height = bh / image_height

    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    box_width = max(0.0, min(1.0, box_width))
    box_height = max(0.0, min(1.0, box_height))

    return x_center, y_center, box_width, box_height


def save_yolo_label(
    final_box,
    image_width: int,
    image_height: int,
    txt_path: str,
) -> int:
    lines = []

    if final_box is not None:
        yolo_box = convert_xyxy_to_yolo(final_box, image_width, image_height)
        if yolo_box is not None:
            x_center, y_center, box_width, box_height = yolo_box
            lines.append(
                f"{YOLO_CLASS_ID} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
            )

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return len(lines)


# ============================================================
# 9. 可视化
# ============================================================
def draw_one_box(draw, box, label, color, font):
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

    tb = draw.textbbox((x1, y1), label, font=font)
    tw = tb[2] - tb[0]
    th = tb[3] - tb[1]

    text_x = x1
    text_y = max(0, y1 - th - 6)

    draw.rectangle(
        [text_x, text_y, text_x + tw + 8, text_y + th + 6],
        fill=color
    )
    draw.text((text_x + 4, text_y + 3), label, fill="white", font=font)


def draw_boxes(image: Image.Image, final_box=None, food_box=None, container_box=None) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    if DRAW_DEBUG_BOXES:
        if food_box is not None:
            draw_one_box(draw, food_box, FOOD_LABEL, "blue", font)

        if container_box is not None:
            draw_one_box(draw, container_box, CONTAINER_LABEL, "green", font)

    if final_box is not None:
        draw_one_box(draw, final_box, FINAL_LABEL, "red", font)

    return img


# ============================================================
# 10. 文件夹处理
# ============================================================
def collect_images(input_dir: Path) -> list[Path]:
    all_images = sorted([
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ])

    # 多进程分片：第 WORKER_ID 个进程只处理 index % NUM_WORKERS == WORKER_ID 的图片
    shard_images = [
        p for idx, p in enumerate(all_images)
        if idx % NUM_WORKERS == WORKER_ID
    ]

    return shard_images

def collect_image_folders(image_root: str) -> list[Path]:
    root = Path(image_root)
    folders = []

    if collect_images(root):
        folders.append(root)

    for folder in sorted([p for p in root.rglob("*") if p.is_dir()]):
        if collect_images(folder):
            folders.append(folder)

    return folders


def prepare_output_dirs(output_root: Path):
    json_dir = output_root / "json"
    vis_dir = output_root / "vis"
    label_dir = output_root / "labels"
    classes_txt = output_root / "classes.txt"
    summary_csv = output_root / f"summary_worker_{WORKER_ID}.csv"

    json_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    with open(classes_txt, "w", encoding="utf-8") as f:
        f.write(f"{YOLO_CLASS_NAME}\n")

    return json_dir, vis_dir, label_dir, classes_txt, summary_csv


def process_one_folder(worker: LocateAnythingWorker, input_dir: Path, output_root: Path):
    json_dir, vis_dir, label_dir, classes_txt, summary_csv = prepare_output_dirs(output_root)
    image_paths = collect_images(input_dir)

    print("=" * 100)
    print("[INFO] input dir   :", input_dir)
    print("[INFO] output root :", output_root)
    print("[INFO] json dir    :", json_dir)
    print("[INFO] vis dir     :", vis_dir)
    print("[INFO] labels dir  :", label_dir)
    print("[INFO] classes txt :", classes_txt)
    print("[INFO] summary csv :", summary_csv)
    print("[INFO] image count :", len(image_paths))

    summary_rows = []

    for idx, img_path in enumerate(image_paths, start=1):
        print("-" * 100)
        print(f"[INFO] [{idx}/{len(image_paths)}] processing: {img_path.name}")

        try:
            image = Image.open(img_path).convert("RGB")
            w, h = image.size

            start = time.time()

            food_answer = worker.predict(
                image=image,
                question=FOOD_PROMPT,
                generation_mode=GENERATION_MODE,
                max_new_tokens=MAX_NEW_TOKENS,
                verbose=False,
            )

            container_answer = worker.predict(
                image=image,
                question=CONTAINER_PROMPT,
                generation_mode=GENERATION_MODE,
                max_new_tokens=MAX_NEW_TOKENS,
                verbose=False,
            )

            food_boxes = parse_boxes(food_answer, w, h)
            container_boxes = parse_boxes(container_answer, w, h)

            food_box = select_first_box(food_boxes, "food")
            container_box = select_first_box(container_boxes, "container")

            final_box, box_status = build_final_merged_box(
                food_box=food_box,
                container_box=container_box,
                image_width=w,
                image_height=h,
            )

            elapsed = time.time() - start

            yolo_path = label_dir / f"{img_path.stem}.txt"
            num_yolo_boxes = save_yolo_label(
                final_box=final_box,
                image_width=w,
                image_height=h,
                txt_path=str(yolo_path),
            )

            vis_img = draw_boxes(
                image=image,
                final_box=final_box,
                food_box=food_box,
                container_box=container_box,
            )

            vis_path = vis_dir / f"{img_path.stem}_vis.jpg"
            vis_img.save(vis_path)

            json_path = json_dir / f"{img_path.stem}.json"

            json_data = {
                "image_name": img_path.name,
                "image_path": str(img_path),
                "image_size": {
                    "width": w,
                    "height": h,
                },
                "target_dish_name": TARGET_DISH_NAME,
                "target_dish_aliases": TARGET_DISH_ALIASES,
                "classes_txt": str(classes_txt),
                "yolo_classes": {
                    "0": YOLO_CLASS_NAME,
                },
                "generation_mode": GENERATION_MODE,
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": DO_SAMPLE,
                "elapsed_seconds": round(elapsed, 3),

                "food_prompt": FOOD_PROMPT,
                "container_prompt": CONTAINER_PROMPT,
                "food_answer": food_answer,
                "container_answer": container_answer,

                "food_boxes": food_boxes,
                "container_boxes": container_boxes,
                "selected_food_box": food_box,
                "selected_container_box": container_box,

                "merged_box": final_box,
                "boxes": [final_box] if final_box is not None else [],
                "box_status": box_status,

                "num_food_boxes": len(food_boxes),
                "num_container_boxes": len(container_boxes),
                "num_boxes": 1 if final_box is not None else 0,

                "union_strategy": {
                    "rule": "direct_union",
                    "description": "if both food and container exist, final_box = union(food, container); else fallback to existing one",
                    "draw_debug_boxes": DRAW_DEBUG_BOXES,
                },

                "yolo_path": str(yolo_path),
                "vis_path": str(vis_path),
                "json_path": str(json_path),
                "num_yolo_boxes": num_yolo_boxes,
                "status": f"success_{box_status}" if final_box is not None else box_status,
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)

            summary_rows.append({
                "image_name": img_path.name,
                "width": w,
                "height": h,
                "num_food_boxes": len(food_boxes),
                "num_container_boxes": len(container_boxes),
                "has_food_box": food_box is not None,
                "has_container_box": container_box is not None,
                "num_boxes": 1 if final_box is not None else 0,
                "num_yolo_boxes": num_yolo_boxes,
                "box_status": box_status,
                "elapsed_seconds": round(elapsed, 3),
                "json_path": str(json_path),
                "vis_path": str(vis_path),
                "yolo_path": str(yolo_path),
                "status": f"success_{box_status}" if final_box is not None else box_status,
            })

            print(f"[INFO] food boxes      : {len(food_boxes)}")
            print(f"[INFO] container boxes : {len(container_boxes)}")
            print(f"[INFO] selected food   : {food_box is not None}")
            print(f"[INFO] selected cont.  : {container_box is not None}")
            print(f"[INFO] final box       : {final_box is not None}")
            print(f"[INFO] box status      : {box_status}")
            print(f"[INFO] yolo boxes      : {num_yolo_boxes}")
            print(f"[INFO] elapsed         : {elapsed:.3f}s")
            print(f"[INFO] json saved      : {json_path}")
            print(f"[INFO] vis saved       : {vis_path}")
            print(f"[INFO] yolo saved      : {yolo_path}")

        except Exception as e:
            traceback.print_exc()

            summary_rows.append({
                "image_name": img_path.name,
                "width": "",
                "height": "",
                "num_food_boxes": "",
                "num_container_boxes": "",
                "has_food_box": "",
                "has_container_box": "",
                "num_boxes": "",
                "num_yolo_boxes": "",
                "box_status": "",
                "elapsed_seconds": "",
                "json_path": "",
                "vis_path": "",
                "yolo_path": "",
                "status": f"failed: {str(e)}",
            })

    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_name",
                "width",
                "height",
                "num_food_boxes",
                "num_container_boxes",
                "has_food_box",
                "has_container_box",
                "num_boxes",
                "num_yolo_boxes",
                "box_status",
                "elapsed_seconds",
                "json_path",
                "vis_path",
                "yolo_path",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


# ============================================================
# 11. 主流程
# ============================================================
def main():
    print("=" * 100)
    print("[INFO] Food + Container direct union LocateAnything script")
    print("[INFO] run mode     :", RUN_MODE)
    print("[INFO] model path   :", MODEL_PATH)
    print("[INFO] visible gpu  :", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("[INFO] worker id    :", WORKER_ID)
    print("[INFO] num workers  :", NUM_WORKERS)
    print("[INFO] target dish  :", TARGET_DISH_NAME)
    print("=" * 100)

    worker = LocateAnythingWorker(
        model_path=MODEL_PATH,
        device=DEVICE,
        dtype=DTYPE,
    )

    if RUN_MODE == "single":
        input_dir = Path(SINGLE_INPUT_DIR)
        output_root = Path(SINGLE_OUTPUT_ROOT)
        process_one_folder(worker, input_dir, output_root)

    elif RUN_MODE == "all":
        image_root = Path(IMAGE_ROOT)
        folders = collect_image_folders(str(image_root))

        print(f"[INFO] found {len(folders)} folders with images")

        for folder in folders:
            rel = folder.relative_to(image_root)
            if str(rel) == ".":
                rel = Path("_root")

            output_root = Path(ALL_OUTPUT_ROOT) / rel
            process_one_folder(worker, folder, output_root)

    else:
        raise ValueError("RUN_MODE must be 'single' or 'all'.")

    print("=" * 100)
    print("[INFO] finished")
    print("=" * 100)


if __name__ == "__main__":
    main()

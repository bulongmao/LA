from pathlib import Path
import shutil
import csv


# ============================================================
# 路径配置
# ============================================================
OUTPUTS_ROOT = Path("/data/ljy/locate_anything_project/outputs")

# 导出后的 label 总目录
EXPORT_ROOT = Path("/data/ljy/locate_anything_project/labels")

# 是否覆盖已有导出目录
OVERWRITE = True

# 是否只导出数字命名的文件夹，例如 1402、1403
ONLY_NUMERIC_FOLDERS = True

# 是否复制 classes.txt
COPY_CLASSES_TXT = False


# ============================================================
# 新增：开始和结束文件夹名称
# ============================================================

# 从哪个文件夹开始导出，包含该文件夹
# 例如 START_FOLDER_NAME = "1402"
# 如果不限制开始文件夹，设为 None
START_FOLDER_NAME = 1374

# 到哪个文件夹结束导出，包含该文件夹
# 例如 END_FOLDER_NAME = "1500"
# 如果不限制结束文件夹，设为 None
END_FOLDER_NAME = 1632


def folder_sort_key(folder: Path):
    """
    文件夹排序：
    数字文件夹按数字大小排序；
    非数字文件夹按字符串排序。
    """
    if folder.name.isdigit():
        return (0, int(folder.name))
    return (1, folder.name)


def folder_name_key(name: str):
    """
    文件夹名称比较用的 key。
    """
    name = str(name).strip()

    if name.isdigit():
        return (0, int(name))

    return (1, name)


def is_valid_folder(folder: Path) -> bool:
    if not folder.is_dir():
        return False

    if ONLY_NUMERIC_FOLDERS and not folder.name.isdigit():
        return False

    labels_dir = folder / "labels"
    if not labels_dir.exists() or not labels_dir.is_dir():
        return False

    return True


def is_in_folder_range(folder: Path) -> bool:
    """
    判断文件夹是否在 START_FOLDER_NAME 和 END_FOLDER_NAME 范围内。
    两端都是包含关系。
    """
    folder_key = folder_name_key(folder.name)

    if START_FOLDER_NAME is not None and str(START_FOLDER_NAME).strip() != "":
        start_key = folder_name_key(START_FOLDER_NAME)
        if folder_key < start_key:
            return False

    if END_FOLDER_NAME is not None and str(END_FOLDER_NAME).strip() != "":
        end_key = folder_name_key(END_FOLDER_NAME)
        if folder_key > end_key:
            return False

    return True


def copy_labels(src_labels_dir: Path, dst_labels_dir: Path) -> int:
    if OVERWRITE and dst_labels_dir.exists():
        shutil.rmtree(dst_labels_dir)

    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(src_labels_dir.glob("*.txt"))

    copied = 0
    for src_file in label_files:
        dst_file = dst_labels_dir / src_file.name
        shutil.copy2(src_file, dst_file)
        copied += 1

    return copied


def main():
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

    rows = []

    folders = sorted(
        [
            p for p in OUTPUTS_ROOT.iterdir()
            if is_valid_folder(p) and is_in_folder_range(p)
        ],
        key=folder_sort_key
    )

    print("=" * 100)
    print("[INFO] export labels by folder")
    print("[INFO] outputs root      :", OUTPUTS_ROOT)
    print("[INFO] export root       :", EXPORT_ROOT)
    print("[INFO] start folder name :", START_FOLDER_NAME)
    print("[INFO] end folder name   :", END_FOLDER_NAME)
    print("[INFO] folder count      :", len(folders))
    print("=" * 100)

    for folder in folders:
        folder_name = folder.name

        src_labels_dir = folder / "labels"
        dst_labels_dir = EXPORT_ROOT / folder_name

        num_labels = copy_labels(src_labels_dir, dst_labels_dir)

        src_classes = folder / "classes.txt"
        dst_classes = dst_labels_dir / "classes.txt"

        copied_classes = False
        if COPY_CLASSES_TXT and src_classes.exists():
            shutil.copy2(src_classes, dst_classes)
            copied_classes = True

        rows.append({
            "folder": folder_name,
            "src_labels_dir": str(src_labels_dir),
            "dst_labels_dir": str(dst_labels_dir),
            "num_label_files": num_labels,
            "copied_classes_txt": copied_classes,
        })

        print(
            f"[OK] {folder_name}: "
            f"{num_labels} label files -> {dst_labels_dir}, "
            f"classes={copied_classes}"
        )

    summary_path = EXPORT_ROOT / "export_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "folder",
                "src_labels_dir",
                "dst_labels_dir",
                "num_label_files",
                "copied_classes_txt",
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 100)
    print("[INFO] finished")
    print("[INFO] summary:", summary_path)
    print("=" * 100)


if __name__ == "__main__":
    main()
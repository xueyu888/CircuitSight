#!/usr/bin/env python3
from pathlib import Path

# ==== 配置 ====
DATASET_ROOT = Path("data/ck_data")
IMG_DIR = DATASET_ROOT / "tiles/images"
LBL_DIR = DATASET_ROOT / "tiles/labels"
SPLITS = ["train", "val"]
IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

TRAIN_TXT = DATASET_ROOT / "Train.txt"
VAL_TXT   = DATASET_ROOT / "Val.txt"
# ==============

def is_zero_only(lbl_path: Path) -> bool:
    if not lbl_path.exists():
        return False
    lines = [l.strip() for l in lbl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        return True  # 空文件也删
    for l in lines:
        cls_id = l.split()[0]
        if cls_id != "0":
            return False
    return True

def find_image_for_label(lbl_path: Path, split: str) -> Path | None:
    stem = lbl_path.stem
    for ext in IMG_EXTS:
        p = IMG_DIR/ split / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def rewrite_list_file(split: str, out_file: Path):
    imgs = sorted((IMG_DIR / split).glob("*"))
    imgs = [p for p in imgs if p.suffix.lower() in IMG_EXTS]
    root = Path.cwd().resolve()
    with open(out_file, "w", encoding="utf-8") as f:
        for p in imgs:
            try:
                f.write(p.resolve().relative_to(root).as_posix() + "\n")
            except Exception:
                f.write(p.as_posix() + "\n")

def main():
    removed = {s: 0 for s in SPLITS}
    for split in SPLITS:
        lbl_dir = LBL_DIR / split
        img_dir = IMG_DIR / split
        if not lbl_dir.exists():
            continue
        for lbl in sorted(lbl_dir.glob("*.txt")):
            if is_zero_only(lbl):
                img = find_image_for_label(lbl, split)
                # 删除标签
                try:
                    lbl.unlink()
                    print(f"[DEL] {lbl}")
                except FileNotFoundError:
                    pass
                # 删除图片
                if img and img.exists():
                    try:
                        img.unlink()
                        print(f"[DEL] {img}")
                    except FileNotFoundError:
                        pass
                removed[split] += 1

    # 重写 Train.txt / Val.txt
    rewrite_list_file("train", TRAIN_TXT)
    rewrite_list_file("val",   VAL_TXT)

    print("\nDone.")
    print(f"Removed train: {removed['train']} pairs")
    print(f"Removed val:   {removed['val']} pairs")
    print(f"Wrote {TRAIN_TXT}")
    print(f"Wrote {VAL_TXT}")

if __name__ == "__main__":
    main()

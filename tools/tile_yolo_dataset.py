#!/usr/bin/env python3
# tools/tile_yolo_dataset.py
import math
from pathlib import Path
from PIL import Image, ImageFile
import shutil

# 允许解大图，去掉 DecompressionBomb 警告
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==== 可调参数 ====
SRC_IMG_DIR = Path("data/ck_data/images/Train")
SRC_LBL_DIR = Path("data/ck_data/labels/Train")
OUT_IMG_DIR = Path("data/ck_data/tiles/images")
OUT_LBL_DIR = Path("data/ck_data/tiles/labels")

TILE = 2048           # 单段最大宽/高（无重叠）
EDGE_MARGIN = 2       # 判定“完整包含”的安全边（像素）
SEARCH_BACK_STEP = 4  # 放不下时，向后回退寻找可用切线的步长（像素）
MIN_SEG = 64          # 单段最小尺寸，避免生成过薄切片

# 指定整图进 val（其他进 train）
VAL_SINGLE_IMAGE = Path("data/ck_data/images/Train/F13121S-D0901(1)-03-模型.png")

# 索引与 yaml
DATASET_ROOT = Path("data/ck_data")
TRAIN_LIST = DATASET_ROOT / "Train.txt"
VAL_LIST   = DATASET_ROOT / "Val.txt"
DATASET_YAML = DATASET_ROOT / "dataset.yaml"
# ==================

def yolo_to_xyxy(line, W, H):
    ps = line.strip().split()
    if len(ps) < 5:
        return None
    cls = int(ps[0])
    cx, cy, w, h = map(float, ps[1:5])
    cx *= W; cy *= H; w *= W; h *= H
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2
    return cls, x1, y1, x2, y2

def xyxy_to_yolo(cls, x1, y1, x2, y2, w_tile, h_tile):
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw/2
    cy = y1 + bh/2
    return f"{cls} {cx / w_tile:.6f} {cy / h_tile:.6f} {bw / w_tile:.6f} {bh / h_tile:.6f}"

def load_labels(lbl_path, W, H):
    if not lbl_path.exists():
        return []
    with open(lbl_path, "r", encoding="utf-8") as f:
        lines = [l for l in f.read().strip().splitlines() if l.strip()]
    boxes = []
    for l in lines:
        r = yolo_to_xyxy(l, W, H)
        if r is not None:
            boxes.append(r)
    return boxes

def ensure_clean_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

# ==== 约束切线规划 ====
def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort()
    merged = []
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    return merged

def forbidden_intervals_1d(boxes_on_axis, edge, L):
    # 对每个框，禁止切线落在 (b1+edge, b2-edge)
    ints = []
    for (b1, b2) in boxes_on_axis:
        s = max(0, b1 + edge)
        e = min(L, b2 - edge)
        if e > s:
            ints.append((s, e))
    return merge_intervals(ints)

def adjust_to_allowed(x, forb):
    """若 x 落在某个禁止区间内，则推到该区间右端点；否则返回 x"""
    for s, e in forb:
        if s < x < e:
            return e
    return x

def adjust_down_to_allowed(x, forb):
    """向左寻找最近的允许位置（不落在任何禁区里）"""
    for s, e in reversed(forb):
        if s < x <= e:
            return s
    return x

def plan_axis_cuts(L, boxes_on_axis, tile, edge, back_step, min_seg):
    """
    在 [0, L] 上放若干切线，使每段长度 <= tile，且切线不落在任何禁止区间里。
    返回切线位置数组（包含 0 和 L）。
    """
    # 快速不可行性：若有框尺寸 > tile-2*edge，必然无解
    for (b1, b2) in boxes_on_axis:
        if (b2 - b1) > tile - 2*edge:
            return None

    forb = forbidden_intervals_1d(boxes_on_axis, edge, L)
    cuts = [0.0]
    while True:
        s = cuts[-1]
        if L - s <= tile:
            # 最后一段
            cuts.append(float(L))
            break

        target = s + tile
        if target > L:
            target = L

        # 先尝试向右调整到允许位置
        e = adjust_to_allowed(target, forb)
        if e - s <= tile and e - s >= min_seg and e <= L:
            cuts.append(float(e))
            continue

        # 再尝试向左回退
        e2 = target
        while e2 - s > tile or any( (s_ < e2 < e_) for (s_, e_) in forb ):
            e2 -= back_step
            if e2 <= s + min_seg:
                break
        if e2 - s <= tile and e2 - s >= min_seg and not any((s_ < e2 < e_) for (s_, e_) in forb):
            cuts.append(float(e2))
            continue

        # 仍无解
        return None

    # 清理浮点误差
    cuts = [0.0] + [c for c in cuts[1:-1] if c - cuts[cuts.index(c)-1] >= min_seg] + [float(L)]
    return cuts

# ==== 生成列表与 yaml ====
def write_list_file(paths, outfile: Path):
    outfile.parent.mkdir(parents=True, exist_ok=True)
    root = Path.cwd()
    with open(outfile, "w", encoding="utf-8") as f:
        for p in sorted(paths):
            try:
                rel = p.resolve().relative_to(root.resolve())
                f.write(rel.as_posix() + "\n")
            except Exception:
                f.write(p.as_posix() + "\n")

def find_max_class_from_labels(lbl_dir: Path):
    max_cls = -1
    for p in lbl_dir.glob("*.txt"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    ls = line.strip().split()
                    if len(ls) >= 1:
                        c = int(ls[0]); max_cls = max(max_cls, c)
        except Exception:
            pass
    return max_cls

def write_dataset_yaml(num_classes: int):
    DATASET_YAML.parent.mkdir(parents=True, exist_ok=True)
    names_lines = [f"  {i}: cls_{i}" for i in range(num_classes)]
    content = f"""# Auto-generated by tools/tile_yolo_dataset.py
path: {DATASET_ROOT.as_posix()}

train: {TRAIN_LIST.name}
val: {VAL_LIST.name}

names:
{chr(10).join(names_lines)}
"""
    with open(DATASET_YAML, "w", encoding="utf-8") as f:
        f.write(content)

# ==== 主流程 ====
def main():
    # 清空输出
    for sub in ["train", "val"]:
        ensure_clean_dir(OUT_IMG_DIR / sub)
        ensure_clean_dir(OUT_LBL_DIR / sub)

    img_paths = sorted(SRC_IMG_DIR.glob("*.png"))
    if not img_paths:
        raise SystemExit(f"No PNG images under {SRC_IMG_DIR}")

    skipped = []
    total_tiles = {"train": 0, "val": 0}

    for img_path in img_paths:
        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        lbl_path = (SRC_LBL_DIR / (img_path.stem + ".txt"))
        boxes = load_labels(lbl_path, W, H)

        # 按轴取区间
        x_intervals = [(bx1, bx2) for (_, bx1, _, bx2, _) in boxes]
        y_intervals = [(by1, by2) for (_, _, by1, _, by2) in boxes]

        x_cuts = plan_axis_cuts(W, x_intervals, TILE, EDGE_MARGIN, SEARCH_BACK_STEP, MIN_SEG)
        y_cuts = plan_axis_cuts(H, y_intervals, TILE, EDGE_MARGIN, SEARCH_BACK_STEP, MIN_SEG)

        if x_cuts is None or y_cuts is None:
            print(f"[WARN] {img_path.name}: no-overlap cut planning failed. "
                  f"Increase TILE or reduce EDGE_MARGIN, or split oversized boxes.")
            skipped.append(img_path.name)
            continue

        is_val = img_path.resolve() == VAL_SINGLE_IMAGE.resolve()
        split = "val" if is_val else "train"

        # 生成无重叠切片
        for xi in range(len(x_cuts)-1):
            x1, x2 = int(round(x_cuts[xi])), int(round(x_cuts[xi+1]))
            if x2 - x1 < MIN_SEG:  # 跳过异常薄切片
                continue
            for yi in range(len(y_cuts)-1):
                y1, y2 = int(round(y_cuts[yi])), int(round(y_cuts[yi+1]))
                if y2 - y1 < MIN_SEG:
                    continue

                w_tile = x2 - x1
                h_tile = y2 - y1
                tile_img = im.crop((x1, y1, x2, y2))

                # 仅写入“完全包含”的框
                tile_labels = []
                for (cls, bx1, by1, bx2, by2) in boxes:
                    if (bx1 >= x1 + EDGE_MARGIN and by1 >= y1 + EDGE_MARGIN and
                        bx2 <= x2 - EDGE_MARGIN and by2 <= y2 - EDGE_MARGIN):
                        lx1 = bx1 - x1; ly1 = by1 - y1
                        lx2 = bx2 - x1; ly2 = by2 - y1
                        tile_labels.append(xyxy_to_yolo(cls, lx1, ly1, lx2, ly2, w_tile, h_tile))

                out_img_name = f"{img_path.stem}_x{x1}_y{y1}.png"
                out_lbl_name = f"{img_path.stem}_x{x1}_y{y1}.txt"

                (OUT_IMG_DIR / split).mkdir(parents=True, exist_ok=True)
                (OUT_LBL_DIR / split).mkdir(parents=True, exist_ok=True)

                tile_img.save(OUT_IMG_DIR / split / out_img_name)
                with open(OUT_LBL_DIR / split / out_lbl_name, "w", encoding="utf-8") as f:
                    f.write("\n".join(tile_labels) + ("\n" if tile_labels else ""))

                total_tiles[split] += 1

        print(f"[OK] {img_path.name}: tiles={total_tiles[split]}, split={split}")

    # 生成 Train.txt / Val.txt
    train_imgs = list((OUT_IMG_DIR / "train").glob("*.png"))
    val_imgs   = list((OUT_IMG_DIR / "val").glob("*.png"))
    write_list_file(train_imgs, TRAIN_LIST)
    write_list_file(val_imgs, VAL_LIST)

    # 生成 dataset.yaml（names 占位）
    max_cls = find_max_class_from_labels(SRC_LBL_DIR)
    if max_cls < 0:
        max_cls = find_max_class_from_labels(OUT_LBL_DIR / "train")
    if max_cls < 0:
        max_cls = 0
    write_dataset_yaml(num_classes=max_cls + 1)

    print("\nTiling done (NO OVERLAP, cutline planning).")
    print(f"Train tiles: {len(train_imgs)}  |  Val tiles: {len(val_imgs)}")
    print(f"Wrote: {TRAIN_LIST}")
    print(f"Wrote: {VAL_LIST}")
    print(f"Wrote: {DATASET_YAML}")
    if skipped:
        print(f"[WARN] Images skipped due to infeasible constraints: {len(skipped)}")
        for n in skipped[:20]:
            print(" -", n)

if __name__ == "__main__":
    main()

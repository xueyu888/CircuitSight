#!/usr/bin/env python3
# tools/tile_yolo_dataset.py  (multi-input: files or folders)
# 功能：
#  - 支持传入任意数量的图片文件，或若干包含图片的文件夹
#  - 自动按 YOLO 标签推断 labels（同目录 .txt 或 images→labels 平行目录）
#  - 切 tile（无重叠，切线完全避开任何目标框），每张原图可抽若干 tile 进验证集
#  - 生成 Train.txt / Val.txt 和 dataset.yaml

from __future__ import annotations
import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple
from PIL import Image, ImageFile

# 允许解大图，去掉 DecompressionBomb 警告
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ========== 默认参数（可被 CLI 覆盖） ==========
DEFAULT_TILE = 2048
DEFAULT_EDGE_MARGIN = 2          # 现在只在极少数地方保留，不再用于“可以压进框内”的逻辑
DEFAULT_SEARCH_BACK_STEP = 4
DEFAULT_MIN_SEG = 64

DEFAULT_VAL_TILES_PER_IMAGE = 1
DEFAULT_VAL_SELECT_PREFER_LABELED = True
DEFAULT_RANDOM_SEED = 0

DEFAULT_OUT_ROOT = Path("data/ck_data/tiles")  # 输出根目录，下面会建 images/{train,val} 和 labels/{train,val}
DEFAULT_DATASET_ROOT = Path("data/ck_data")     # Train.txt / Val.txt / dataset.yaml 写在这里

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# =================================================
#  YOLO 坐标/标签工具
# =================================================

def yolo_to_xyxy(line: str, W: int, H: int):
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


def xyxy_to_yolo(cls: int, x1: float, y1: float, x2: float, y2: float, w_tile: int, h_tile: int) -> str:
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw/2
    cy = y1 + bh/2
    return f"{cls} {cx / w_tile:.6f} {cy / h_tile:.6f} {bw / w_tile:.6f} {bh / h_tile:.6f}"


def load_labels(lbl_path: Path, W: int, H: int):
    if not lbl_path or not lbl_path.exists():
        return []
    with open(lbl_path, "r", encoding="utf-8") as f:
        lines = [l for l in f.read().strip().splitlines() if l.strip()]
    boxes = []
    for l in lines:
        r = yolo_to_xyxy(l, W, H)
        if r is not None:
            boxes.append(r)
    return boxes

# =================================================
#  目录/文件管理
# =================================================

def ensure_clean_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def iter_images_from_inputs(inputs: List[Path]) -> List[Path]:
    imgs: List[Path] = []
    for inp in inputs:
        if inp.is_dir():
            for p in inp.rglob("*"):
                if p.suffix.lower() in IMG_EXTS and p.is_file():
                    imgs.append(p)
        elif inp.is_file() and inp.suffix.lower() in IMG_EXTS:
            imgs.append(inp)
    # 去重 + 排序
    uniq = sorted(set(p.resolve() for p in imgs))
    if not uniq:
        raise SystemExit("No images found from inputs.")
    return uniq


def resolve_label_path(img_path: Path, images_root: Path | None, labels_root: Path | None) -> Path | None:
    # 规则1：同目录同名 .txt
    same_dir = img_path.with_suffix(".txt")
    if same_dir.exists():
        return same_dir

    # 规则2：images → labels 平行目录（保持相对路径）
    # 2.1 显式给了 roots 且 img 在 images_root 下
    if images_root and labels_root:
        try:
            rel = img_path.resolve().relative_to(images_root.resolve())
            cand = (labels_root / rel).with_suffix(".txt")
            if cand.exists():
                return cand
        except Exception:
            pass

    # 2.2 自动把路径片段中的 "images" 替换为 "labels"
    parts = list(img_path.parts)
    for i, seg in enumerate(parts):
        if seg.lower() == "images":
            parts[i] = "labels"
            cand = Path(*parts).with_suffix(".txt")
            if cand.exists():
                return cand
            break

    return None  # 找不到也行（表示空标签）

# =================================================
#  切线规划（避免切线穿过目标）
# =================================================

def merge_intervals(intervals: List[Tuple[float, float]]):
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


def forbidden_intervals_1d(boxes_on_axis, edge: int, L: int):
    """
    修改点 1：
    原来是禁止切线落在 (b1+edge, b2-edge)，允许线压进目标框 edge 像素。
    现在改成：只要是有标注的框，[b1, b2] 整段都视为禁止切线区域。
    """
    ints = []
    for (b1, b2) in boxes_on_axis:
        s = max(0, b1)   # 从框左边界开始
        e = min(L, b2)   # 到框右边界结束
        if e > s:
            ints.append((s, e))
    return merge_intervals(ints)


def adjust_to_allowed(x: float, forb):
    for s, e in forb:
        if s < x < e:
            return e
    return x


def plan_axis_cuts(L: int, boxes_on_axis, tile: int, edge: int, back_step: int, min_seg: int):
    """
    修改点 2：
    不再允许“tile 尺寸 - 2*edge”那种逻辑，因为现在切线完全不能进框。
    只要有框尺寸 > tile，就直接无解。
    """
    # 快速不可行性：若有框尺寸 > tile，必然无解（否则一定要切穿它）
    for (b1, b2) in boxes_on_axis:
        if (b2 - b1) > tile:
            return None

    forb = forbidden_intervals_1d(boxes_on_axis, edge, L)
    cuts = [0.0]
    while True:
        s = cuts[-1]
        if L - s <= tile:  # 最后一段
            cuts.append(float(L))
            break

        target = s + tile
        if target > L:
            target = L

        # 向右挪到允许区间
        e = adjust_to_allowed(target, forb)
        if e - s <= tile and e - s >= min_seg and e <= L:
            cuts.append(float(e))
            continue

        # 向左回退（粗略回退 back_step）
        e2 = target
        while e2 - s > tile or any((s_ < e2 < e_) for (s_, e_) in forb):
            e2 -= back_step
            if e2 <= s + min_seg:
                break
        if e2 - s <= tile and e2 - s >= min_seg and not any((s_ < e2 < e_) for (s_, e_) in forb):
            cuts.append(float(e2))
            continue

        return None  # 仍无解

    # 清理浮点误差 & 极薄段
    out = [0.0]
    for i in range(1, len(cuts) - 1):
        if cuts[i] - out[-1] >= min_seg:
            out.append(cuts[i])
    out.append(float(L))
    return out

# =================================================
#  清单与 YAML
# =================================================

def write_list_file(paths: List[Path], outfile: Path):
    outfile.parent.mkdir(parents=True, exist_ok=True)
    root = Path.cwd().resolve()
    with open(outfile, "w", encoding="utf-8") as f:
        for p in sorted(paths):
            try:
                rel = p.resolve().relative_to(root)
                f.write(rel.as_posix() + "\n")
            except Exception:
                f.write(p.as_posix() + "\n")


def find_max_class_from_labels(lbl_dir: Path) -> int:
    max_cls = -1
    if not lbl_dir.exists():
        return max_cls
    for p in lbl_dir.rglob("*.txt"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    ls = line.strip().split()
                    if len(ls) >= 1:
                        c = int(ls[0]); max_cls = max(max_cls, c)
        except Exception:
            pass
    return max_cls


def write_dataset_yaml(dataset_root: Path, train_list: Path, val_list: Path, num_classes: int):
    dataset_root.mkdir(parents=True, exist_ok=True)
    names_lines = [f"  {i}: cls_{i}" for i in range(num_classes)]
    content = f"""# Auto-generated by tools/tile_yolo_dataset.py
path: {dataset_root.as_posix()}

train: {train_list.name}
val: {val_list.name}

names:
{chr(10).join(names_lines)}
"""
    with open(dataset_root / "dataset.yaml", "w", encoding="utf-8") as f:
        f.write(content)

# =================================================
#  主流程
# =================================================

def process_images(
    images: List[Path],
    images_root: Path | None,
    labels_root: Path | None,
    out_root: Path,
    dataset_root: Path,
    tile: int,
    edge: int,
    back_step: int,
    min_seg: int,
    val_tiles_per_image: int,
    prefer_labeled: bool,
    rng_seed: int,
):
    random.seed(rng_seed)

    out_img_train = out_root / "images" / "train"
    out_img_val   = out_root / "images" / "val"
    out_lbl_train = out_root / "labels" / "train"
    out_lbl_val   = out_root / "labels" / "val"

    for d in [out_img_train, out_img_val, out_lbl_train, out_lbl_val]:
        ensure_clean_dir(d)

    train_imgs_written: List[Path] = []
    val_imgs_written: List[Path] = []

    skipped: List[str] = []

    for img_path in images:
        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        lbl_path = resolve_label_path(img_path, images_root, labels_root)
        boxes = load_labels(lbl_path, W, H)

        x_intervals = [(bx1, bx2) for (_, bx1, _, bx2, _) in boxes]
        y_intervals = [(by1, by2) for (_, _, by1, _, by2) in boxes]

        x_cuts = plan_axis_cuts(W, x_intervals, tile, edge, back_step, min_seg)
        y_cuts = plan_axis_cuts(H, y_intervals, tile, edge, back_step, min_seg)

        if x_cuts is None or y_cuts is None:
            print(f"[WARN] {img_path.name}: no-overlap cut planning failed. "
                  f"Increase TILE or reduce EDGE_MARGIN, or split oversized boxes.")
            skipped.append(img_path.name)
            continue

        # 先生成元数据
        tile_meta = []  # (x1,y1,x2,y2, labels)
        for xi in range(len(x_cuts)-1):
            x1, x2 = int(round(x_cuts[xi])), int(round(x_cuts[xi+1]))
            if x2 - x1 < min_seg:
                continue
            for yi in range(len(y_cuts)-1):
                y1, y2 = int(round(y_cuts[yi])), int(round(y_cuts[yi+1]))
                if y2 - y1 < min_seg:
                    continue

                w_tile = x2 - x1
                h_tile = y2 - y1

                tile_labels = []
                for (cls, bx1, by1, bx2, by2) in boxes:
                    # 修改点 3：
                    # 原逻辑要求：框在 tile 内，且离 tile 边缘至少 edge 像素
                    # 现在我们只要求“完全落在该 tile 内”，不再做第二次 edge 过滤，
                    # 因为切线本身已经保证不会穿过这个框。
                    if (bx1 >= x1 and by1 >= y1 and
                        bx2 <= x2 and by2 <= y2):
                        lx1 = bx1 - x1; ly1 = by1 - y1
                        lx2 = bx2 - x1; ly2 = by2 - y1
                        tile_labels.append(xyxy_to_yolo(cls, lx1, ly1, lx2, ly2, w_tile, h_tile))

                tile_meta.append((x1, y1, x2, y2, tile_labels))

        if not tile_meta:
            print(f"[WARN] {img_path.name}: no tiles produced.")
            continue

        # 选 val 索引
        candidate_idx = [i for i, m in enumerate(tile_meta) if len(m[4]) > 0] if prefer_labeled else list(range(len(tile_meta)))
        if not candidate_idx:
            candidate_idx = list(range(len(tile_meta)))
        k = min(val_tiles_per_image, len(candidate_idx))
        val_indices = set(random.sample(candidate_idx, k))

        # 落盘
        for i, (x1, y1, x2, y2, tile_labels) in enumerate(tile_meta):
            split = "val" if i in val_indices else "train"
            tile_img = im.crop((x1, y1, x2, y2))
            out_img_name = f"{img_path.stem}_x{x1}_y{y1}.png"
            out_lbl_name = f"{img_path.stem}_x{x1}_y{y1}.txt"

            if split == "val":
                img_out_path = out_img_val / out_img_name
                lbl_out_path = out_lbl_val / out_lbl_name
                val_imgs_written.append(img_out_path)
            else:
                img_out_path = out_img_train / out_img_name
                lbl_out_path = out_lbl_train / out_lbl_name
                train_imgs_written.append(img_out_path)

            tile_img.save(img_out_path)
            with open(lbl_out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(tile_labels) + ("\n" if tile_labels else ""))

        print(f"[OK] {img_path.name}: tiles(total)={len(tile_meta)}, val={len(val_indices)}, train={len(tile_meta)-len(val_indices)}")

    # 写 Train/Val 列表
    train_list = dataset_root / "Train.txt"
    val_list = dataset_root / "Val.txt"
    write_list_file(train_imgs_written, train_list)
    write_list_file(val_imgs_written, val_list)

    # 估类数
    num_classes = -1
    # 优先从源 labels 根统计
    if labels_root and labels_root.exists():
        num_classes = max(num_classes, find_max_class_from_labels(labels_root))
    # 再从切片 labels 统计
    num_classes = max(num_classes, find_max_class_from_labels(out_root / "labels" / "train"))
    num_classes = max(num_classes, find_max_class_from_labels(out_root / "labels" / "val"))
    if num_classes < 0:
        num_classes = 0

    write_dataset_yaml(dataset_root, train_list, val_list, num_classes + 1)

    print("\nTiling done (NO OVERLAP, cutline planning; no box is ever cut).")
    print(f"Train tiles: {len(train_imgs_written)}  |  Val tiles: {len(val_imgs_written)}")
    print(f"Wrote: {train_list}")
    print(f"Wrote: {val_list}")
    print(f"Wrote: {dataset_root / 'dataset.yaml'}")

# =================================================
#  CLI
# =================================================

def build_argparser():
    ap = argparse.ArgumentParser(description="Tile large images for YOLO training, with auto val selection.")
    ap.add_argument("inputs", nargs="+", type=Path,
                    help="任意数量的图片文件或文件夹，支持混合。文件夹会递归扫描常见图片后缀。")
    ap.add_argument("--images-root", type=Path, default=None,
                    help="可选：images 根目录。若提供且 inputs 在其下，将与 --labels-root 组合推断标签相对路径。")
    ap.add_argument("--labels-root", type=Path, default=None,
                    help="可选：labels 根目录。优先使用同目录 .txt；找不到时，尝试 images→labels 平行路径；仍找不到则视为无标签。")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                    help=f"输出根目录（默认: {DEFAULT_OUT_ROOT}）")
    ap.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
                    help=f"Train.txt/Val.txt/dataset.yaml 输出位置（默认: {DEFAULT_DATASET_ROOT}）")

    ap.add_argument("--tile", type=int, default=DEFAULT_TILE)
    ap.add_argument("--edge", type=int, default=DEFAULT_EDGE_MARGIN,
                    help="边缘安全距离（当前只影响切线搜索的一些限制，不再允许压进目标框内部）")
    ap.add_argument("--back-step", type=int, default=DEFAULT_SEARCH_BACK_STEP)
    ap.add_argument("--min-seg", type=int, default=DEFAULT_MIN_SEG)

    ap.add_argument("--val-tiles-per-image", type=int, default=DEFAULT_VAL_TILES_PER_IMAGE)
    ap.add_argument("--prefer-labeled", action="store_true", default=DEFAULT_VAL_SELECT_PREFER_LABELED)
    ap.add_argument("--no-prefer-labeled", dest="prefer_labeled", action="store_false")
    ap.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)

    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    images = iter_images_from_inputs(args.inputs)

    process_images(
        images=images,
        images_root=args.images_root,
        labels_root=args.labels_root,
        out_root=args.out_root,
        dataset_root=args.dataset_root,
        tile=args.tile,
        edge=args.edge,
        back_step=args.back_step,
        min_seg=args.min_seg,
        val_tiles_per_image=args.val_tiles_per_image,
        prefer_labeled=args.prefer_labeled,
        rng_seed=args.seed,
    )

if __name__ == "__main__":
    main()

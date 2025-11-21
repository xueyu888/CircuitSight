#!/usr/bin/env python3
"""
wire_extractor.py

目的：
  仅把“导线”从电路图中抽出并保存为图像；不做 jumper/拓扑/JSON。

满足你的硬性要求：
  1) 器件框可选：可用/可不用（不去掉器件也能跑）；
  2) 其它参数尽量可配置（CLI 参数化）；
  3) 每一步都导出中间图；
  4) 全量类型标注，Pylance 友好。

依赖：
  pip install opencv-python scikit-image numpy

示例：
  python wire_extractor.py \
    --input examples/schematic.png \
    --outdir runs/wires_demo \
    --remove-devices true \
    --det-json dets.json \
    --binarize otsu \
    --small-comp-ratio 0.10 \
    --thicken 3 \
    --overlay-alpha 0.40

输出（固定命名）：
  00_input.png
  01_gray.png
  02_bin.png
  03_skeleton.png
  04_skeleton_no_device.png
  05_skeleton_clean.png
  06_wires_thick.png
  07_overlay.png
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
from skimage.filters import threshold_sauvola
from skimage.morphology import skeletonize
from skimage.measure import label as cc_label, regionprops

# =========================
# 数据结构
# =========================

@dataclass(frozen=True)
class BBox:
    """像素坐标的矩形框。"""
    x1: int
    y1: int
    x2: int
    y2: int
    cls: str = "device"

    def clip(self, w: int, h: int) -> "BBox":
        x1 = max(0, min(self.x1, w - 1))
        y1 = max(0, min(self.y1, h - 1))
        x2 = max(0, min(self.x2, w))
        y2 = max(0, min(self.y2, h))
        if x2 <= x1:
            x2 = min(w, x1 + 1)
        if y2 <= y1:
            y2 = min(h, y1 + 1)
        return BBox(x1, y1, x2, y2, self.cls)

    def as_slice(self) -> Tuple[slice, slice]:
        return (slice(self.y1, self.y2), slice(self.x1, self.x2))

# =========================
# 基础 I/O
# =========================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def imwrite(path: str, img: NDArray[np.uint8]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


def read_image(path: str) -> NDArray[np.uint8]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img

# =========================
# 预处理
# =========================

def to_gray(img: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if img.ndim == 2:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def binarize(
    gray: NDArray[np.uint8],
    method: str = "otsu",
    sauvola_win: Optional[int] = None,
    sauvola_k: float = 0.20,
) -> NDArray[np.uint8]:
    """返回白底(255)/黑线(0)的二值图。"""
    if method.lower() == "sauvola":
        # 窗口未指定时，按短边 2% 估计；需为奇数
        win = sauvola_win or max(15, int(min(gray.shape[:2]) * 0.02) | 1)
        th = threshold_sauvola(gray, window_size=win, k=sauvola_k)
        bw = (gray > th).astype(np.uint8) * 255
    else:
        _t, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 统一白底黑线
    if float(np.mean(bw)) < 128.0:
        bw = cv2.bitwise_not(bw)
    return bw


def do_skeleton(bw_white_bg_black_fg: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """对黑色前景做骨架化，返回白底黑线（1px）图。"""
    fg = (bw_white_bg_black_fg == 0)
    skel_bool = skeletonize(fg)
    out = np.full_like(bw_white_bg_black_fg, 255, dtype=np.uint8)
    out[skel_bool] = 0
    return out

# =========================
# 器件处理（可选）
# =========================

def load_bboxes_csv(csv_path: str, image_path: str) -> List[BBox]:
    out: List[BBox] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.lower().startswith("image_path"):
                continue
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) < 6:
                continue
            ip, x1, y1, x2, y2, cls = parts[:6]
            if os.path.normpath(ip) != os.path.normpath(image_path):
                continue
            out.append(BBox(int(x1), int(y1), int(x2), int(y2), cls))
    return out


def load_bboxes_json(json_path: str, image_path: str) -> List[BBox]:
    data = json.load(open(json_path, "r", encoding="utf-8"))
    out: List[BBox] = []
    for d in data:
        ip = str(d.get("image_path", ""))
        if os.path.normpath(ip) != os.path.normpath(image_path):
            continue
        out.append(
            BBox(
                int(d["x1"]),
                int(d["y1"]),
                int(d["x2"]),
                int(d["y2"]),
                str(d.get("class", "device")),
            )
        )
    return out


def subtract_device_boxes(
    skel: NDArray[np.uint8],
    boxes: Sequence[BBox],
    classes_include: Optional[Iterable[str]] = None,
    classes_exclude: Optional[Iterable[str]] = None,
) -> NDArray[np.uint8]:
    h, w = skel.shape
    out = skel.copy()
    include = set([c.lower() for c in classes_include]) if classes_include else None
    exclude = set([c.lower() for c in classes_exclude]) if classes_exclude else set()
    for b in boxes:
        cls = b.cls.lower()
        if include is not None and cls not in include:
            continue
        if cls in exclude:
            continue
        ys, xs = b.clip(w, h).as_slice()
        out[ys, xs] = 255  # 置白，等价于“从骨架里扣掉”
    return out

# =========================
# 清噪与可视化
# =========================

def remove_small_components(
    skel: NDArray[np.uint8],
    ratio: float = 0.10,
) -> NDArray[np.uint8]:
    """删除面积 < max_area*ratio 的骨架连通域（保留为白底黑线）。"""
    # 前景=黑
    labels = cc_label((skel == 0).astype(np.uint8), connectivity=2)
    props = regionprops(labels)
    if not props:
        return skel.copy()
    max_area = max(p.area for p in props)
    thr = max(1, int(max_area * ratio))
    keep_mask = np.zeros_like(labels, dtype=np.uint8)
    for p in props:
        if p.area >= thr:
            keep_mask[labels == p.label] = 1
    out = np.full_like(skel, 255, dtype=np.uint8)
    out[keep_mask == 1] = 0
    return out


def thicken_lines(
    skel: NDArray[np.uint8],
    kernel: int = 3,
) -> NDArray[np.uint8]:
    """把 1px 骨架稍微加粗（白底黑线）。"""
    k = max(1, int(kernel))
    k = k if k % 2 == 1 else k + 1  # 奇数核观感更好
    elem = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    fg = (skel == 0).astype(np.uint8)
    dil = cv2.dilate(fg, elem, iterations=1)
    out = np.full_like(skel, 255, dtype=np.uint8)
    out[dil > 0] = 0
    return out


def overlay_on_image(
    orig_bgr: NDArray[np.uint8],
    wires_black: NDArray[np.uint8],
    alpha: float = 0.40,
) -> NDArray[np.uint8]:
    """将黑线覆盖到原图上，可调透明度（黑线=加深）。"""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    overlay = orig_bgr.astype(np.float32)
    mask = (wires_black == 0)
    # 对线条位置做暗化：new = (1-alpha)*orig
    overlay[mask] = overlay[mask] * (1.0 - alpha)
    return np.clip(overlay, 0, 255).astype(np.uint8)

# =========================
# 主流程
# =========================

def process(
    image_path: str,
    outdir: str,
    *,
    remove_devices: bool = True,
    det_csv: Optional[str] = None,
    det_json: Optional[str] = None,
    include_classes: Optional[List[str]] = None,
    exclude_classes: Optional[List[str]] = None,
    binarize_method: str = "otsu",
    sauvola_win: Optional[int] = None,
    sauvola_k: float = 0.20,
    small_comp_ratio: float = 0.10,
    thicken_kernel: int = 3,
    overlay_alpha: float = 0.40,
) -> None:
    ensure_dir(outdir)

    # 读图 & 保存
    img = read_image(image_path)
    imwrite(os.path.join(outdir, "00_input.png"), img)

    # 灰度
    gray = to_gray(img)
    imwrite(os.path.join(outdir, "01_gray.png"), gray)

    # 二值
    bw = binarize(gray, method=binarize_method, sauvola_win=sauvola_win, sauvola_k=sauvola_k)
    imwrite(os.path.join(outdir, "02_bin.png"), bw)

    # 骨架
    skel = do_skeleton(bw)
    imwrite(os.path.join(outdir, "03_skeleton.png"), skel)

    # 可选：扣掉器件
    skel_no_dev = skel.copy()
    if remove_devices and (det_csv or det_json):
        boxes: List[BBox] = []
        if det_csv:
            boxes.extend(load_bboxes_csv(det_csv, image_path))
        if det_json:
            boxes.extend(load_bboxes_json(det_json, image_path))
        skel_no_dev = subtract_device_boxes(
            skel,
            boxes,
            classes_include=include_classes,
            classes_exclude=exclude_classes,
        )
    imwrite(os.path.join(outdir, "04_skeleton_no_device.png"), skel_no_dev)

    # 删除小连通域（清文本/杂质）
    skel_clean = remove_small_components(skel_no_dev, ratio=small_comp_ratio)
    imwrite(os.path.join(outdir, "05_skeleton_clean.png"), skel_clean)

    # 加粗可视化
    thick = thicken_lines(skel_clean, kernel=thicken_kernel)
    imwrite(os.path.join(outdir, "06_wires_thick.png"), thick)

    # 叠加原图
    overlay = overlay_on_image(img, thick, alpha=overlay_alpha)
    imwrite(os.path.join(outdir, "07_overlay.png"), overlay)

# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="只提取导线并输出各步骤图像")
    ap.add_argument("--input", required=True, help="输入电路图 (PNG/JPG)")
    ap.add_argument("--outdir", required=True, help="输出目录")

    ap.add_argument("--remove-devices", type=lambda s: s.lower() == "true", default=True, help="是否使用设备框从骨架中扣掉器件区域")
    ap.add_argument("--det-csv", default=None, help="设备检测 CSV：image_path,x1,y1,x2,y2,class")
    ap.add_argument("--det-json", default=None, help="设备检测 JSON：[{image_path,x1,y1,x2,y2,class}, ...]")
    ap.add_argument("--include-classes", nargs="*", default=None, help="仅处理这些类名（可空）")
    ap.add_argument("--exclude-classes", nargs="*", default=None, help="排除这些类名（可空）")

    ap.add_argument("--binarize", choices=["otsu", "sauvola"], default="otsu", help="二值化方法")
    ap.add_argument("--sauvola-win", type=int, default=None, help="sauvola 窗口（奇数，默认按图大小估计）")
    ap.add_argument("--sauvola-k", type=float, default=0.20, help="sauvola k（0.1~0.3 常用）")

    ap.add_argument("--small-comp-ratio", type=float, default=0.10, help="删除小连通域阈值 = 最大连通域面积 * ratio")
    ap.add_argument("--thicken", type=int, default=3, help="导线加粗核尺寸（奇数更好看）")
    ap.add_argument("--overlay-alpha", type=float, default=0.40, help="叠加到原图的暗化比例 (0~1)")

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    process(
        image_path=args.input,
        outdir=args.outdir,
        remove_devices=args.remove_devices,
        det_csv=args.det_csv,
        det_json=args.det_json,
        include_classes=args.include_classes,
        exclude_classes=args.exclude_classes,
        binarize_method=args.binarize,
        sauvola_win=args.sauvola_win,
        sauvola_k=args.sauvola_k,
        small_comp_ratio=args.small_comp_ratio,
        thicken_kernel=args.thicken,
        overlay_alpha=args.overlay_alpha,
    )
    print("[OK] 导线图已生成。输出目录:", args.outdir)


if __name__ == "__main__":
    main()

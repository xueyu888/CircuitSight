#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cabinet-wise wire-first topology extraction (border-intersection by default)
============================================================================

默认行为：
  1) 仅当导线像素与“器件框边界”相交，才认为该器件与该导线连接。
  2) 在做连通域前抠白所有器件框内部像素，避免“线穿过设备导致跨设备短接”。
  3) 不使用靠近判定；默认不膨胀线（line_dilate=0）。
  4) 叠加的设备间连线支持自定义颜色和粗细（默认较粗的蓝色）。

输出（每个 cabinet 子目录）：
  - bin.png, lines_h.png, lines_v.png, lines_union.png
  - labels_color.png（导线连通域着色：抠白后结果）
  - topology_overlay.png（原图叠加：粗蓝连线+框+编号+贴边短线）
  - topology_schematic_boxes.png（骨架+彩色框+简化符号）
  - topology_layout.png（圆环拓扑图，编号/配色一致）
  - graph.json（仅拓扑：节点/边，无坐标）
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
from skimage.filters import threshold_sauvola
from skimage.measure import label as cc_label
import yaml


# =============================
# 基本数据结构
# =============================

@dataclass(frozen=True)
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int
    cls: str
    score: float = 1.0
    track_id: Optional[str] = None

    def as_slice(self) -> Tuple[slice, slice]:
        return slice(self.y1, self.y2), slice(self.x1, self.x2)

    def clip(self, w: int, h: int) -> "BBox":
        x1 = max(0, min(self.x1, w - 1))
        y1 = max(0, min(self.y1, h - 1))
        x2 = max(0, min(self.x2, w))
        y2 = max(0, min(self.y2, h))
        if x2 <= x1:
            x2 = min(w, x1 + 1)
        if y2 <= y1:
            y2 = min(h, y1 + 1)
        return BBox(x1, y1, x2, y2, self.cls, self.score, self.track_id)

    def center(self) -> Tuple[int, int]:
        return int((self.x1 + self.x2) / 2), int((self.y1 + self.y2) / 2)

    def width(self) -> int:
        return int(self.x2 - self.x1)

    def height(self) -> int:
        return int(self.y2 - self.y1)

    def expanded(self, margin: int, w: int, h: int) -> "BBox":
        return BBox(
            self.x1 - margin,
            self.y1 - margin,
            self.x2 + margin,
            self.y2 + margin,
            self.cls,
            self.score,
            self.track_id,
        ).clip(w, h)


# =============================
# I/O 辅助
# =============================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def imwrite(path: str, img: NDArray[np.uint8]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


def to_gray(img: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def read_image(path: str) -> NDArray[np.uint8]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


# =============================
# 读取检测
# =============================

def load_dets_json(json_path: str) -> Dict[str, List[BBox]]:
    """Return mapping image_path -> [BBox,...] from generic JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    per_image: Dict[str, List[BBox]] = {}
    for d in data:
        ip = os.path.normpath(str(d["image_path"]))
        b = BBox(
            int(d["x1"]),
            int(d["y1"]),
            int(d["x2"]),
            int(d["y2"]),
            str(d["class"]),
            float(d.get("score", 1.0)),
            d.get("track_id"),
        )
        per_image.setdefault(ip, []).append(b)
    return per_image


def _read_yaml_names(yaml_path: str) -> Dict[int, str]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    if isinstance(y.get("names"), list):
        return {i: str(n) for i, n in enumerate(y["names"]) }
    return {int(k): str(v) for k, v in y.get("names", {}).items()}


def load_ultra_labels(labels_dir: str, images_root: str, names_yaml: str) -> Dict[str, List[BBox]]:
    """Parse Ultralytics normalized .txt labels -> pixel BBox with class names."""
    names = _read_yaml_names(names_yaml)
    per_image: Dict[str, List[BBox]] = {}
    for fname in os.listdir(labels_dir):
        if not fname.endswith(".txt"):
            continue
        stem = os.path.splitext(fname)[0]
        img_path: Optional[str] = None
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            cand = os.path.join(images_root, stem + ext)
            if os.path.exists(cand):
                img_path = cand
                break
        if img_path is None:
            for root, _dirs, files in os.walk(images_root):
                for f in files:
                    if os.path.splitext(f)[0] == stem and f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                        img_path = os.path.join(root, f)
                        break
                if img_path is not None:
                    break
        if img_path is None:
            continue
        img = read_image(img_path)
        h, w = img.shape[:2]
        boxes: List[BBox] = []
        with open(os.path.join(labels_dir, fname), "r", encoding="utf-8") as f:
            for ln in f:
                ps = ln.strip().split()
                if len(ps) < 5:
                    continue
                cid = int(float(ps[0]))
                cx, cy, bw, bh = map(float, ps[1:5])
                cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
                x1 = int(cx - bw / 2)
                y1 = int(cy - bh / 2)
                x2 = int(cx + bw / 2)
                y2 = int(cy + bh / 2)
                score = float(ps[5]) if len(ps) >= 6 else 1.0
                boxes.append(BBox(x1, y1, x2, y2, names.get(cid, str(cid)), score))
        per_image[os.path.normpath(img_path)] = boxes
    return per_image


# =============================
# 二值化 & 直线提取
# =============================

def binarize(gray: NDArray[np.uint8], method: str, sauvola_win: Optional[int], sauvola_k: float) -> NDArray[np.uint8]:
    """输出白底(255)/黑字(0)。"""
    if method.lower() == "sauvola":
        win = sauvola_win or max(81, int(min(gray.shape[:2]) * 0.025) | 1)
        th = threshold_sauvola(gray, window_size=win, k=sauvola_k)
        bw = (gray > th).astype(np.uint8) * 255
    else:
        _t, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(np.mean(bw)) < 128.0:
        bw = cv2.bitwise_not(bw)
    return bw


def extract_hv_lines(
    bw_white_bg_black_fg: NDArray[np.uint8],
    h_len: int,
    v_len: int,
) -> Tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]:
    """定向开运算抽取水平/垂直线。返回白底黑线。"""
    img = (bw_white_bg_black_fg == 0).astype(np.uint8) * 255
    h_len = max(1, int(h_len))
    v_len = max(1, int(v_len))
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    open_h = cv2.morphologyEx(img, cv2.MORPH_OPEN, k_h)
    open_v = cv2.morphologyEx(img, cv2.MORPH_OPEN, k_v)
    uni = cv2.max(open_h, open_v)

    def inv(x: NDArray[np.uint8]) -> NDArray[np.uint8]:
        out = np.full_like(x, 255, dtype=np.uint8)
        out[x > 0] = 0
        return out

    return inv(open_h), inv(open_v), inv(uni)


# =============================
# 工具：颜色解析
# =============================

def parse_bgr(s: Optional[str]) -> Tuple[int, int, int]:
    """
    支持 "B,G,R"（如 255,0,0）或十六进制 "#RRGGBB"（如 #007bff）。
    返回 OpenCV 用的 BGR 元组。
    """
    if not s:
        return (255, 0, 0)  # 默认蓝色 BGR
    s = s.strip()
    if s.startswith("#"):
        hexs = s.lstrip("#")
        if len(hexs) == 6:
            r = int(hexs[0:2], 16)
            g = int(hexs[2:4], 16)
            b = int(hexs[4:6], 16)
            return (b, g, r)
        raise ValueError("edge color hex must be #RRGGBB")
    if "," in s:
        parts = [int(p) for p in s.split(",")]
        if len(parts) == 3:
            return (parts[0], parts[1], parts[2])
        raise ValueError("edge-bgr must be 'B,G,R'")
    raise ValueError("edge-bgr must be 'B,G,R' or '#RRGGBB'")


# =============================
# 线-框相交（仅线驱动，边界相交 + 防跳接）
# =============================

def connected_labels(line_img_white_bg: NDArray[np.uint8]) -> NDArray[np.int32]:
    """对黑线做连通标记；返回 int32 标签图。"""
    fg = (line_img_white_bg == 0).astype(np.uint8)
    labels = cc_label(fg, connectivity=2)
    return labels.astype(np.int32)


def dilate_lines(line_img_white_bg: NDArray[np.uint8], radius: int) -> NDArray[np.uint8]:
    """radius <= 0 时不做任何膨胀，直接返回原图；否则白底黑线。"""
    if radius is None or int(radius) <= 0:
        return line_img_white_bg.copy()
    fg = (line_img_white_bg == 0).astype(np.uint8) * 255
    k = max(1, int(radius)) * 2 + 1
    elem = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    fg2 = cv2.dilate(fg, elem, iterations=1)
    out = np.full_like(line_img_white_bg, 255, dtype=np.uint8)
    out[fg2 > 0] = 0
    return out


def carve_lines_inside_devices(
    lines_white_bg: NDArray[np.uint8],
    devices: Sequence[BBox],
    carve_margin: int = 1,
) -> NDArray[np.uint8]:
    """
    将每个设备框的内部区域抠成白色，从而切断“穿过设备”的线连通性。
    carve_margin>0 会稍微缩小抠白区域，避免误抠到边界像素。
    """
    out = lines_white_bg.copy()
    H, W = out.shape[:2]
    m = max(0, int(carve_margin))
    for b in devices:
        x1 = max(0, min(W, b.x1 + m))
        y1 = max(0, min(H, b.y1 + m))
        x2 = max(0, min(W, b.x2 - m))
        y2 = max(0, min(H, b.y2 - m))
        if x2 > x1 and y2 > y1:
            out[y1:y2, x1:x2] = 255  # 白色=无线
    return out


def build_wire_hits_for_device(
    labels: NDArray[np.int32],
    lines_white_bg: NDArray[np.uint8],
    box: BBox,
    *,
    border_thick: int = 1,  # 框边界厚度（像素）
) -> Tuple[List[int], Dict[int, Tuple[int, int]]]:
    """
    仅以“框边界”与线的相交来判定连接：
      - 在框的边界（指定像素厚度）与线像素(黑)的重叠处读取连通域标签
      - 返回命中 label 列表及每个 label 的平均命中点（用于可视化贴线）
    """
    h, w = lines_white_bg.shape[:2]
    border_thick = max(1, int(border_thick))

    # 画出框的边界(单通道0/1)
    border = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(border, (box.x1, box.y1), (box.x2 - 1, box.y2 - 1), 1, thickness=border_thick)

    # 与线(黑像素)相交的位置
    ys, xs = np.where((border > 0) & (lines_white_bg == 0))
    if ys.size == 0:
        return [], {}

    labs = labels[ys, xs]
    uniq = [int(v) for v in np.unique(labs) if v != 0]

    hit_pts: Dict[int, Tuple[int, int]] = {}
    for lb in uniq:
        m = (labs == lb)
        cy = int(np.mean(ys[m]))
        cx = int(np.mean(xs[m]))
        hit_pts[lb] = (cx, cy)
    return uniq, hit_pts


def nearest_point_on_border(b: BBox, p: Tuple[int, int]) -> Tuple[int, int]:
    """求点 p 到矩形边界的最近点（正交投影）。"""
    x, y = p
    x = max(b.x1, min(x, b.x2))
    y = max(b.y1, min(y, b.y2))
    dl = abs(x - b.x1); dr = abs(b.x2 - x); dt = abs(y - b.y1); db = abs(b.y2 - y)
    m = min(dl, dr, dt, db)
    if m == dl:
        return (b.x1, y)
    if m == dr:
        return (b.x2, y)
    if m == dt:
        return (x, b.y1)
    return (x, b.y2)


def build_edges_wire_only(
    devices: Sequence[BBox],
    labels: NDArray[np.int32],
    lines_white_bg: NDArray[np.uint8],
    *,
    border_thick: int = 1,
) -> Tuple[List[Tuple[int, int]], Dict[int, Tuple[int, int]], Dict[int, List[Tuple[Tuple[int, int], Tuple[int, int]]]]]:
    """
    只依据导线构建边：两器件若共享同一条线的连通域(label)，且该线在各自框边界处与线相交 -> 连接
    另外返回 pin_centers 与 attach_segments（可视化贴线用）。
    """
    net2devs: Dict[int, List[int]] = {}
    pin_centers: Dict[int, Tuple[int, int]] = {}
    attach_segments: Dict[int, List[Tuple[Tuple[int, int], Tuple[int, int]]]] = {}

    for i, b in enumerate(devices):
        labs, pts = build_wire_hits_for_device(
            labels, lines_white_bg, b,
            border_thick=border_thick,
        )
        pin_centers[i] = b.center()

        segs: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        for _lb, hp in pts.items():
            tgt = nearest_point_on_border(b, hp)
            # L 形短段，用于可视化贴边（不影响拓扑判定）
            mid1 = (hp[0], tgt[1])
            mid2 = (tgt[0], hp[1])
            lseg = (hp, mid1) if abs(hp[0] - tgt[0]) > abs(hp[1] - tgt[1]) else (hp, mid2)
            segs.append((lseg[0], lseg[1]))
            segs.append((lseg[1], tgt))
        attach_segments[i] = segs

        for lb in labs:
            net2devs.setdefault(lb, []).append(i)

    edges_set: set[Tuple[int, int]] = set()
    for _lb, idxs in net2devs.items():
        if len(idxs) < 2:
            continue
        s = sorted(set(idxs))
        for a in range(len(s)):
            for b in range(a + 1, len(s)):
                edges_set.add((s[a], s[b]))
    return sorted(list(edges_set)), pin_centers, attach_segments


# =============================
# 可视化
# =============================

def class_abbr(cls: str) -> str:
    mapping = {
        "capacitor_variable": "C~", "capacitor_fixed": "C", "resistor": "R",
        "ground": "GND", "earth": "EARTH", "fuse": "FUSE", "fuse_indicator": "FI",
        "sw": "SW", "Transformer": "T", "3W_Transformer": "T3W",
        "Reactor": "L", "Arrester": "ARS", "Earth_Switch": "ESW",
        "motor_3phase": "M3", "indicator_lamp": "LAMP", "cabinet": "CAB",
    }
    return mapping.get(cls, cls[:8])


def idx_color(i: int, n: int) -> Tuple[int, int, int]:
    if n <= 0:
        return (0, 140, 255)
    hue = (i * 179 // max(1, n)) % 179
    hsv = np.uint8([[[hue, 180, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_device_symbol(canvas: NDArray[np.uint8], cls: str, p: Tuple[int, int], *, scale: int, color: Tuple[int, int, int] = (0, 0, 0)) -> None:
    x, y = int(p[0]), int(p[1]); s = max(10, int(scale)); t = max(1, s // 8)
    if cls in ("capacitor_fixed", "capacitor_variable"):
        cv2.line(canvas, (x - s // 2, y - s), (x - s // 2, y + s), color, t, cv2.LINE_AA)
        cv2.line(canvas, (x + s // 2, y - s), (x + s // 2, y + s), color, t, cv2.LINE_AA)
        if cls == "capacitor_variable":
            cv2.arrowedLine(canvas, (x - s, y - s), (x + s, y + s), color, t, tipLength=0.25)
    elif cls == "resistor":
        cv2.rectangle(canvas, (x - s, y - t * 2), (x + s, y + t * 2), color, t, cv2.LINE_AA)
    elif cls in ("ground", "earth"):
        cv2.line(canvas, (x, y - s // 2), (x, y + s // 2), color, t, cv2.LINE_AA)
        cv2.line(canvas, (x - s // 2, y + s // 2), (x + s // 2, y + s // 2), color, t, cv2.LINE_AA)
    elif cls in ("sw", "Earth_Switch"):
        cv2.line(canvas, (x - s, y), (x - t, y), color, t, cv2.LINE_AA)
        cv2.line(canvas, (x + t, y), (x + s, y - s // 2), color, t, cv2.LINE_AA)
    elif cls in ("Transformer", "3W_Transformer"):
        r = s // 2
        cv2.circle(canvas, (x - r, y), r, color, t, cv2.LINE_AA)
        cv2.circle(canvas, (x + r, y), r, color, t, cv2.LINE_AA)
    elif cls == "Reactor":
        for k in range(3):
            cx = x - s + k * s // 2
            cv2.ellipse(canvas, (cx, y), (s // 2, s // 3), 0, 0, 180, color, t, cv2.LINE_AA)
    elif cls == "fuse":
        cv2.line(canvas, (x - s, y), (x - s // 4, y), color, t, cv2.LINE_AA)
        cv2.rectangle(canvas, (x - s // 4, y - t * 2), (x + s // 4, y + t * 2), color, t, cv2.LINE_AA)
        cv2.line(canvas, (x + s // 4, y), (x + s, y), color, t, cv2.LINE_AA)
    elif cls in ("Arrester", "fuse_indicator"):
        cv2.rectangle(canvas, (x - s // 2, y - s), (x + s // 2, y + s), color, t, cv2.LINE_AA)
    elif cls in ("motor_3phase", "indicator_lamp"):
        r = s
        cv2.circle(canvas, (x, y), r, color, t, cv2.LINE_AA)
    else:
        cv2.circle(canvas, (x, y), max(5, t * 3), color, -1, cv2.LINE_AA)


def draw_idx_label(canvas: NDArray[np.uint8], text: str, p: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    cv2.putText(canvas, text, (p[0] + 6, p[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, (p[0] + 6, p[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def draw_topology_overlay(
    roi_bgr: NDArray[np.uint8],
    devices: Sequence[BBox],
    edges: Sequence[Tuple[int, int]],
    pin_centers: Mapping[int, Tuple[int, int]],
    node_colors: Sequence[Tuple[int, int, int]],
    attach_segments: Mapping[int, List[Tuple[Tuple[int, int], Tuple[int, int]]]],
    *,
    edge_color: Tuple[int, int, int] = (255, 0, 0),   # 默认蓝色(BGR)
    edge_thick: int = 4,                              # 默认比较粗
    attach_color: Tuple[int, int, int] = (160, 160, 160),
    attach_thick: int = 2,
) -> NDArray[np.uint8]:
    vis = roi_bgr.copy()
    # 贴合引导线（灰，细一点）
    for _i, segs in attach_segments.items():
        for s in segs:
            cv2.line(vis, s[0], s[1], attach_color, max(1, int(attach_thick)), cv2.LINE_AA)

    # 设备间连线（可配色/粗细）
    for i, j in edges:
        p1 = pin_centers.get(i, devices[i].center())
        p2 = pin_centers.get(j, devices[j].center())
        cv2.line(vis, p1, p2, edge_color, max(1, int(edge_thick)), cv2.LINE_AA)

    # 框和标签
    for i, b in enumerate(devices):
        col = node_colors[i]
        cv2.rectangle(vis, (b.x1, b.y1), (b.x2, b.y2), col, 2, cv2.LINE_AA)
        draw_idx_label(vis, f"{i}:{class_abbr(b.cls)}", (b.x1, b.y1), col)
    return vis


def draw_topology_schematic_boxes(
    skeleton_white_bg: NDArray[np.uint8],
    devices: Sequence[BBox],
    node_colors: Sequence[Tuple[int, int, int]],
    *, box_margin: int = 0, symbol_rel: float = 0.33, skel_dilate: int = 2
) -> NDArray[np.uint8]:
    H, W = skeleton_white_bg.shape[:2]
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)
    fg = (skeleton_white_bg == 0).astype(np.uint8) * 255
    if skel_dilate > 1:
        k = skel_dilate if skel_dilate % 2 == 1 else skel_dilate + 1
        elem = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        fg = cv2.dilate(fg, elem, iterations=1)
    canvas[fg > 0] = (220, 220, 220)

    for i, b in enumerate(devices):
        bb = BBox(b.x1 - box_margin, b.y1 - box_margin, b.x2 + box_margin, b.y2 + box_margin, b.cls)
        col = node_colors[i]
        cv2.rectangle(canvas, (bb.x1, bb.y1), (bb.x2, bb.y2), col, 2, cv2.LINE_AA)
        cx, cy = (bb.x1 + bb.x2) // 2, (bb.y1 + bb.y2) // 2
        scale = int(max(10, min(bb.width(), bb.height()) * float(np.clip(symbol_rel, 0.1, 0.6))))
        draw_device_symbol(canvas, b.cls, (cx, cy), scale=scale, color=(0, 0, 0))
        draw_idx_label(canvas, f"{i}:{class_abbr(b.cls)}", (bb.x1, bb.y1), col)
    return canvas


def draw_topology_layout(
    devices: Sequence[BBox],
    edges: Sequence[Tuple[int, int]],
    node_colors: Sequence[Tuple[int, int, int]],
    min_size: int = 800,
) -> NDArray[np.uint8]:
    n = len(devices)
    if n == 0:
        return np.full((min_size, min_size, 3), 255, dtype=np.uint8)
    size = max(min_size, int(28 * n))
    W, H = size, size
    cx, cy = W // 2, H // 2
    R = int(min(W, H) * 0.38)
    img = np.full((H, W, 3), 255, dtype=np.uint8)
    pts: List[Tuple[int, int]] = []
    for k in range(n):
        ang = 2.0 * math.pi * (k / n)
        x = int(cx + R * math.cos(ang))
        y = int(cy + R * math.sin(ang))
        pts.append((x, y))
    for i, j in edges:
        if 0 <= i < n and 0 <= j < n:
            cv2.line(img, pts[i], pts[j], (190, 190, 190), 1, cv2.LINE_AA)
    for k, p in enumerate(pts):
        col = node_colors[k]
        cv2.circle(img, p, 5, col, -1, cv2.LINE_AA)
        vx, vy = p[0] - cx, p[1] - cy
        L = math.hypot(vx, vy) + 1e-6
        ox = int(p[0] + 12 * vx / L)
        oy = int(p[1] + 12 * vy / L)
        cv2.putText(img, str(k), (ox, oy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, str(k), (ox, oy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv2.LINE_AA)
        lab = class_abbr(devices[k].cls)
        cv2.putText(img, lab, (ox + 16, oy + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
    return img


# =============================
# 每个柜体处理
# =============================

def process_cabinet(
    full_img: NDArray[np.uint8],
    img_path: str,
    cab_box: BBox,
    device_boxes: Sequence[BBox],
    outdir: str,
    *,
    pad: int,
    bin_method: str,
    sauvola_win: Optional[int],
    sauvola_k: float,
    h_kernel: Optional[int],
    v_kernel: Optional[int],
    device_margin: int,   # 兼容占位；默认0未用
    probe_gap: int,       # 兼容占位；默认0未用
    probe_width: int,     # 在新逻辑中作为“边界厚度”
    line_dilate: int,
    box_margin: int,
    symbol_rel: float,
    schem_skel_dilate: int,
    layout_min_size: int,
    overlay_edge_color: Tuple[int, int, int],
    overlay_edge_thick: int,
) -> None:
    H, W = full_img.shape[:2]
    cab = cab_box.expanded(pad, W, H)
    roi = full_img[cab.y1:cab.y2, cab.x1:cab.x2].copy()
    gray = to_gray(roi)

    # 1) 二值化
    bw = binarize(gray, bin_method, sauvola_win, sauvola_k)
    imwrite(os.path.join(outdir, "bin.png"), bw)

    # 2) HV 抽线
    Hc, Wc = bw.shape[:2]
    h_len = h_kernel if h_kernel is not None else max(25, int(Wc * 0.015))
    v_len = v_kernel if v_kernel is not None else max(25, int(Hc * 0.015))
    lines_h, lines_v, lines_union = extract_hv_lines(bw, h_len, v_len)
    imwrite(os.path.join(outdir, "lines_h.png"), lines_h)
    imwrite(os.path.join(outdir, "lines_v.png"), lines_v)
    imwrite(os.path.join(outdir, "lines_union.png"), lines_union)

    # 3) 收集柜内器件（排除 cabinet 自身）
    inside: List[BBox] = []
    for b in device_boxes:
        cx, cy = b.center()
        if cab.x1 <= cx < cab.x2 and cab.y1 <= cy < cab.y2 and b.cls.lower() not in ("cabinet", "cabinets"):
            inside.append(BBox(b.x1 - cab.x1, b.y1 - cab.y1, b.x2 - cab.x1, b.y2 - cab.y1, b.cls, b.score, b.track_id))

    node_colors = [idx_color(i, len(inside)) for i in range(len(inside))]

    # 4) 线图用于相交：默认不膨胀（line_dilate=0）
    if line_dilate > 0:
        lines_for_intersection = dilate_lines(lines_union, line_dilate)
        imwrite(os.path.join(outdir, "lines_dilated.png"), lines_for_intersection)
    else:
        lines_for_intersection = lines_union.copy()

    # 5) 先把设备内部的线抠掉，再做连通域（防止“跳框”）
    lines_for_labeling = carve_lines_inside_devices(lines_for_intersection, inside, carve_margin=1)
    labels = connected_labels(lines_for_labeling)
    if labels.max() > 0:
        rng = np.random.default_rng(0)
        colors = (rng.integers(0, 255, size=(labels.max() + 1, 3))).astype(np.uint8)
        colors[0] = np.array([255, 255, 255], dtype=np.uint8)
        imwrite(os.path.join(outdir, "labels_color.png"), colors[labels])

    # 6) 仅以“框边界与线相交”构建边（wire-only, border-based）
    edges, pin_centers, attach_segments = build_edges_wire_only(
        inside, labels, lines_for_intersection,
        border_thick=max(1, int(probe_width))  # 用 probe-width 作为边界厚度
    )

    # 7) 叠加图（原图 + 框 + 编号 + 连线）
    overlay = draw_topology_overlay(
        roi, inside, edges, pin_centers, node_colors, attach_segments,
        edge_color=overlay_edge_color, edge_thick=int(overlay_edge_thick)
    )
    imwrite(os.path.join(outdir, "topology_overlay.png"), overlay)

    # 8) 框+符号+骨架
    schem = draw_topology_schematic_boxes(
        lines_union, inside, node_colors, box_margin=box_margin,
        symbol_rel=symbol_rel, skel_dilate=schem_skel_dilate
    )
    imwrite(os.path.join(outdir, "topology_schematic_boxes.png"), schem)

    # 9) 圆环拓扑
    layout = draw_topology_layout(inside, edges, node_colors, min_size=layout_min_size)
    imwrite(os.path.join(outdir, "topology_layout.png"), layout)

    # 10) JSON（仅拓扑：节点/边）
    nodes = [{"id": i, "class": b.cls} for i, b in enumerate(inside)]
    edges_json = [{"u": int(u), "v": int(v)} for (u, v) in edges]
    with open(os.path.join(outdir, "graph.json"), "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "edges": edges_json}, f, ensure_ascii=False, indent=2)


# =============================
# 任务编排
# =============================

def is_image_file(p: str) -> bool:
    low = p.lower()
    return low.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))


def list_images(path: str) -> List[str]:
    out: List[str] = []
    if os.path.isdir(path):
        for root, _dirs, files in os.walk(path):
            for f in files:
                if is_image_file(f):
                    out.append(os.path.join(root, f))
    else:
        if is_image_file(path):
            out.append(path)
    return sorted(out)


def process_one_image(
    img_path: str,
    dets_map: Dict[str, List[BBox]],
    out_base: str,
    *,
    bin_method: str,
    sauvola_win: Optional[int],
    sauvola_k: float,
    h_kernel: Optional[int],
    v_kernel: Optional[int],
    pad: int,
    device_margin: int,
    probe_gap: int,
    probe_width: int,
    line_dilate: int,
    box_margin: int,
    symbol_rel: float,
    schem_skel_dilate: int,
    layout_size: int,
    cabinet_class_name: str,
    overlay_edge_color: Tuple[int, int, int],
    overlay_edge_thick: int,
) -> None:
    img = read_image(img_path)
    H, W = img.shape[:2]
    dets = dets_map.get(os.path.normpath(img_path), [])

    cabs: List[BBox] = []
    devs: List[BBox] = []
    for b in dets:
        if b.cls.lower() in (cabinet_class_name.lower(), "cabinets"):
            cabs.append(b.clip(W, H))
        else:
            devs.append(b.clip(W, H))

    if not cabs:
        cabs = [BBox(0, 0, W, H, cabinet_class_name, 1.0, None)]

    for idx, cab in enumerate(cabs):
        subdir = os.path.join(out_base, os.path.splitext(os.path.basename(img_path))[0], f"cabinet_{idx:02d}")
        ensure_dir(subdir)
        process_cabinet(
            img, img_path, cab, devs, subdir,
            pad=pad,
            bin_method=bin_method,
            sauvola_win=sauvola_win,
            sauvola_k=sauvola_k,
            h_kernel=h_kernel,
            v_kernel=v_kernel,
            device_margin=device_margin,
            probe_gap=probe_gap,
            probe_width=probe_width,
            line_dilate=line_dilate,
            box_margin=box_margin,
            symbol_rel=symbol_rel,
            schem_skel_dilate=schem_skel_dilate,
            layout_min_size=layout_size,
            overlay_edge_color=overlay_edge_color,
            overlay_edge_thick=overlay_edge_thick,
        )


# =============================
# CLI
# =============================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Wire-first cabinet topology extractor (border-intersection only by default)")
    ap.add_argument("--input", required=True, help="Image file or directory")
    ap.add_argument("--outdir", required=True, help="Output directory")

    # detections
    ap.add_argument("--det-json", default=None, help="Detections JSON")
    ap.add_argument("--ultra-labels", default=None, help="Ultralytics labels dir (.txt)")
    ap.add_argument("--images-root", default=None, help="Images root for labels")
    ap.add_argument("--names-yaml", default=None, help="YAML with `names:`")
    ap.add_argument("--cabinet-class", default="cabinet", help="Cabinet class name (e.g., 'cabinet' or 'cabinets')")

    # binarize
    ap.add_argument("--binarize", choices=["otsu", "sauvola"], default="sauvola")
    ap.add_argument("--sauvola-win", type=int, default=None)
    ap.add_argument("--sauvola-k", type=float, default=0.16)

    # line kernels
    ap.add_argument("--h-kernel", type=int, default=None)
    ap.add_argument("--v-kernel", type=int, default=None)

    # geometry / border
    ap.add_argument("--pad", type=int, default=48, help="Cabinet crop padding")

    # 兼容旧参数（默认关闭，不再使用探测环；仅保留以不破坏 CLI）
    ap.add_argument("--device-margin", type=int, default=0, help="(兼容) 框扩张基础边距（默认0，未使用）")
    ap.add_argument("--probe-gap", type=int, default=0, help="(兼容) 探测环相对框外的间隙（默认0，未使用）")

    # 新：用作“框边界厚度”
    ap.add_argument("--probe-width", type=int, default=1, help="框边界厚度（像素）。仅用于边界相交判定")

    # 线膨胀（默认关闭）
    ap.add_argument("--line-dilate", type=int, default=0, help="线膨胀半径：>0才启用（默认0=关闭）")

    # visualization
    ap.add_argument("--box-margin", type=int, default=0)
    ap.add_argument("--symbol-rel", type=float, default=0.33)
    ap.add_argument("--schem-skel-dilate", type=int, default=2)
    ap.add_argument("--layout-size", type=int, default=1000)

    # overlay edge appearance
    ap.add_argument("--edge-bgr", type=str, default="255,0,0",
                    help="叠加连线的颜色(B,G,R)或#RRGGBB，默认蓝色")
    ap.add_argument("--edge-thick", type=int, default=4,
                    help="叠加连线粗细(像素)，默认4")

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.outdir)

    overlay_edge_color = parse_bgr(args.edge_bgr)
    overlay_edge_thick = max(1, int(args.edge_thick))

    if args.det_json:
        dets_map = load_dets_json(args.det_json)
    elif args.ultra_labels and args.images_root and args.names_yaml:
        dets_map = load_ultra_labels(args.ultra_labels, args.images_root, args.names_yaml)
    else:
        raise SystemExit("Please provide either --det-json or (--ultra-labels + --images-root + --names-yaml)")

    images = list_images(args.input)
    if not images:
        raise SystemExit("No images found under --input")

    for ip in images:
        process_one_image(
            ip,
            dets_map,
            args.outdir,
            bin_method=args.binarize,
            sauvola_win=args.sauvola_win,
            sauvola_k=args.sauvola_k,
            h_kernel=args.h_kernel,
            v_kernel=args.v_kernel,
            pad=args.pad,
            device_margin=args.device_margin,
            probe_gap=args.probe_gap,
            probe_width=args.probe_width,
            line_dilate=args.line_dilate,
            box_margin=args.box_margin,
            symbol_rel=args.symbol_rel,
            schem_skel_dilate=args.schem_skel_dilate,
            layout_size=args.layout_size,
            cabinet_class_name=args.cabinet_class,
            overlay_edge_color=overlay_edge_color,
            overlay_edge_thick=overlay_edge_thick,
        )
        print(f"[OK] Processed: {ip}")


if __name__ == "__main__":
    main()

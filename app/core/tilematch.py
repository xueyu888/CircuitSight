from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, List
import io
import base64
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg") # 必须加这个防止 WSL 报错
import matplotlib.pyplot as plt
from app.config import settings

@dataclass
class TileMatchResult:
    score: float              
    best_shift: Tuple[int, int]
    heatmap_b64: str     


class TileMatcher:
    def __init__(self):
        # 梯度算子 (Sobel)
        self._kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        self._ky = np.array([[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]], dtype=np.float32)

    def _to_gray(self, img: Image.Image) -> np.ndarray:
        # 假设输入已经是反色后的黑底白线
        # 转为 0-1 float
        return np.asarray(img.convert("L"), dtype=np.float32) / 255.0

    def _conv2d(self, img: np.ndarray, k: np.ndarray) -> np.ndarray:
        kh, kw = k.shape
        ph, pw = kh // 2, kw // 2
        padded = np.pad(img, ((ph, ph), (pw, pw)), mode="constant")
        out = np.zeros_like(img)
        # 简单的滑窗卷积，性能一般但无需额外库
        # 为了加速，可以用 scipy.signal.convolve2d，这里手写虽然慢但依赖少
        # 考虑到图片尺寸只有 512x192，纯 numpy 还行
        
        # 优化版：利用 numpy 的 view window 加速，或者直接用 opencv filter2D
        import cv2
        return cv2.filter2D(img, -1, k)

    def compute_hog_feature(self, pil_img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        gray = self._to_gray(pil_img)
        gx = self._conv2d(gray, self._kx); gy = self._conv2d(gray, self._ky)
        mag = np.sqrt(gx**2 + gy**2) + 1e-6
        ang = (np.arctan2(gy, gx) + np.pi)

        bins = settings.ORI_BINS
        bin_idx = (ang / (2 * np.pi) * bins).astype(np.int32)
        bin_idx = np.clip(bin_idx, 0, bins - 1)

        H, W = gray.shape
        gx_n, gy_n = settings.GRID_X, settings.GRID_Y
        cw, ch = max(1, W // gx_n), max(1, H // gy_n)

        feats  = np.zeros((gy_n, gx_n, bins), dtype=np.float32)
        energy = np.zeros((gy_n, gx_n), dtype=np.float32)

        for cy in range(gy_n):
            for cx in range(gx_n):
                ys, xs = cy * ch, cx * cw
                ye = H if cy == gy_n - 1 else ys + ch
                xe = W if cx == gx_n - 1 else xs + cw
                b_cell = bin_idx[ys:ye, xs:xe]
                m_cell = mag[ys:ye, xs:xe]

                e = m_cell.sum()
                energy[cy, cx] = e

                if e < 1e-4:   # 空白格子：保持 0 向量
                    continue

                hist = np.zeros((bins,), dtype=np.float32)
                for bi in range(bins):
                    hist[bi] = m_cell[b_cell == bi].sum()

                feats[cy, cx, :] = hist / (np.linalg.norm(hist) + 1e-6)

        return feats, energy


    def match_features(self, feat_A, feat_B, ene_A=None, ene_B=None) -> TileMatchResult:
        if isinstance(feat_A, tuple): feat_A, ene_A = feat_A
        if isinstance(feat_B, tuple): feat_B, ene_B = feat_B

        Gy, Gx, _ = feat_A.shape
        max_shift = settings.MAX_SHIFT

        best_score, best_shift, best_sim_map = -1.0, (0, 0), None

        for dy in range(-max_shift, max_shift + 1):
            for dx in range(-max_shift, max_shift + 1):
                ys0, ye0 = max(0, dy), min(Gy, Gy + dy)
                xs0, xe0 = max(0, dx), min(Gx, Gx + dx)
                ys1, ye1 = max(0, -dy), min(Gy, Gy - dy)
                xs1, xe1 = max(0, -dx), min(Gx, Gx - dx)

                if ye0 - ys0 <= 0 or xe0 - xs0 <= 0: 
                    continue

                A = feat_A[ys0:ye0, xs0:xe0]
                B = feat_B[ys1:ye1, xs1:xe1]

                sim = (A * B).sum(axis=-1)     # 余弦（已归一化）
                if ene_A is not None and ene_B is not None:
                    EA = ene_A[ys0:ye0, xs0:xe0]
                    EB = ene_B[ys1:ye1, xs1:xe1]
                    w = np.minimum(EA, EB)
                    mask = w > 1e-4            # 只看“有边”的格子
                    if mask.any():
                        score = float((sim[mask] * w[mask]).sum() / (w[mask].sum() + 1e-6))
                    else:
                        score = 0.0
                    # 用于可视化的热力图（空格子记为 -1）
                    sim_map = np.full_like(sim, -1.0)
                    sim_map[mask] = sim[mask]
                else:
                    score = float(sim.mean())
                    sim_map = sim

                if score > best_score:
                    best_score, best_shift, best_sim_map = score, (dy, dx), sim_map

        heatmap_b64 = ""
        if best_sim_map is not None:
            heatmap_b64 = self._render_heatmap(best_sim_map)

        return TileMatchResult(score=max(0.0, best_score), best_shift=best_shift, heatmap_b64=heatmap_b64)


    def _render_heatmap(self, sim_map: np.ndarray) -> str:
        # 拉伸对比度以便观察
        clip_val = settings.HEATMAP_CLIP
        m = np.clip((sim_map - (1 - clip_val)) / clip_val, 0, 1)
        
        fig, ax = plt.subplots(figsize=(4, 3))
        im = ax.imshow(m, vmin=0, vmax=1, cmap="jet") # jet 彩色更直观
        ax.axis('off')
        # fig.colorbar(im, ax=ax) 
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
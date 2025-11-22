import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import cv2
from PIL import Image
from sklearn.decomposition import PCA
from typing import List

class Visualizer:
    @staticmethod
    def _pil_to_base64(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def draw_feature_matches(self, match_data: dict) -> str:
        if match_data.get("imgs") is None: return ""

        img1, img2 = match_data["imgs"]
        kp1, kp2 = match_data["kps"]
        good_matches = match_data["debug_matches"]
        mask = match_data.get("debug_mask")
        grid_data = match_data.get("debug_grids")

        h1, w1 = img1.shape
        h2, w2 = img2.shape
        
        vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        vis[:h1, :w1, :] = np.dstack([img1]*3)
        vis[:h2, w1:w1+w2, :] = np.dstack([img2]*3)

        # === 1. 绘制网格 (Overlay) ===
        if grid_data:
            grid_size = grid_data["grid_size"]
            grid_q = grid_data["q_active"]
            grid_r = grid_data["r_active"]
            match_q = grid_data.get("q_matched")
            match_r = grid_data.get("r_matched")
            
            overlay = vis.copy()
            
            COLOR_HIT = (0, 255, 0)    # 绿块
            COLOR_MISS = (255, 0, 0)   # 蓝块 (注意cv2是BGR)
            COLOR_GRID = (80, 80, 80)  # 灰线
            
            rows, cols = grid_q.shape
            for r in range(rows):
                for c in range(cols):
                    y1 = r * grid_size
                    x1 = c * grid_size
                    y2 = y1 + grid_size
                    x2 = x1 + grid_size
                    
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_GRID, 1)
                    cv2.rectangle(overlay, (x1+w1, y1), (x2+w1, y2), COLOR_GRID, 1)
                    
                    if grid_q[r, c] > 0:
                        color = COLOR_HIT if match_q[r, c] > 0 else COLOR_MISS
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                    if grid_r[r, c] > 0:
                        color = COLOR_HIT if match_r[r, c] > 0 else COLOR_MISS
                        cv2.rectangle(overlay, (x1+w1, y1), (x2+w1, y2), color, -1)

            cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)

        # === 2. 绘制所有特征点 (红色 = 未匹配/被过滤) ===
        # COLOR_UNMATCHED = (0, 0, 255) # Red (BGR)
        # 稍微小一点，radius=2
        for kp in kp1:
            cv2.circle(vis, (int(kp.pt[0]), int(kp.pt[1])), 2, (0, 0, 255), -1)
        for kp in kp2:
            cv2.circle(vis, (int(kp.pt[0]) + w1, int(kp.pt[1])), 2, (0, 0, 255), -1)

        # === 3. 绘制匹配线 & 端点 (黄色 = 匹配成功) ===
        COLOR_LINE = (255, 255, 255) # 白线
        COLOR_MATCH = (0, 255, 255)  # 黄点 (BGR: B=0, G=255, R=255)
        
        if mask is not None and len(good_matches) > 0:
            matchesMask = mask.ravel().tolist()
            for i, (m, inlier) in enumerate(zip(good_matches, matchesMask)):
                if inlier:
                    p1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1]))
                    p2 = (int(kp2[m.trainIdx].pt[0]) + w1, int(kp2[m.trainIdx].pt[1]))
                    
                    cv2.line(vis, p1, p2, COLOR_LINE, 1, cv2.LINE_AA)
                    
                    # 画大一点的黄点覆盖红点
                    cv2.circle(vis, p1, 4, COLOR_MATCH, -1)
                    cv2.circle(vis, p2, 4, COLOR_MATCH, -1)

        cv2.line(vis, (w1, 0), (w1, max(h1, h2)), (255, 255, 255), 2)
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        return self._pil_to_base64(Image.fromarray(vis_rgb))
    
    def plot_global_pca(self, *args, **kwargs): return ""
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
    def _fig_to_base64(fig: plt.Figure) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    @staticmethod
    def _pil_to_base64(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def draw_feature_matches(self, match_data: dict) -> str:
        """
        绘制特征匹配连线图
        """
        # 如果没有匹配数据或匹配失败
        if match_data["score"] == 0.0 or match_data.get("imgs") is None:
            return ""

        img1, img2 = match_data["imgs"]
        kp1, kp2 = match_data["kps"]
        good_matches = match_data["debug_matches"]
        mask = match_data.get("debug_mask")

        # 将 mask 转换为 list 用于 drawMatches
        matchesMask = mask.ravel().tolist() if mask is not None else None

        # 绘制参数
        draw_params = dict(matchColor=(0, 255, 0), # Inliers 用绿色
                           singlePointColor=None,
                           matchesMask=matchesMask, # 只画 Inliers
                           flags=2)

        res_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)
        
        # 转 PIL -> Base64
        pil_img = Image.fromarray(res_img)
        return self._pil_to_base64(pil_img)

    def plot_global_pca(self, rule_embs: np.ndarray, rule_labels: List[str], 
                        query_embs: np.ndarray, query_filenames: List[str]) -> str:
        # (保持不变)
        if len(rule_embs) < 3: return ""
        X = np.vstack([rule_embs, query_embs])
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        n_rules = len(rule_embs)
        rule_pts = X_pca[:n_rules]
        query_pts = X_pca[n_rules:]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(rule_pts[:, 0], rule_pts[:, 1], c='blue', alpha=0.3, s=50, label='Rules Base')
        for i, txt in enumerate(rule_labels):
            ax.text(rule_pts[i, 0], rule_pts[i, 1], txt, fontsize=7, alpha=0.5, color='blue')
        ax.scatter(query_pts[:, 0], query_pts[:, 1], c='red', marker='*', s=200, label='Query Result')
        for i, txt in enumerate(query_filenames):
            ax.text(query_pts[i, 0], query_pts[i, 1], txt, fontsize=9, fontweight='bold', color='darkred')
        ax.set_title("Vector Space Distribution (PCA)")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()
        return self._fig_to_base64(fig)
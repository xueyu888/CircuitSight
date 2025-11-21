import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFont, ImageFilter
from sklearn.decomposition import PCA
from typing import List, Tuple, Optional
from app.config import settings

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

    def _simulate_preprocess(self, img: Image.Image) -> Image.Image:
        """模拟模型预处理：转黑底白线"""
        img = img.convert("RGB")
        # 1. 二值化
        gray = img.convert("L")
        bw = gray.point(lambda x: 255 if x > settings.BINARIZE_THRESHOLD else 0)
        img = bw.convert("RGB")
        
        # 2. 线条加粗 (让视觉更明显)
        w, h = img.size
        if min(w, h) > 500:
            img = img.filter(ImageFilter.MinFilter(3))
            
        # 3. 反色 (黑底白线)
        img = ImageOps.invert(img)
        
        # 4. Resize & Pad
        target_h, target_w = settings.INPUT_RESOLUTION
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)
        
        final_img = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        final_img.paste(img_resized, (paste_x, paste_y))
        
        return final_img

    def draw_side_by_side_comparison(self, 
                                     query_img: Image.Image, 
                                     ref_img: Image.Image, 
                                     query_vec: np.ndarray, 
                                     ref_vec: np.ndarray,
                                     ref_label: str) -> str:
        """
        绘制左右对比图，并应用像素密度惩罚。
        """
        # 1. 预处理两张图
        vis_q = self._simulate_preprocess(query_img)
        vis_r = self._simulate_preprocess(ref_img)
        
        w, h = vis_q.size
        
        # 2. 创建画布
        gap = 10
        canvas_w = w * 2 + gap
        canvas_h = h + 30
        canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
        
        canvas.paste(vis_q, (0, 30))
        canvas.paste(vis_r, (w + gap, 30))
        
        draw = ImageDraw.Draw(canvas)
        
        # 字体
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()

        draw.text((10, 5), "Query (Input)", fill="white", font=title_font)
        draw.text((w + gap + 10, 5), f"Match: {ref_label}", fill="#4CAF50", font=title_font)

        # 3. 计算网格
        rows, cols = settings.GRID_ROWS, settings.GRID_COLS
        total_dim = query_vec.shape[0]
        patch_dim = total_dim // (rows * cols)
        
        step_w = w / cols
        step_h = h / rows
        
        for r in range(rows):
            for c in range(cols):
                # === A. 向量相似度 ===
                idx = r * cols + c
                start = idx * patch_dim
                end = (idx + 1) * patch_dim
                v_q = query_vec[start:end]
                v_r = ref_vec[start:end]
                
                norm_q = np.linalg.norm(v_q)
                norm_r = np.linalg.norm(v_r)
                
                if norm_q > 0 and norm_r > 0:
                    cosine = np.dot(v_q, v_r) / (norm_q * norm_r)
                    vec_score = max(0, cosine) * 100
                else:
                    vec_score = 0.0

                # === B. 像素密度校验 (关键改进) ===
                # 1. 裁剪出对应的格子区域
                box_local = (int(c * step_w), int(r * step_h), int((c+1) * step_w), int((r+1) * step_h))
                patch_q = vis_q.crop(box_local)
                patch_r = vis_r.crop(box_local)
                
                # 2. 计算非零像素量 (亮度总和，或者非黑像素数)
                # 转换为 numpy 数组计算更并在
                arr_q = np.array(patch_q.convert("L"))
                arr_r = np.array(patch_r.convert("L"))
                
                # 统计亮度大于 30 的像素点数量 (忽略极暗噪点)
                pixels_q = np.count_nonzero(arr_q > 30)
                pixels_r = np.count_nonzero(arr_r > 30)
                
                # 3. 计算密度差异比率
                # 加上 1e-6 防止除以 0
                content_ratio = min(pixels_q, pixels_r) / (max(pixels_q, pixels_r) + 1e-6)
                
                # === C. 综合评分 ===
                final_score = vec_score
                
                # 如果两边内容量差异巨大 (比如一个有图，一个全黑)，强制扣分
                # 阈值：如果较少的一方像素量不到较多一方的 30%
                if max(pixels_q, pixels_r) > 100: # 只有当至少有一方有内容时才校验
                    if content_ratio < 0.3: 
                        # 严重不匹配：直接把分数打折
                        # 例如：vec_score=88, ratio=0.01 (全黑 vs 有图) -> final ≈ 8
                        final_score = final_score * (content_ratio * 2.0) 
                        # *2.0 是稍微给点面子，不直接乘0.01，但也足够把分数拉红
                
                # 限制在 0-100
                final_score = min(100, max(0, final_score))

                # === D. 绘图 ===
                color = "#4CAF50" if final_score > 80 else "#F44336"
                
                # 左框
                left_q = c * step_w
                top_q = 30 + r * step_h
                draw.rectangle([left_q, top_q, left_q + step_w, top_q + step_h], outline=color, width=2)
                
                # 右框
                left_r = w + gap + c * step_w
                top_r = 30 + r * step_h
                draw.rectangle([left_r, top_r, left_r + step_w, top_r + step_h], outline=color, width=2)
                
                # 写分值
                text = f"{final_score:.0f}%"
                
                # 绘制带背景的文字，保证清晰
                def draw_text_centered(cx, cy, txt, col):
                    # 简易居中：假设文字宽 40 高 20
                    x = cx - 20
                    y = cy - 10
                    draw.text((x, y), txt, fill=col, font=font, stroke_width=2, stroke_fill="black")

                draw_text_centered(left_q + step_w/2, top_q + step_h/2, text, color)
                draw_text_centered(left_r + step_w/2, top_r + step_h/2, text, color)

        return self._pil_to_base64(canvas)

    def plot_global_pca(self, rule_embs: np.ndarray, rule_labels: List[str], 
                        query_embs: np.ndarray, query_filenames: List[str]) -> str:
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
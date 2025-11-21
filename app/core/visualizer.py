import matplotlib
# 必须在 pyplot 导入前设置
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from sklearn.decomposition import PCA
from typing import List, Tuple

class Visualizer:
    @staticmethod
    def _fig_to_base64(fig: plt.Figure) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    def plot_top_matches(self, filename: str, matches: List[Tuple[str, float]], threshold_score: float) -> str:
        """
        绘制 Top-N 匹配柱状图 (百分比模式)
        matches: List[(label, score)] -> score 是 0-100 的数值
        threshold_score: 判定阈值 (例如 68.0)
        """
        if not matches:
            return ""
            
        labels = [m[0] for m in matches]
        scores = [m[1] for m in matches]
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # 逻辑：分数 > 阈值 显示绿色(Trusted)，否则灰色
        colors = ['#4CAF50' if s >= threshold_score else '#9E9E9E' for s in scores]
        
        bars = ax.bar(range(len(scores)), scores, color=colors)
        
        # 绘制阈值线
        ax.axhline(y=threshold_score, color='r', linestyle='--', linewidth=1, label=f'Threshold {threshold_score:.1f}%')
        
        # 设置 Y 轴为 0-110 (留点顶部空间)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Similarity Confidence (%)")
        ax.set_title(f"Top Matches for: {filename}")
        ax.legend(loc='upper right')
        
        # X 轴标签
        ax.set_xticks(range(len(scores)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # 在柱子上标数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

        return self._fig_to_base64(fig)

    def plot_global_pca(self, rule_embs: np.ndarray, rule_labels: List[str], 
                        query_embs: np.ndarray, query_filenames: List[str]) -> str:
        """绘制全局 PCA 散点图"""
        if len(rule_embs) < 3:
            return ""

        X = np.vstack([rule_embs, query_embs])
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        n_rules = len(rule_embs)
        rule_pts = X_pca[:n_rules]
        query_pts = X_pca[n_rules:]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 规则库点
        ax.scatter(rule_pts[:, 0], rule_pts[:, 1], c='blue', alpha=0.3, s=50, label='Rules Base')
        for i, txt in enumerate(rule_labels):
            ax.text(rule_pts[i, 0], rule_pts[i, 1], txt, fontsize=7, alpha=0.5, color='blue')
            
        # 查询点
        ax.scatter(query_pts[:, 0], query_pts[:, 1], c='red', marker='*', s=200, label='Query Result')
        for i, txt in enumerate(query_filenames):
            ax.text(query_pts[i, 0], query_pts[i, 1], txt, fontsize=9, fontweight='bold', color='darkred')
            
        ax.set_title("Vector Space Distribution (PCA)")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()
        
        return self._fig_to_base64(fig)
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


class EmbeddingVisualizer:
    """
    提供 embedding 的 PCA / t-SNE / 距离矩阵 / 雷达图 / query 对 rules 的可视化。
    所有图片输出到 output/embeddings 目录。
    """

    def __init__(self, out_dir: str = "output/embeddings") -> None:
        self.out_dir = out_dir
        ensure_dir(out_dir)

    def _save(self, fig: plt.Figure, filename: str) -> None:
        out_path = f"{self.out_dir}/{filename}"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {out_path}")

    # ====================
    # PCA 2D
    # ====================
    def plot_pca_2d(
        self,
        emb_dict: Dict[str, np.ndarray],
        filename: str = "pca.png",
    ) -> None:
        paths = list(emb_dict.keys())
        X = np.stack(list(emb_dict.values()), axis=0)

        pca = PCA(n_components=2)
        pts = pca.fit_transform(X)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(pts[:, 0], pts[:, 1], s=80)

        for i, p in enumerate(paths):
            ax.text(pts[i, 0], pts[i, 1], Path(p).name, fontsize=9)

        ax.set_title("PCA 2D Embeddings")
        ax.grid(True)

        self._save(fig, filename)

    def plot_pca_with_query(
        self,
        rule_embs: Dict[str, np.ndarray],
        query_embs: Dict[str, np.ndarray],
        filename: str = "pca_with_query.png",
    ) -> None:
        merged: Dict[str, np.ndarray] = {}
        # 规则：文件名
        for p, v in rule_embs.items():
            merged[Path(p).name] = v
        # 查询：前缀 [Q]
        for p, v in query_embs.items():
            merged[f"[Q]{Path(p).name}"] = v

        self.plot_pca_2d(merged, filename=filename)

    # ====================
    # t-SNE
    # ====================
    def plot_tsne(
        self,
        emb_dict: Dict[str, np.ndarray],
        filename: str = "tsne.png",
    ) -> None:
        paths = list(emb_dict.keys())
        X = np.stack(list(emb_dict.values()), axis=0)

        n = len(X)
        perp = max(1, min(n - 1, 30))

        tsne = TSNE(n_components=2, perplexity=perp, learning_rate=200)
        pts = tsne.fit_transform(X)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(pts[:, 0], pts[:, 1], s=80)
        for i, p in enumerate(paths):
            ax.text(pts[i, 0], pts[i, 1], p, fontsize=9)

        ax.set_title(f"t-SNE (perplexity={perp})")

        self._save(fig, filename)

    def plot_tsne_with_query(
        self,
        rule_embs: Dict[str, np.ndarray],
        query_embs: Dict[str, np.ndarray],
        filename: str = "tsne_with_query.png",
    ) -> None:
        merged: Dict[str, np.ndarray] = {}
        for p, v in rule_embs.items():
            merged[Path(p).name] = v
        for p, v in query_embs.items():
            merged[f"[Q]{Path(p).name}"] = v

        self.plot_tsne(merged, filename=filename)

    # ====================
    # 距离矩阵
    # ====================
    def plot_distance_matrix(
        self,
        emb_dict: Dict[str, np.ndarray],
        filename: str = "distance_matrix.png",
    ) -> None:
        paths = list(emb_dict.keys())
        X = np.stack(list(emb_dict.values()), axis=0)

        N = len(paths)
        dist = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                dist[i, j] = np.linalg.norm(X[i] - X[j])

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(dist, cmap="viridis")
        fig.colorbar(im)

        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels(paths, rotation=90)
        ax.set_yticklabels(paths)
        ax.set_title("Embedding Distance Matrix")

        self._save(fig, filename)

    def plot_distance_matrix_with_query(
        self,
        rule_embs: Dict[str, np.ndarray],
        query_embs: Dict[str, np.ndarray],
        filename: str = "distance_matrix_with_query.png",
    ) -> None:
        merged: Dict[str, np.ndarray] = {}
        for p, v in rule_embs.items():
            merged[Path(p).name] = v
        for p, v in query_embs.items():
            merged[f"[Q]{Path(p).name}"] = v

        self.plot_distance_matrix(merged, filename=filename)

    # ====================
    # 雷达图
    # ====================
    def plot_radar(
        self,
        embedding: np.ndarray,
        title: str = "Embedding Radar",
        top_n: int = 8,
    ) -> None:
        v = embedding[:top_n]
        dims = [f"d{i}" for i in range(top_n)]

        values = np.concatenate([v, [v[0]]])
        labels = dims + [dims[0]]

        num_points = len(values)
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        ax.plot(angles, values, linewidth=2)
        ax.fill(angles, values, alpha=0.25)

        ax.set_xticks(angles)
        ax.set_xticklabels(labels)
        ax.set_title(title)

        safe_title = title.replace("/", "_").replace("\\", "_").replace(".", "_")
        filename = f"radar_{safe_title}.png"
        self._save(fig, filename)

    # ====================
    # query vs rules 柱状图
    # ====================
    def plot_query_vs_rules(
        self,
        query_emb: np.ndarray,
        rule_embs: np.ndarray,
        rule_labels: List[str],
        query_name: str = "query",
        top_k: int = 10,
    ) -> None:
        dists = np.linalg.norm(rule_embs - query_emb[None, :], axis=1)

        idx = np.argsort(dists)[:top_k]
        top_dists = dists[idx]
        top_labels = [rule_labels[i] for i in idx]
        x_labels = [f"{i}:{lab}" for i, lab in zip(idx, top_labels)]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(top_dists)), top_dists)
        ax.set_xticks(range(len(top_dists)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right")

        ax.set_ylabel("L2 distance")
        ax.set_title(f"Distances from query [{query_name}] to rules")

        filename = f"query_dist_{query_name.replace('/', '_').replace('.', '_')}.png"
        self._save(fig, filename)

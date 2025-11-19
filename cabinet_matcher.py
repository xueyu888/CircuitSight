from __future__ import annotations

import json
import pickle
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np
import torch
import clip
from PIL import Image, ImageOps

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from visualizer import EmbeddingVisualizer
from torchvision import transforms
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json

# ========================
# 配置模块
# ========================



@dataclass
class Config:
    """
    全局配置（无硬编码路径），可以通过 config.json 覆盖。
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        # 设备和模型
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name: str = "ViT-B/32"

        # 图像大小（CLIP 训练是 224，建议 224 / 336 / 448）
        self.image_size: int = 224

        # 路径
        self.db_path: str = "rules.db"
        self.index_path: str = "faiss.index"

        # 预处理：颜色不敏感 -> 二值化（先关掉，调试完再考虑打开）
        self.enable_binarize: bool = False
        self.binarize_threshold: int = 128  # 0-255

        # 预处理：内容裁剪（去掉大块白边/标题栏）
        self.enable_content_crop: bool = True
        # 把灰度值 < 245 的当“有内容”，255 是纯白
        self.content_bg_threshold: int = 245

        # Debug：是否保存预处理后的图像
        self.debug_save_preprocessed: bool = True
        self.debug_preprocessed_dir: str = "output/preprocessed"

        # 从外部配置覆盖
        if config_path:
            path = Path(config_path)
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                for k, v in data.items():
                    setattr(self, k, v)





# ========================
# 嵌入模型模块
# ========================


class EmbeddingModel:
    """
    CLIP 图像编码器封装：
    - 支持单张/批量
    - 可选二值化
    - 内容裁剪 + 等比缩放 + padding
    - 把“送进模型之前”的各阶段图存盘，方便你肉眼看
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model, _ = clip.load(
            config.model_name, device=config.device
        )
        self.model.eval()

        # 自定义预处理：只做 ToTensor + Normalize
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

        # Debug 目录
        if self.config.debug_save_preprocessed:
            Path(self.config.debug_preprocessed_dir).mkdir(
                parents=True, exist_ok=True
            )

    # ---------- 低级预处理 ----------

    def _binarize(self, img: Image.Image) -> Image.Image:
        """
        对输入图像进行二值化处理，以减少颜色影响，只保留结构。
        """
        gray = img.convert("L")
        threshold = self.config.binarize_threshold
        bw = gray.point(lambda p: 255 if p > threshold else 0).convert("L")
        bw_rgb = ImageOps.autocontrast(bw).convert("RGB")
        return bw_rgb

    def _basic_prepare(self, img_path: str) -> Image.Image:
        """
        基础预处理：读图 + 可选二值化（不做缩放）
        返回一个 RGB 的 PIL.Image
        """
        img = Image.open(img_path).convert("RGB")
        if self.config.enable_binarize:
            img = self._binarize(img)
        return img

    def _crop_to_content(self, img: Image.Image) -> Image.Image:
        """
        找出“非背景区域”的 bounding box，只保留电路主体。
        背景判定：灰度值 >= content_bg_threshold 视为背景。
        """
        if not self.config.enable_content_crop:
            return img

        gray = img.convert("L")
        arr = np.array(gray)

        thr = self.config.content_bg_threshold
        # 背景是“接近白”的区域；我们要找的是非背景
        mask = arr < thr

        if not mask.any():
            # 全是白的，直接返回原图，避免崩
            return img

        ys, xs = np.where(mask)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        cropped = img.crop((x_min, y_min, x_max + 1, y_max + 1))
        return cropped

    def _resize_with_padding(self, img: Image.Image) -> Image.Image:
        """
        等比缩放 + padding 到 (image_size, image_size)：
        - 保持内容长宽比
        - 内容最大化地占满画布
        - 减少“一个大一个小”的影响
        """
        target = self.config.image_size
        w, h = img.size

        if w == 0 or h == 0:
            return img.resize((target, target), Image.BICUBIC)

        scale = min(target / w, target / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        img_resized = img.resize((new_w, new_h), Image.BICUBIC)

        # 创建纯白背景的方图，把内容居中贴上去
        canvas = Image.new("RGB", (target, target), (255, 255, 255))
        offset_x = (target - new_w) // 2
        offset_y = (target - new_h) // 2
        canvas.paste(img_resized, (offset_x, offset_y))

        return canvas

    def _debug_save(self, img: Image.Image, src_path: str, stage: str) -> None:
        """
        把预处理过程中的图像存盘，stage 用来区分阶段：
        - raw       : 只读图/二值化之后
        - cropped   : 内容裁剪之后
        - final     : resize + padding之后（真正喂给模型的）
        """
        if not self.config.debug_save_preprocessed:
            return

        src_name = Path(src_path).name
        safe_stage = stage.replace(" ", "_")
        out_name = f"{safe_stage}_{self.config.image_size}px_{src_name}"
        out_path = Path(self.config.debug_preprocessed_dir) / out_name
        img.save(out_path)
        print(f"[debug] saved {stage} image -> {out_path}")

    # ---------- 编码单张图 ----------

    def encode_image(self, img_path: str) -> np.ndarray:
        """
        单张图 -> 1 个 embedding 向量 (float32, L2 归一化)
        """
        # 1) 基础预处理
        img = self._basic_prepare(img_path)
        self._debug_save(img, img_path, stage="raw")

        # 2) 内容裁剪
        img = self._crop_to_content(img)
        self._debug_save(img, img_path, stage="cropped")

        # 3) 等比缩放 + padding
        img_final = self._resize_with_padding(img)
        self._debug_save(img_final, img_path, stage="final")

        # 4) 转 tensor + normalize
        tensor = self.preprocess(img_final).unsqueeze(0).to(self.config.device)

        with torch.no_grad():
            emb = self.model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().astype("float32")[0]

    # ---------- 批量编码 ----------

    def encode_images(
        self, img_paths: Sequence[str]
    ) -> Dict[str, np.ndarray]:
        """
        多张图批量编码，返回 {img_path: embedding} 字典。
        """
        if not img_paths:
            return {}

        tensors: List[torch.Tensor] = []
        valid_paths: List[str] = []

        for p in img_paths:
            path_obj = Path(p)
            if not path_obj.exists():
                continue

            img = self._basic_prepare(str(path_obj))
            self._debug_save(img, str(path_obj), stage="raw")

            img = self._crop_to_content(img)
            self._debug_save(img, str(path_obj), stage="cropped")

            img_final = self._resize_with_padding(img)
            self._debug_save(img_final, str(path_obj), stage="final")

            t = self.preprocess(img_final)
            tensors.append(t)
            valid_paths.append(str(path_obj))

        if not tensors:
            return {}

        batch = torch.stack(tensors, dim=0).to(self.config.device)

        with torch.no_grad():
            emb = self.model.encode_image(batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embs = emb.cpu().numpy().astype("float32")  # shape: [N, D]

        return {path: embs[i] for i, path in enumerate(valid_paths)}

# ========================
# 规则数据库模块 (SQLite)
# ========================

@dataclass
class RuleRecord:
    id: int
    label: str
    embedding: np.ndarray


class RuleDB:
    """
    SQLite 规则库：
    - 存 embedding + label
    - 提供记录级别访问用于可视化 / FastAPI
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.conn = sqlite3.connect(config.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cabinet_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding BLOB NOT NULL,
                label TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def reset(self) -> None:
        """调试用：清空规则表。"""
        self.conn.execute("DELETE FROM cabinet_rules")
        self.conn.commit()

    def add_rule(self, embedding: np.ndarray, label: str) -> None:
        emb_bin = pickle.dumps(embedding)
        self.conn.execute(
            "INSERT INTO cabinet_rules (embedding, label) VALUES (?, ?)",
            (emb_bin, label),
        )
        self.conn.commit()

    def add_rules_batch(
        self, embeddings: Sequence[np.ndarray], labels: Sequence[str]
    ) -> None:
        assert len(embeddings) == len(labels)
        cursor = self.conn.cursor()
        for emb, label in zip(embeddings, labels):
            emb_bin = pickle.dumps(emb)
            cursor.execute(
                "INSERT INTO cabinet_rules (embedding, label) VALUES (?, ?)",
                (emb_bin, label),
            )
        self.conn.commit()

    def load_all(self) -> Tuple[np.ndarray, List[str]]:
        cur = self.conn.execute("SELECT embedding, label FROM cabinet_rules")
        embs: List[np.ndarray] = []
        labels: List[str] = []
        for emb_bin, label in cur:
            embs.append(pickle.loads(emb_bin))
            labels.append(label)
        if not embs:
            return np.zeros((0, 1), dtype="float32"), []
        return np.stack(embs).astype("float32"), labels

    def fetch_all_records(self) -> List[RuleRecord]:
        cur = self.conn.execute(
            "SELECT id, embedding, label FROM cabinet_rules ORDER BY id ASC"
        )
        records: List[RuleRecord] = []
        for rid, emb_bin, label in cur:
            emb = pickle.loads(emb_bin)
            records.append(RuleRecord(id=rid, label=label, embedding=emb))
        return records

    def fetch_record(self, rule_id: int) -> Optional[RuleRecord]:
        cur = self.conn.execute(
            "SELECT id, embedding, label FROM cabinet_rules WHERE id = ?",
            (rule_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        rid, emb_bin, label = row
        emb = pickle.loads(emb_bin)
        return RuleRecord(id=rid, label=label, embedding=emb)


# ========================
# Faiss 向量检索模块
# ========================

class FaissIndex:
    """
    Faiss 向量检索封装：
    - 仅负责向量索引与搜索
    - label 列表由上层注入
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.index: Optional[faiss.IndexFlatL2] = None
        self.labels: List[str] = []

    def build(self, embeddings: np.ndarray, labels: List[str]) -> None:
        if embeddings.size == 0:
            self.index = None
            self.labels = []
            return

        d = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(embeddings)
        self.labels = labels

    def query(
        self, embedding: np.ndarray, k: int = 3
    ) -> List[Tuple[str, float]]:
        if self.index is None or not self.labels:
            return []
        emb = embedding.reshape(1, -1).astype("float32")
        D, I = self.index.search(emb, k)  # D 是 squared L2
        return [
            (self.labels[i], float(np.sqrt(D[0][j])))
            for j, i in enumerate(I[0])
        ]

    def query_batch(
        self, embeddings: np.ndarray, k: int = 3
    ) -> List[List[Tuple[str, float]]]:
        if self.index is None or not self.labels:
            return [[] for _ in range(embeddings.shape[0])]
        D, I = self.index.search(embeddings.astype("float32"), k)
        results: List[List[Tuple[str, float]]] = []
        for row_d, row_i in zip(D, I):
            results.append(
                [
                    (self.labels[i], float(np.sqrt(row_d[j])))
                    for j, i in enumerate(row_i)
                ]
            )
        return results

    def save(self) -> None:
        if self.index:
            faiss.write_index(self.index, self.config.index_path)

    def load(self, labels: List[str]) -> None:
        idx_path = Path(self.config.index_path)
        if idx_path.exists():
            self.index = faiss.read_index(str(idx_path))
            self.labels = labels


# ========================
# 高层封装：CabinetMatcher
# ========================

class CabinetMatcher:
    """
    对外暴露的高层 API：
    - add_rule(s): 加规则
    - match / match_batch: 查询匹配
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        self.model = EmbeddingModel(self.config)
        self.db = RuleDB(self.config)
        self.index = FaissIndex(self.config)

        embs, labels = self.db.load_all()
        if embs.size > 0:
            self.index.build(embs, labels)
        else:
            self.index.index = None
            self.index.labels = []

    def rebuild_index(self) -> None:
        embs, labels = self.db.load_all()
        if embs.size > 0:
            self.index.build(embs, labels)
        else:
            self.index.index = None
            self.index.labels = []
        self.index.save()

    # ---------- 规则新增 ----------

    def add_rule(self, img_path: str, label: str) -> None:
        emb = self.model.encode_image(img_path)
        self.db.add_rule(emb, label)
        self.rebuild_index()

    def add_rules_batch(
        self, img_paths: Sequence[str], labels: Sequence[str]
    ) -> None:
        assert len(img_paths) == len(labels)
        emb_dict = self.model.encode_images(img_paths)

        valid_embs: List[np.ndarray] = []
        valid_labels: List[str] = []
        for path, label in zip(img_paths, labels):
            if path in emb_dict:
                valid_embs.append(emb_dict[path])
                valid_labels.append(label)

        if not valid_embs:
            return

        self.db.add_rules_batch(valid_embs, valid_labels)
        self.rebuild_index()

    # ---------- 查询 ----------

    def match(self, img_path: str, k: int = 3) -> List[Tuple[str, float]]:
        emb = self.model.encode_image(img_path)
        return self.index.query(emb, k=k)

    def match_batch(
        self, img_paths: Sequence[str], k: int = 3
    ) -> Dict[str, List[Tuple[str, float]]]:
        emb_dict = self.model.encode_images(img_paths)
        if not emb_dict:
            return {}

        ordered_paths = list(emb_dict.keys())
        embs = np.stack([emb_dict[p] for p in ordered_paths], axis=0)
        batch_results = self.index.query_batch(embs, k=k)

        return {path: res for path, res in zip(ordered_paths, batch_results)}


# ========================
# 规则检查 / 文本 & HTML 导出
# ========================

class RuleInspector:
    """
    方便人类看的“窥探工具”：
    - 文本 dump
    - HTML 表格
    """

    def __init__(self, db: RuleDB) -> None:
        self.db = db

    @staticmethod
    def _format_vector(v: np.ndarray, max_dims: int = 8) -> str:
        v = v.astype(float)
        if v.size <= max_dims:
            parts = ", ".join(f"{x:.4f}" for x in v)
            return f"[{parts}] (dim={v.size})"
        head = ", ".join(f"{x:.4f}" for x in v[:max_dims])
        return f"[{head}, ...] (dim={v.size})"

    def dump_as_text(self, max_dims: int = 8) -> str:
        records = self.db.fetch_all_records()
        lines: List[str] = []
        for rec in records:
            vec_str = self._format_vector(rec.embedding, max_dims=max_dims)
            lines.append(
                f"id={rec.id:04d} label={rec.label} embedding={vec_str}"
            )
        return "\n".join(lines)

    def dump_as_html(self, max_dims: int = 8) -> str:
        records = self.db.fetch_all_records()
        rows: List[str] = []
        for rec in records:
            vec_str = self._format_vector(rec.embedding, max_dims=max_dims)
            rows.append(
                f"<tr>"
                f"<td>{rec.id}</td>"
                f"<td>{rec.label}</td>"
                f"<td><code>{vec_str}</code></td>"
                f"</tr>"
            )

        table = (
            "<table border='1' cellspacing='0' cellpadding='4'>"
            "<thead>"
            "<tr><th>ID</th><th>Label</th><th>Embedding (preview)</th></tr>"
            "</thead>"
            "<tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )

        html = f"""
        <html>
        <head>
            <meta charset="utf-8" />
            <title>Cabinet Rules Inspector</title>
            <style>
                body {{ font-family: sans-serif; }}
                table {{ border-collapse: collapse; }}
                th, td {{ padding: 4px 8px; }}
                code {{ font-size: 12px; }}
            </style>
        </head>
        <body>
            <h1>Cabinet Rules Inspector</h1>
            {table}
        </body>
        </html>
        """
        return html


# ========================
# FastAPI 封装
# ========================

class APIServer:
    """
    把 matcher + inspector + visualizer 暴露成 FastAPI
    """

    def __init__(
        self,
        matcher: CabinetMatcher,
        inspector: RuleInspector,
        visualizer: Optional[EmbeddingVisualizer] = None,
    ) -> None:
        self.matcher = matcher
        self.inspector = inspector
        self.visualizer = visualizer or EmbeddingVisualizer()
        self.app = FastAPI(title="Cabinet Matcher API", version="0.1.0")

        self._register_routes()

    def _register_routes(self) -> None:
        app = self.app
        db = self.matcher.db

        @app.get("/rules")
        def list_rules() -> JSONResponse:
            records = db.fetch_all_records()
            data = [
                {
                    "id": r.id,
                    "label": r.label,
                    "dim": int(r.embedding.shape[0]),
                }
                for r in records
            ]
            return JSONResponse(content=data)

        @app.get("/rules/{rule_id}")
        def get_rule(rule_id: int) -> JSONResponse:
            rec = db.fetch_record(rule_id)
            if rec is None:
                raise HTTPException(status_code=404, detail="Rule not found")
            preview = [float(x) for x in rec.embedding[:16]]
            return JSONResponse(
                content={
                    "id": rec.id,
                    "label": rec.label,
                    "dim": int(rec.embedding.shape[0]),
                    "embedding_preview": preview,
                }
            )

        @app.get("/inspect/html")
        def inspect_html() -> HTMLResponse:
            html = self.inspector.dump_as_html(max_dims=16)
            return HTMLResponse(content=html)

        @app.get("/inspect/text")
        def inspect_text() -> HTMLResponse:
            text = self.inspector.dump_as_text(max_dims=16)
            # 用 <pre> 包一层，浏览器里好看一点
            html = (
                "<html><body><pre>"
                + text.replace("&", "&amp;").replace("<", "&lt;")
                + "</pre></body></html>"
            )
            return HTMLResponse(content=html)


# ========================
# 全局 app（FastAPI）
# ========================

# 正常使用：直接用这个 app 启动 uvicorn
# poetry run uvicorn cabinet_matcher:app --reload

_config = Config()
_matcher = CabinetMatcher(_config)
_inspector = RuleInspector(_matcher.db)
_visualizer = EmbeddingVisualizer()
_api_server = APIServer(_matcher, _inspector, _visualizer)
app: FastAPI = _api_server.app


# ========================
# 调试用 main：本地跑一遍 + 可视化 + 打印
# ========================

if __name__ == "__main__":
    # 调试时：清掉旧 DB / index，避免规则重复
    for p in ("rules.db", "faiss.index"):
        path = Path(p)
        if path.exists():
            path.unlink()

    cfg = Config()
    matcher = CabinetMatcher(cfg)
    vis = EmbeddingVisualizer()
    inspector = RuleInspector(matcher.db)

    # 1. 写入一批 demo 规则
    img_paths = [
        "data/emmbding/image1.png",
        "data/emmbding/image2.png",
        "data/emmbding/image3.png",
        "data/emmbding/image4.png",
        "data/emmbding/image4-1.png",
        "data/emmbding/image4-0.png",
        "data/emmbding/image4-2.png",
    ]
    labels = ["方案_A", "方案_B", "方案_C", "方案_4", "方案_4-1", "方案_4-0", "方案_4-2"]
    matcher.add_rules_batch(img_paths, labels)

    # 2. 查询一张新图
    query_paths = ["data/emmbding/q1.png",
                   "data/emmbding/q2.png",
                   "data/emmbding/q3.png",
                   "data/emmbding/q4.png",
                   "data/emmbding/q5.png"]
    results = matcher.match_batch(query_paths, k=5)
    for path, res in results.items():
        print(f"图像: {path}")
        for label, dist in res:
            print(f"  -> {label}, 距离(L2): {dist:.8f}")
        print()

    # 3. 可视化（规则 + query 一起）
    rule_emb_dict = matcher.model.encode_images(img_paths)
    query_emb_dict = matcher.model.encode_images(query_paths)

    vis.plot_pca_2d(rule_emb_dict, filename="pca.png")
    vis.plot_tsne(rule_emb_dict, filename="tsne.png")
    vis.plot_distance_matrix(rule_emb_dict, filename="distance_matrix.png")

    vis.plot_pca_with_query(rule_emb_dict, query_emb_dict)
    vis.plot_tsne_with_query(rule_emb_dict, query_emb_dict)
    vis.plot_distance_matrix_with_query(rule_emb_dict, query_emb_dict)

    # 雷达图 + query vs rules 柱状图
    vis.plot_radar(
        list(rule_emb_dict.values())[0],
        title="demo_rule_embedding",
        top_n=8,
    )

    all_embs, all_labels = matcher.db.load_all()
    query_emb = list(query_emb_dict.values())[0]
    from pathlib import Path as _Path

    vis.plot_query_vs_rules(
        query_emb=query_emb,
        rule_embs=all_embs,
        rule_labels=all_labels,
        query_name=_Path(query_paths[0]).name,
        top_k=10,
    )

    # 4. 文本方式 dump 数据库里的规则
    print("==== DB dump (text) ====")
    print(inspector.dump_as_text(max_dims=8))

    print(
        "\n调试完成。正常用 FastAPI 时，直接运行：\n"
        "  poetry run uvicorn cabinet_matcher:app --reload\n"
        "然后在浏览器里打开：\n"
        "  http://localhost:8000/inspect/html\n"
        "  http://localhost:8000/rules\n"
    )

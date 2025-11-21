# app/core/engine.py 完整覆盖

import faiss
import numpy as np
import pickle
import sqlite3
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from app.config import settings

class RecognitionEngine:
    def __init__(self):
        self._init_db()
        self.index = None
        self.labels = []
        self.rule_embs_cache = None
        self.densities_cache = {} # 新增：缓存 label -> densities
        self.reload_index()

    def _init_db(self):
        db_path = Path(settings.DB_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        # 新增 densities 字段
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rules (
                id INTEGER PRIMARY KEY, 
                label TEXT, 
                embedding BLOB,
                densities BLOB
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_label ON rules (label)")
        
        # 简单的迁移逻辑：如果旧表没有 densities 列，加一列
        try:
            conn.execute("SELECT densities FROM rules LIMIT 1")
        except sqlite3.OperationalError:
            print("[Engine] Migrating DB: Adding densities column...")
            conn.execute("ALTER TABLE rules ADD COLUMN densities BLOB")
            
        conn.commit()
        conn.close()

    def clear_rules(self):
        conn = sqlite3.connect(settings.DB_PATH)
        conn.execute("DELETE FROM rules")
        conn.commit()
        conn.close()
        self.index = None
        self.labels = []
        self.rule_embs_cache = np.zeros((0, 0))
        self.densities_cache = {}
        print("[Engine] All rules cleared.")

    # 修改：接收 densities 参数
    def add_rules(self, embeddings: np.ndarray, labels: List[str], densities_list: List[List[float]]) -> Tuple[int, List[str]]:
        conn = sqlite3.connect(settings.DB_PATH)
        cursor = conn.cursor()
        added_count = 0
        ignored_labels = []

        for i, label in enumerate(labels):
            emb = embeddings[i]
            dens = densities_list[i] # 取出对应的密度列表
            
            cursor.execute("SELECT 1 FROM rules WHERE label = ?", (label,))
            if cursor.fetchone():
                ignored_labels.append(label)
            else:
                cursor.execute("INSERT INTO rules (label, embedding, densities) VALUES (?, ?, ?)", 
                               (label, pickle.dumps(emb), pickle.dumps(dens)))
                added_count += 1
        conn.commit()
        conn.close()
        if added_count > 0: self.reload_index()
        return added_count, ignored_labels

    def reload_index(self):
        conn = sqlite3.connect(settings.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT label, embedding, densities FROM rules")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            self.index = None
            self.labels = []
            self.rule_embs_cache = np.zeros((0, 0))
            self.densities_cache = {}
            print("[Engine] Database is empty.")
            return

        self.labels = []
        embs = []
        self.densities_cache = {}

        for row in rows:
            lbl = row[0]
            self.labels.append(lbl)
            embs.append(pickle.loads(row[1]))
            
            # 加载密度数据 (兼容旧数据可能为 None)
            if row[2]:
                self.densities_cache[lbl] = pickle.loads(row[2])
            else:
                # 如果旧数据没密度，给个默认全1 (不惩罚)
                # 假设是 3x1 = 3个格子
                self.densities_cache[lbl] = [1.0] * settings.GRID_ROWS * settings.GRID_COLS

        self.rule_embs_cache = np.stack(embs).astype('float32')
        d = self.rule_embs_cache.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(self.rule_embs_cache)
        print(f"[Engine] Index reloaded with {len(self.labels)} rules.")

    def search(self, queries: np.ndarray, k: int = 3) -> Tuple[List[List[float]], List[List[str]]]:
        if self.index is None or queries.size == 0: return [], []
        dists, indices = self.index.search(queries.astype('float32'), k)
        
        out_labels = []
        out_dists = []
        for i in range(len(queries)):
            row_labels = []
            row_dists = []
            for j, idx in enumerate(indices[i]):
                if idx != -1:
                    row_labels.append(self.labels[idx])
                    row_dists.append(float(np.sqrt(dists[i][j]))) 
            out_labels.append(row_labels)
            out_dists.append(row_dists)
        return out_dists, out_labels

    def get_embedding_by_label(self, label: str) -> Optional[np.ndarray]:
        try:
            idx = self.labels.index(label)
            return self.rule_embs_cache[idx]
        except ValueError:
            return None
            
    # 新增：获取密度
    def get_density_by_label(self, label: str) -> List[float]:
        return self.densities_cache.get(label, [])
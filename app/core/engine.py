import faiss
import numpy as np
import pickle
import sqlite3
from typing import List, Tuple, Dict, Any
from pathlib import Path
from app.config import settings
from app.core.tilematch import TileMatcher

class RecognitionEngine:
    def __init__(self):
        self._init_db()
        self.index = None
        self.labels = []
        self.rule_embs_cache = None 
        self.rule_descriptors_cache: List[np.ndarray] = [] 
        self.tile_matcher = TileMatcher()
        self.reload_index()

    def _init_db(self):
        db_path = Path(settings.DB_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("ALTER TABLE rules ADD COLUMN descriptors BLOB")
        except:
            pass
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rules (
                id INTEGER PRIMARY KEY, 
                label TEXT, 
                embedding BLOB,
                descriptors BLOB 
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_label ON rules (label)")
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
        self.rule_descriptors_cache = []
        print("[Engine] All rules cleared.")

    def add_rules(self, embeddings: np.ndarray, descriptors_list: List[np.ndarray], labels: List[str]) -> Tuple[int, List[str]]:
        conn = sqlite3.connect(settings.DB_PATH)
        cursor = conn.cursor()
        added_count = 0
        ignored_labels = []

        for emb, desc, label in zip(embeddings, descriptors_list, labels):
            cursor.execute("SELECT 1 FROM rules WHERE label = ?", (label,))
            if cursor.fetchone():
                ignored_labels.append(label)
            else:
                cursor.execute(
                    "INSERT INTO rules (label, embedding, descriptors) VALUES (?, ?, ?)", 
                    (label, pickle.dumps(emb), pickle.dumps(desc))
                )
                added_count += 1
        
        conn.commit()
        conn.close()
        if added_count > 0:
            self.reload_index()
        return added_count, ignored_labels

    def reload_index(self):
        conn = sqlite3.connect(settings.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT label, embedding, descriptors FROM rules")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            self.index = None
            self.labels = []
            self.rule_descriptors_cache = []
            return

        self.labels = []
        embs = []
        self.rule_descriptors_cache = []

        for row in rows:
            if row[2] is None: continue
            self.labels.append(row[0])
            embs.append(pickle.loads(row[1]))
            self.rule_descriptors_cache.append(pickle.loads(row[2]))
            
        if not embs:
            self.index = None
            return

        self.rule_embs_cache = np.stack(embs).astype('float32')
        d = self.rule_embs_cache.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(self.rule_embs_cache)
        print(f"[Engine] Reloaded {len(self.labels)} rules.")

    def search(self, queries: np.ndarray, query_descriptors_list: List[np.ndarray], k: int = 3) -> List[List[Dict[str, Any]]]:
        if self.index is None or queries.size == 0:
            return []
        
        # 1. 粗排：ConvNeXt 向量检索
        # 多取一些候选 (Top 20)，防止正确的图因为向量距离稍远被漏掉
        candidate_k = min(len(self.labels), max(k * 8, 32))

        dists, indices = self.index.search(queries.astype('float32'), candidate_k)
        
        results = []
        
        for i in range(len(queries)):
            query_hog = query_descriptors_list[i]
            candidates = []
            
            for j, idx in enumerate(indices[i]):
                if idx == -1: continue
                label = self.labels[idx]
                target_hog = self.rule_descriptors_cache[idx]
                
                # 2. 精排：Grid HOG 结构匹配
                match_res = self.tile_matcher.match_features(query_hog, target_hog)
                structure_score = match_res.score # 0.0 ~ 1.0
                
                vec_dist = float(np.sqrt(dists[i][j]))
                
                # 3. 混合排序逻辑
                # 我们希望以结构分为主，向量距离为辅
                # 最终距离 (越小越排前)
                # 如果 structure_score = 1.0 (完美匹配), final_dist = 0.0
                final_dist_metric = (1.0 - structure_score) * 0.8 + (vec_dist * 0.2)
                
                candidates.append({
                    "label": label,
                    "final_dist": final_dist_metric,
                    "vector_dist": vec_dist,       # 原始向量距离 (展示用)
                    "raw_struct_score": structure_score, # 原始结构分 (展示用)
                    "heatmap": match_res.heatmap_b64
                })
            
            # 按混合距离排序
            candidates.sort(key=lambda x: x["final_dist"])
            results.append(candidates[:k])
            
        return results
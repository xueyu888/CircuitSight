from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse  # <--- 确保导入
from typing import List, Tuple, Optional
from pydantic import BaseModel
from PIL import Image
import io
import os
import shutil
import numpy as np
from pathlib import Path # <--- 确保导入

from app.config import settings
from app.core.model import FeatureExtractor
from app.core.engine import RecognitionEngine
from app.core.visualizer import Visualizer
from app.schema import BatchResponse, RecognitionResult, MatchResult, VectorResponse

app = FastAPI(title="CircuitSight AI")

extractor = FeatureExtractor()
engine = RecognitionEngine()
visualizer = Visualizer()

# 确保规则图片存储目录存在
os.makedirs(settings.RULES_STORAGE_DIR, exist_ok=True)

def load_images(files: List[UploadFile]) -> Tuple[List[str], List[Image.Image]]:
    images = []
    filenames = []
    for file in files:
        try:
            content = file.file.read()
            img = Image.open(io.BytesIO(content))
            images.append(img)
            filenames.append(file.filename)
        except Exception as e:
            print(f"Error loading {file.filename}: {e}")
    return filenames, images

# ==========================================
# Web 访问报告接口 (已补回)
# ==========================================
@app.get("/report", response_class=HTMLResponse)
async def view_report():
    """
    在浏览器中直接查看生成的 report.html
    访问地址: http://<服务器IP>:8000/report
    """
    report_path = Path("report.html")
    if not report_path.exists():
        return """
        <html>
            <head><title>无报告</title></head>
            <body style="font-family: sans-serif; text-align: center; padding: 50px;">
                <h1 style="color: #666;">⚠️ 报告尚未生成</h1>
                <p>请先在服务器运行测试脚本：</p>
                <code style="background: #eee; padding: 10px; display: block; margin: 20px auto; width: fit-content;">
                    poetry run python app/scripts/test_client.py ...
                </code>
            </body>
        </html>
        """
    return report_path.read_text(encoding="utf-8")

@app.post("/api/vectors", response_model=List[VectorResponse])
async def get_vectors(files: List[UploadFile] = File(...)):
    filenames, images = load_images(files)
    if not images: return []
    # 仅计算向量，忽略密度
    features, _ = extractor.encode_batch_with_density(images, filenames)
    results = []
    for fname, feat in zip(filenames, features):
        results.append(VectorResponse(filename=fname, vector=feat.tolist()))
    return results

@app.delete("/api/rules")
async def clear_rules():
    """清空规则库及存储的图片"""
    engine.clear_rules()
    if os.path.exists(settings.RULES_STORAGE_DIR):
        shutil.rmtree(settings.RULES_STORAGE_DIR)
        os.makedirs(settings.RULES_STORAGE_DIR)
    return {"status": "success", "message": "所有规则及图片已清空"}

@app.post("/api/rules/add")
async def add_rules(files: List[UploadFile] = File(...), labels: Optional[str] = Form(None)):
    filenames, images = load_images(files)
    if not images: raise HTTPException(400, "无效图片")

    if labels:
        label_list = [l.strip() for l in labels.split(",") if l.strip()]
        if len(label_list) != len(images): raise HTTPException(400, "数量不匹配")
    else:
        label_list = [os.path.splitext(f)[0] for f in filenames]

    # 1. 计算向量 + 密度
    features, densities = extractor.encode_batch_with_density(images, filenames)
    
    # 2. 存入库
    added_count, ignored_list = engine.add_rules(features, label_list, densities)
    
    saved_img_count = 0
    for img, label in zip(images, label_list):
        if label not in ignored_list:
            save_path = os.path.join(settings.RULES_STORAGE_DIR, f"{label}.png")
            img.save(save_path, format="PNG")
            saved_img_count += 1

    return {
        "status": "success", 
        "added": added_count, 
        "saved_images": saved_img_count,
        "ignored": len(ignored_list), 
        "ignored_labels": ignored_list, 
        "assigned_labels": label_list
    }

@app.post("/api/recognize", response_model=BatchResponse)
async def recognize(
    files: List[UploadFile] = File(...),
    top_n: int = Form(3),
    threshold: float = Form(0.8)
):
    filenames, images = load_images(files)
    if not images: return BatchResponse(results=[], total_processed=0)
    
    # 1. 获取查询图的 向量 + 密度
    query_feats, query_densities = extractor.encode_batch_with_density(images, filenames)
    
    # 2. 初步检索 (基于向量)
    # 我们多取一些 (Top N * 2)，因为后面可能会有分数下降导致重排
    dists_batch, labels_batch = engine.search(query_feats, k=top_n * 2)
    
    results = []
    
    # 3. 遍历每个查询图
    for i, fname in enumerate(filenames):
        matches = []
        current_query_vec = query_feats[i]
        current_query_dens = query_densities[i] # List[float] 长度为格子的数量
        
        if i < len(dists_batch):
            for dist, label in zip(dists_batch[i], labels_batch[i]):
                
                # === A. 基础向量分 ===
                similarity = 1 - (dist ** 2) / 2
                vec_score = max(0, similarity * 100)
                
                # === B. 密度惩罚计算 ===
                # 取出参考图的密度
                ref_dens = engine.get_density_by_label(label)
                
                # 如果参考图是旧数据没密度，或者长度不对，跳过惩罚
                if not ref_dens or len(ref_dens) != len(current_query_dens):
                    final_score = vec_score
                else:
                    # 逐格计算惩罚
                    num_patches = len(ref_dens)
                    penalty_factor = 1.0
                    
                    for p_idx in range(num_patches):
                        d_q = current_query_dens[p_idx]
                        d_r = ref_dens[p_idx]
                        
                        # 密度对比
                        ratio = min(d_q, d_r) / (max(d_q, d_r) + 1e-6)
                        
                        # 只有当至少一方有实质内容(>0.01)时才校验
                        if max(d_q, d_r) > 0.01:
                            if ratio < 0.3: # 差异巨大
                                penalty_factor -= (1.0 / num_patches)
                    
                    penalty_factor = max(0.1, penalty_factor)
                    final_score = vec_score * penalty_factor

                # 这里其实逻辑稍微有点绕，为了兼容旧前端，我们保留 distance 字段，但它是修正后的等效距离
                equiv_dist = np.sqrt(2 * (1 - final_score/100)) if final_score <= 100 else 0.0
                
                matches.append(MatchResult(
                    label=label, 
                    distance=equiv_dist, 
                    score=final_score, 
                    trusted=final_score > 60.0, # 60分及格
                    viz_base64=None 
                ))
        
        # === C. 重新排序 (Re-Ranking) ===
        matches.sort(key=lambda x: x.score, reverse=True)
        matches = matches[:top_n]
        
        # === D. 生成可视化图 ===
        for m in matches:
            best_match_vec = engine.get_embedding_by_label(m.label)
            ref_img_path = os.path.join(settings.RULES_STORAGE_DIR, f"{m.label}.png")
            if best_match_vec is not None and os.path.exists(ref_img_path):
                try:
                    ref_img = Image.open(ref_img_path)
                    m.viz_base64 = visualizer.draw_side_by_side_comparison(
                        images[i], ref_img, current_query_vec, best_match_vec, m.label
                    )
                except Exception:
                    pass

        default_viz = matches[0].viz_base64 if matches else None
        
        results.append(RecognitionResult(
            filename=fname, matches=matches, visualization=default_viz 
        ))
        
    global_viz = visualizer.plot_global_pca(engine.rule_embs_cache, engine.labels, query_feats, filenames)
    return BatchResponse(results=results, global_visualization=global_viz, total_processed=len(images))

@app.get("/")
def health_check():
    return {"status": "running", "model": settings.MODEL_NAME}
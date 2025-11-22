from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from typing import List, Tuple, Optional
from pydantic import BaseModel
from PIL import Image
import io
import os
import shutil
import numpy as np
from pathlib import Path

from app.config import settings
from app.core.model import FeatureExtractor
from app.core.engine import RecognitionEngine
from app.core.visualizer import Visualizer
# 引入新的特征匹配
from app.core.overlap import calculate_feature_match_score 
from app.schema import BatchResponse, RecognitionResult, MatchResult, VectorResponse

app = FastAPI(title="CircuitSight AI")

extractor = FeatureExtractor()
engine = RecognitionEngine()
visualizer = Visualizer()

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

@app.get("/report", response_class=HTMLResponse)
async def view_report():
    report_path = Path("report.html")
    if not report_path.exists(): return "Report not found"
    return report_path.read_text(encoding="utf-8")

@app.post("/api/vectors", response_model=List[VectorResponse])
async def get_vectors(files: List[UploadFile] = File(...)):
    filenames, images = load_images(files)
    if not images: return []
    features, _ = extractor.encode_batch_with_density(images, filenames)
    results = []
    for fname, feat in zip(filenames, features):
        results.append(VectorResponse(filename=fname, vector=feat.tolist()))
    return results

@app.delete("/api/rules")
async def clear_rules():
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
    else:
        label_list = [os.path.splitext(f)[0] for f in filenames]

    features, densities = extractor.encode_batch_with_density(images, filenames)
    added_count, ignored_list = engine.add_rules(features, label_list, densities)
    
    for img, label in zip(images, label_list):
        if label not in ignored_list:
            save_path = os.path.join(settings.RULES_STORAGE_DIR, f"{label}.png")
            img.save(save_path, format="PNG")

    return {"status": "success", "added": added_count, "ignored": len(ignored_list)}

@app.post("/api/recognize", response_model=BatchResponse)
async def recognize(
    files: List[UploadFile] = File(...),
    top_n: int = Form(3),
    threshold: float = Form(0.8)
):
    filenames, images = load_images(files)
    if not images: return BatchResponse(results=[], total_processed=0)
    
    # 1. 粗排
    query_feats, _ = extractor.encode_batch_with_density(images, filenames)
    CANDIDATE_K = top_n * 3
    dists_batch, labels_batch = engine.search(query_feats, k=CANDIDATE_K)
    
    results = []
    
    for i, fname in enumerate(filenames):
        candidates = []
        query_img = images[i]
        
        if i < len(dists_batch):
            for vec_dist, label in zip(dists_batch[i], labels_batch[i]):
                
                # 基础向量分 (仅作参考)
                vec_score = max(0, (1 - vec_dist**2/2) * 100)
                
                # === 精排：特征点几何匹配 ===
                match_data = {"score": 0.0}
                ref_img_path = os.path.join(settings.RULES_STORAGE_DIR, f"{label}.png")
                
                if os.path.exists(ref_img_path):
                    try:
                        ref_img = Image.open(ref_img_path)
                        # 计算 AKAZE 特征匹配分 (0.0 - 1.0)
                        match_data = calculate_feature_match_score(
                            query_img, ref_img, settings.INPUT_RESOLUTION
                        )
                    except Exception as e:
                        print(f"Feature matching failed for {label}: {e}")
                
                feat_score = match_data["score"] * 100
                
                # === 混合打分 ===
                # 特征分权重 0.8 (它最准)，向量分权重 0.2 (防止特征太少时误判)
                final_score = feat_score * 0.8 + vec_score * 0.2
                
                # 阈值判断 (特征匹配通常很准，0.4 以上就算不错了)
                is_trusted = final_score > 40.0
                
                # 生成连线图
                viz_b64 = visualizer.draw_feature_matches(match_data)

                candidates.append(MatchResult(
                    label=label,
                    distance=vec_dist, 
                    score=final_score, 
                    trusted=is_trusted,
                    viz_base64=viz_b64
                ))
        
        # 重排 & 截断
        candidates.sort(key=lambda x: x.score, reverse=True)
        final_matches = candidates[:top_n]
        
        default_viz = final_matches[0].viz_base64 if final_matches else None
        
        results.append(RecognitionResult(
            filename=fname, matches=final_matches, visualization=default_viz 
        ))
        
    global_viz = visualizer.plot_global_pca(engine.rule_embs_cache, engine.labels, query_feats, filenames)
    return BatchResponse(results=results, global_visualization=global_viz, total_processed=len(images))

@app.get("/")
def health_check():
    return {"status": "running", "model": settings.MODEL_NAME}
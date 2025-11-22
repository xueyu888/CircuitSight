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
from app.core.overlap import calculate_feature_match_score 
from app.schema import BatchResponse, RecognitionResult, MatchResult, VectorResponse

app = FastAPI(title="CircuitSight AI")

extractor = FeatureExtractor()
engine = RecognitionEngine()
visualizer = Visualizer()

os.makedirs(settings.RULES_STORAGE_DIR, exist_ok=True)
os.makedirs(settings.DEBUG_DIR, exist_ok=True)

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
    return {"status": "success"}

@app.post("/api/rules/add")
async def add_rules(files: List[UploadFile] = File(...), labels: Optional[str] = Form(None)):
    filenames, images = load_images(files)
    if not images: raise HTTPException(400, "No images")
    if labels:
        label_list = [l.strip() for l in labels.split(",") if l.strip()]
    else:
        label_list = [os.path.splitext(f)[0] for f in filenames]

    features, densities = extractor.encode_batch_with_density(images, filenames)
    added_count, ignored_list = engine.add_rules(features, label_list, densities)
    
    for img, label in zip(images, label_list):
        if label not in ignored_list:
            img.save(os.path.join(settings.RULES_STORAGE_DIR, f"{label}.png"), format="PNG")

    return {"status": "success", "added": added_count}

@app.post("/api/recognize", response_model=BatchResponse)
async def recognize(
    files: List[UploadFile] = File(...),
    top_n: int = Form(3),
    threshold: float = Form(0.8)
):
    filenames, images = load_images(files)
    if not images: return BatchResponse(results=[], total_processed=0)
    
    query_feats, _ = extractor.encode_batch_with_density(images, filenames)
    dists_batch, labels_batch = engine.search(query_feats, k=top_n * 3)
    
    results = []
    
    for i, fname in enumerate(filenames):
        candidates = []
        query_img = images[i]
        
        print(f"Processing: {fname}")
        
        if i < len(dists_batch):
            for vec_dist, label in zip(dists_batch[i], labels_batch[i]):
                vec_score = max(0, (1 - vec_dist**2/2) * 100)
                
                match_data = {"score": 0.0, "inliers": 0}
                grid_stats = {"stats_matched": 0, "stats_total": 0}
                viz_b64 = None
                
                ref_img_path = os.path.join(settings.RULES_STORAGE_DIR, f"{label}.png")
                if os.path.exists(ref_img_path):
                    try:
                        ref_img = Image.open(ref_img_path)
                        match_data = calculate_feature_match_score(
                            query_img, ref_img, (1024, 512)
                        )
                        viz_b64 = visualizer.draw_feature_matches(match_data)
                        if match_data.get("debug_grids"):
                            grid_stats = match_data["debug_grids"]
                    except Exception as e:
                        print(f"  Error: {e}")
                
                feat_score = match_data["score"] * 100
                final_score = feat_score * 0.9 + vec_score * 0.1
                
                grid_m = grid_stats.get("stats_matched", 0)
                grid_t = grid_stats.get("stats_total", 0)
                print(f"  -> {label}: {final_score:.1f}% (Grid {grid_m}/{grid_t})")
                
                candidates.append(MatchResult(
                    label=label,
                    distance=vec_dist, 
                    score=final_score, 
                    trusted=final_score > 50.0,
                    match_count=match_data.get("inliers", 0),
                    kp_query_total=grid_m,  
                    kp_ref_total=grid_t,
                    viz_base64=viz_b64
                ))
        
        candidates.sort(key=lambda x: x.score, reverse=True)
        final_matches = candidates[:top_n]
        default_viz = final_matches[0].viz_base64 if final_matches else None
        results.append(RecognitionResult(filename=fname, matches=final_matches, visualization=default_viz))
        
    return BatchResponse(results=results, global_visualization=None, total_processed=len(images))

@app.get("/")
def health_check():
    return {"status": "running"}
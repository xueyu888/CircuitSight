from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List, Tuple, Optional
from PIL import Image
import io
import os

from app.config import settings
from app.core.model import FeatureExtractor
from app.core.engine import RecognitionEngine
from app.core.visualizer import Visualizer
from app.schema import BatchResponse, RecognitionResult, MatchResult, VectorResponse

app = FastAPI(title="CircuitSight AI")

extractor = FeatureExtractor()
engine = RecognitionEngine()
visualizer = Visualizer()

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

@app.post("/api/vectors", response_model=List[VectorResponse])
async def get_vectors(files: List[UploadFile] = File(...)):
    filenames, images = load_images(files)
    if not images: return []
    features, _ = extractor.encode_batch(images, filenames)
    results = []
    for fname, feat in zip(filenames, features):
        results.append(VectorResponse(filename=fname, vector=feat.tolist()))
    return results

@app.delete("/api/rules")
async def clear_rules():
    engine.clear_rules()
    return {"status": "success", "message": "所有规则已清空"}

@app.post("/api/rules/add")
async def add_rules(files: List[UploadFile] = File(...), labels: Optional[str] = Form(None)):
    filenames, images = load_images(files)
    if not images: raise HTTPException(400, "未接收到有效图片")

    if labels:
        label_list = [l.strip() for l in labels.split(",") if l.strip()]
    else:
        print(f"[Server] 自动使用文件名作为标签。")
        label_list = [os.path.splitext(f)[0] for f in filenames]

    features, descriptors = extractor.encode_batch(images, filenames)
    added_count, ignored_list = engine.add_rules(features, descriptors, label_list)
    
    return {
        "status": "success", "added": added_count, 
        "ignored": len(ignored_list), "ignored_labels": ignored_list,
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
    
    # 1. 提取特征
    query_feats, query_descs = extractor.encode_batch(images, filenames)
    
    # 2. 混合搜索
    search_results = engine.search(query_feats, query_descs, k=top_n)
    
    results = []
    
    for i, fname in enumerate(filenames):
        matches = []
        raw_matches_for_plot = [] 
        
        if i < len(search_results):
            for item in search_results[i]:
                # 取出三个指标
                vec_dist = item['vector_dist']
                struct_score_val = item['raw_struct_score'] * 100 # 转百分比 0-100
                
                # 计算最终混合得分 (用于判定 trusted)
                # 我们以结构分为主要信任依据
                # 如果 struct_score_val > 70% 或者 vec_dist 很小，认为 trusted
                # 这里沿用之前的 threshold 参数逻辑：针对 vector distance
                # 但更推荐直接看 structure score
                
                # 为了兼容，我们算一个“最终混合分”
                final_score = max(0, (1.0 - item['final_dist']) * 100)
                
                # 信任逻辑：必须结构相似度高
                is_trusted = struct_score_val >= 60.0 # 结构分 > 60% 认为匹配

                matches.append(MatchResult(
                    label=item['label'],
                    final_score=final_score,
                    vector_dist=vec_dist,        # 明确：这是 ConvNeXt 的结果
                    structure_score=struct_score_val, # 明确：这是 Grid HOG 的结果
                    trusted=is_trusted,
                    heatmap=item['heatmap']
                ))
                
                raw_matches_for_plot.append((item['label'], struct_score_val))
        
        # 柱状图现在显示的是结构分
        viz_b64 = visualizer.plot_top_matches(fname, raw_matches_for_plot, 60.0)
        
        results.append(RecognitionResult(
            filename=fname, matches=matches, visualization=viz_b64
        ))
        
    global_viz = visualizer.plot_global_pca(
        engine.rule_embs_cache, engine.labels, query_feats, filenames
    )
    
    return BatchResponse(results=results, global_visualization=global_viz, total_processed=len(images))

@app.get("/")
def health_check():
    return {"status": "running", "model": settings.MODEL_NAME}
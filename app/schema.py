from pydantic import BaseModel
from typing import List, Optional

class MatchResult(BaseModel):
    label: str
    
    # === 核心指标拆解 ===
    final_score: float    # 最终混合得分 (用于排序, 0-100)
    vector_dist: float    # 嵌入模型给出的 L2 距离 (粗排结果)
    structure_score: float # 匹配算法给出的相似度 (精排结果, 0-100)
    
    trusted: bool
    heatmap: Optional[str] = None 

class RecognitionResult(BaseModel):
    filename: str
    matches: List[MatchResult]
    visualization: Optional[str] = None 

class BatchResponse(BaseModel):
    results: List[RecognitionResult]
    global_visualization: Optional[str] = None
    total_processed: int

class VectorResponse(BaseModel):
    filename: str
    vector: List[float]
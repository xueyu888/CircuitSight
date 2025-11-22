from pydantic import BaseModel
from typing import List, Optional

class MatchResult(BaseModel):
    label: str
    distance: float
    score: float
    trusted: bool
    
    # === 统计字段 ===
    match_count: int = 0      # RANSAC 内点数
    kp_query_total: int = 0
    kp_ref_total: int = 0
    
    # === 新增：网格统计 ===
    grid_matched: int = 0     # 匹配的格子数 (分子)
    grid_total: int = 0       # 总活跃格子数 (分母, Max值)
    
    viz_base64: Optional[str] = None 

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
from pydantic import BaseModel
from typing import List, Optional

class MatchResult(BaseModel):
    label: str
    distance: float
    score: float
    trusted: bool
    # 新增：用于存放该匹配项的对比图
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
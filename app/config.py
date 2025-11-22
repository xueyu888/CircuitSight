from pydantic_settings import BaseSettings
from typing import Tuple
import os

class Settings(BaseSettings):
    # === 服务配置 ===
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # === 模型配置 ===
    MODEL_TYPE: str = "convnext"
    MODEL_NAME: str = "convnext_tiny.in12k_ft_in1k"
    
    # === 切分策略 ===
    GRID_ROWS: int = 3
    GRID_COLS: int = 1
    
    # === 关键修改：分辨率调整 ===
    # 改为 高1024 x 宽512 (瘦高型，适配电路图)
    # 这样缩放时，长条形的柜子图不会被压扁，细节保留更好
    INPUT_RESOLUTION: Tuple[int, int] = (1024, 512)
    
    DEVICE: str = "cuda" 
    
    # === 预处理配置 ===
    ENABLE_BINARIZE: bool = True      
    BINARIZE_THRESHOLD: int = 160     
    DEBUG_PREPROCESS: bool = True
    
    # === 存储路径 ===
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DB_PATH: str = os.path.join(BASE_DIR, "data", "rules.db")
    UPLOAD_DIR: str = os.path.join(BASE_DIR, "data", "uploads")
    DEBUG_DIR: str = os.path.join(BASE_DIR, "data", "debug")
    RULES_STORAGE_DIR: str = os.path.join(BASE_DIR, "data", "rules_storage")

    class Config:
        env_file = ".env"

settings = Settings()
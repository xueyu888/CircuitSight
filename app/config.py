

# --- 方案 B: 使用 CLIP (当前激活) ---
# MODEL_TYPE: str = "clip"
# MODEL_NAME: str = "ViT-B/32"
# # CLIP 必须是正方形，否则位置编码错乱导致效果极差
# INPUT_RESOLUTION: Tuple[int, int] = (224, 224) 
# DEVICE: str = "cuda" 


from pydantic_settings import BaseSettings
from typing import Tuple
import os

class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    MODEL_TYPE: str = "convnext"
    MODEL_NAME: str = "convnext_tiny.in12k_ft_in1k"
    
    # 使用瘦高分辨率，适配柜体图的比例
    INPUT_RESOLUTION: Tuple[int, int] = (512, 192) 
    
    ENABLE_BINARIZE: bool = False      
    BINARIZE_THRESHOLD: int = 160     
    
    DEBUG_PREPROCESS: bool = True
    
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DB_PATH: str = os.path.join(BASE_DIR, "data", "rules.db")
    UPLOAD_DIR: str = os.path.join(BASE_DIR, "data", "uploads")
    DEBUG_DIR: str = os.path.join(BASE_DIR, "data", "debug")

    # === 新增：结构对齐参数 (Grid HOG) ===
    GRID_X: int = 20     # 水平切成 20 份
    GRID_Y: int = 50     # 垂直切成 50 份 (因为是瘦高图，垂直切多一点)
    ORI_BINS: int = 9    # 梯度方向分为 9 个区间
    MAX_SHIFT: int = 2   # 允许上下左右偏移 2 个格子
    HEATMAP_CLIP: float = 0.2

    class Config:
        env_file = ".env"

settings = Settings()
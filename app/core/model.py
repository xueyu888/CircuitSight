import torch
import timm
import numpy as np
import os
from abc import ABC, abstractmethod
from typing import List, Tuple  # <--- 修复点 1: 补全 Tuple
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms
from app.config import settings

class BaseExtractor(ABC):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if settings.DEBUG_PREPROCESS:
            # 确保调试目录存在
            os.makedirs(settings.DEBUG_DIR, exist_ok=True)

    def _save_debug(self, img_obj, stage_name: str, filename: str):
        """保存调试图片"""
        if not settings.DEBUG_PREPROCESS: return
        
        safe_name = os.path.basename(filename)
        save_path = os.path.join(settings.DEBUG_DIR, f"{safe_name}_{stage_name}.png")
        
        try:
            if isinstance(img_obj, torch.Tensor):
                tensor = img_obj.clone().detach().cpu()
                # ImageNet 反归一化
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                tensor = tensor * std + mean
                tensor = torch.clamp(tensor, 0, 1)
                transforms.ToPILImage()(tensor).save(save_path)
            elif isinstance(img_obj, Image.Image):
                img_obj.save(save_path)
        except Exception as e:
            print(f"[Debug] Save failed: {e}")

    def _clean_image(self, img: Image.Image, filename: str) -> Image.Image:
        """
        第一步：清洗图片。
        二值化 -> 线条加粗 -> 反色(黑底白线)。
        """
        img = img.convert("RGB")
        
        # 1. 二值化
        if settings.ENABLE_BINARIZE:
            gray = img.convert("L")
            bw = gray.point(lambda x: 255 if x > settings.BINARIZE_THRESHOLD else 0)
            img = bw.convert("RGB")

        # 2. 线条加粗 (MinFilter 在白底上会扩张黑色像素，即加粗线条)
        w, h = img.size
        if min(w, h) > 500:
            img = img.filter(ImageFilter.MinFilter(3))

        # 3. 反色 (变成黑底白线)
        # 这一步至关重要，消除了 Padding 的边界框效应
        img = ImageOps.invert(img)
        
        self._save_debug(img, "01_cleaned_full", filename)
        return img

    def _resize_pad_patch(self, patch: Image.Image, filename: str, patch_idx: int) -> torch.Tensor:
        """
        第二步：将切出来的碎片缩放到模型输入尺寸，并转 Tensor
        """
        target_h, target_w = settings.INPUT_RESOLUTION
        w, h = patch.size
        
        if w == 0 or h == 0: 
             # 防止空切片崩溃，生成纯黑图
             patch_resized = Image.new("RGB", (target_w, target_h), (0,0,0))
        else:
            # 1. Resize (保持比例)
            scale = min(target_w / w, target_h / h)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            patch_resized = patch.resize((new_w, new_h), Image.LANCZOS)
        
        # 2. Padding (黑色填充)
        # 因为之前已经反色了，背景是黑的，这里用 (0,0,0) 填充完美融合
        new_img = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        new_img.paste(patch_resized, (paste_x, paste_y))
        
        self._save_debug(new_img, f"02_patch_{patch_idx}", filename)
        
        # 3. Tensor
        t = self.to_tensor(new_img)
        t = self.normalize(t)
        return t

    @abstractmethod
    def encode_batch(self, images: list[Image.Image], filenames: list[str] = None) -> np.ndarray:
        pass


class ConvNextExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        print(f"[Init] Loading ConvNeXt ({settings.MODEL_NAME}) with Grid {settings.GRID_ROWS}x{settings.GRID_COLS}...")
        self.model = timm.create_model(settings.MODEL_NAME, pretrained=True, num_classes=0)
        self.model.to(self.device)
        self.model.eval()
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def encode_batch(self, images: list[Image.Image], filenames: list[str] = None) -> np.ndarray:
        # 简单封装，不需要密度
        features, _ = self.encode_batch_with_density(images, filenames)
        return features

    def _calculate_patch_density(self, patch: Image.Image) -> float:
        """计算 Patch 的像素密度 (0.0 - 1.0)"""
        # 转灰度 -> 转数组
        arr = np.array(patch.convert("L"))
        # 统计非黑像素 (阈值30，过滤噪点)
        non_zero = np.count_nonzero(arr > 30)
        total = arr.size
        if total == 0: return 0.0
        return non_zero / total

    def encode_batch_with_density(self, images: list[Image.Image], filenames: list[str] = None) -> Tuple[np.ndarray, List[List[float]]]:
        """
        核心实现：同时返回 [向量] 和 [每个格子的密度信息]
        """
        if not images: return np.array([]), []
        if not filenames: filenames = [f"batch_{i}" for i in range(len(images))]

        rows, cols = settings.GRID_ROWS, settings.GRID_COLS
        final_embeddings = []
        batch_densities = [] # 存储每张图、每个格子的密度

        for i, img in enumerate(images):
            fname = filenames[i]
            cleaned_img = self._clean_image(img, fname)
            w, h = cleaned_img.size
            
            patch_tensors = []
            img_densities = [] # 当前图的密度列表
            
            step_w, step_h = w / cols, h / rows
            p_idx = 0
            for r in range(rows):
                for c in range(cols):
                    left = int(c * step_w)
                    upper = int(r * step_h)
                    right = int((c + 1) * step_w)
                    lower = int((r + 1) * step_h)
                    
                    patch = cleaned_img.crop((left, upper, right, lower))
                    
                    # 1. 计算并记录密度
                    density = self._calculate_patch_density(patch)
                    img_densities.append(density)
                    
                    # 2. 转向量准备
                    t = self._resize_pad_patch(patch, fname, p_idx)
                    patch_tensors.append(t)
                    p_idx += 1
            
            batch_densities.append(img_densities)
            
            patch_batch = torch.stack(patch_tensors).to(self.device)
            with torch.no_grad():
                features = self.model(patch_batch)
                features = torch.nn.functional.normalize(features, p=2, dim=1)
            
            combined_feature = features.flatten().cpu().numpy()
            final_embeddings.append(combined_feature)

        result = np.stack(final_embeddings).astype("float32")
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        result = result / (norms + 1e-6)
        
        return result, batch_densities

# 工厂类
class FeatureExtractor:
    def __init__(self):
        # 强制使用 ConvNeXt，适合网格切分
        self.impl = ConvNextExtractor()

    def encode_batch(self, images: list[Image.Image], filenames: list[str] = None) -> np.ndarray:
        return self.impl.encode_batch(images, filenames)

    # 修复点 2: 增加这个转发方法，供 main.py 调用
    def encode_batch_with_density(self, images: list[Image.Image], filenames: list[str] = None) -> Tuple[np.ndarray, List[List[float]]]:
        return self.impl.encode_batch_with_density(images, filenames)
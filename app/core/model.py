import torch
import timm
import clip
import numpy as np
import os
from typing import Tuple, List
from abc import ABC, abstractmethod
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms
from app.config import settings
from app.core.tilematch import TileMatcher

class BaseExtractor(ABC):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if settings.DEBUG_PREPROCESS:
            os.makedirs(settings.DEBUG_DIR, exist_ok=True)
        
        # 初始化结构匹配器
        self.tile_matcher = TileMatcher()

    def _save_debug(self, img_obj, stage_name: str, filename: str):
        if not settings.DEBUG_PREPROCESS: return
        safe_name = os.path.basename(filename)
        save_path = os.path.join(settings.DEBUG_DIR, f"{safe_name}_{stage_name}.png")
        
        if isinstance(img_obj, torch.Tensor):
            tensor = img_obj.clone().detach().cpu()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            if "clip" in settings.MODEL_TYPE:
                mean = torch.tensor([0.4814, 0.4578, 0.4082]).view(3, 1, 1)
                std = torch.tensor([0.2686, 0.2613, 0.2758]).view(3, 1, 1)
            tensor = tensor * std + mean
            tensor = torch.clamp(tensor, 0, 1)
            transforms.ToPILImage()(tensor).save(save_path)
        elif isinstance(img_obj, Image.Image):
            img_obj.save(save_path)

    def _tight_crop_black_bg(self, img: Image.Image, thr: int = 10) -> Image.Image:
        # 现在是“黑底白线”，thr 越小越容易保留弱线
        g = np.array(img.convert("L"))
        mask = g > thr
        if not mask.any():
            return img
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        return img.crop((x0, y0, x1 + 1, y1 + 1))

    # --- 替换你的 _common_preprocess ---
    def _common_preprocess(self, img: Image.Image, filename: str) -> Image.Image:
        self._save_debug(img, "01_original", filename)
        img = img.convert("RGB")

        if settings.ENABLE_BINARIZE:
            gray = img.convert("L")
            bw = gray.point(lambda x: 255 if x > settings.BINARIZE_THRESHOLD else 0)
            img = bw.convert("RGB")
            self._save_debug(img, "02_binary", filename)

        # 反色：黑底白线
        img = ImageOps.invert(img)
        self._save_debug(img, "03_inverted", filename)

        # **先裁掉黑底背景**
        img = self._tight_crop_black_bg(img, thr=10)
        self._save_debug(img, "04_cropped", filename)

        # 线条加粗（可选）
        w, h = img.size
        if min(w, h) > 500:
            img = img.filter(ImageFilter.MinFilter(3))
            self._save_debug(img, "05_thickened", filename)

        # Letterbox 到 (H, W)
        target_h, target_w = settings.INPUT_RESOLUTION
        ow, oh = img.size
        scale = min(target_w / ow, target_h / oh)
        nw, nh = max(1, int(round(ow * scale))), max(1, int(round(oh * scale)))
        img_resized = img.resize((nw, nh), Image.LANCZOS)

        canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        offx, offy = (target_w - nw) // 2, (target_h - nh) // 2
        canvas.paste(img_resized, (offx, offy))

        self._save_debug(canvas, "06_padded_black", filename)
        return canvas

        def compute_descriptors(self, img: Image.Image, filename: str) -> np.ndarray:
            """计算局部 Grid HOG 描述子"""
            pil_img = self._common_preprocess(img, filename)
            return self.tile_matcher.compute_hog_feature(pil_img)

        @abstractmethod
        def encode_batch(self, images: list[Image.Image], filenames: list[str] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
            pass


class ConvNextExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        print(f"[Init] Loading ConvNeXt ({settings.MODEL_NAME})...")
        self.model = timm.create_model(settings.MODEL_NAME, pretrained=True, num_classes=0)
        self.model.to(self.device)
        self.model.eval()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def encode_batch(self, images: list[Image.Image], filenames: list[str] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
        if not images: return np.array([]), []
        if not filenames: filenames = [f"batch_{i}" for i in range(len(images))]

        tensors = []
        descriptors = []

        for i, img in enumerate(images):
            fname = filenames[i]
            
            # 1. 预处理图用于 CNN
            pil_img = self._common_preprocess(img, fname)
            tensor = self.to_tensor(pil_img)
            tensor = self.normalize(tensor)
            tensors.append(tensor)
            self._save_debug(tensor, "06_final_tensor", fname)
            
            # 2. 预处理图用于 HOG
            # (实际上 _common_preprocess 是幂等的，这里为了逻辑清晰重新生成特征)
            # 也可以复用 pil_img，TileMatcher 内部会转灰度
            hog_feat, hog_energy = self.tile_matcher.compute_hog_feature(pil_img)
            descriptors.append((hog_feat, hog_energy))

        batch_tensor = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            features = self.model(batch_tensor)
            features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        return features.cpu().numpy().astype("float32"), descriptors


class CLIPExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        print(f"[Init] Loading CLIP...")
        self.model, _ = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        self.normalize = transforms.Normalize(mean=(0.4814, 0.4578, 0.4082), std=(0.2686, 0.2613, 0.2758))
        self.to_tensor = transforms.ToTensor()

    def encode_batch(self, images: list[Image.Image], filenames: list[str] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
        if not images: return np.array([]), []
        if not filenames: filenames = [f"clip_{i}" for i in range(len(images))]

        tensors = []
        descriptors = []
        for i, img in enumerate(images):
            fname = filenames[i]
            pil_img = self._common_preprocess(img, fname)
            tensor = self.to_tensor(pil_img)
            tensor = self.normalize(tensor)
            tensors.append(tensor)
            self._save_debug(tensor, "06_final_tensor", fname)
            
            descriptors.append(self.tile_matcher.compute_hog_feature(pil_img))

        batch_tensor = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(batch_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype("float32"), descriptors

class FeatureExtractor:
    def __init__(self):
        if settings.MODEL_TYPE.lower() == "clip":
            self.impl = CLIPExtractor()
        else:
            self.impl = ConvNextExtractor()

    def encode_batch(self, images: list[Image.Image], filenames: list[str] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
        return self.impl.encode_batch(images, filenames)
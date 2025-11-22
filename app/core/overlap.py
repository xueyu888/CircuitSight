import cv2
import numpy as np
from PIL import Image, ImageFilter

def _prep_gray_for_features(pil_img: Image.Image, target_hw=(512, 192)) -> np.ndarray:
    """
    特征提取专用的预处理：
    1. 保持比例缩放
    2. 居中填充
    3. 【关键】形态学膨胀，让线条变粗，增加特征点数量
    """
    target_h, target_w = target_hw
    
    # 1. 转灰度
    img = pil_img.convert("L")
    
    # 2. 缩放
    w, h = img.size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    
    # 3. 填充白底
    canvas = Image.new("L", (target_w, target_h), 255) 
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(img_resized, (paste_x, paste_y))
    
    # 转 numpy
    arr = np.array(canvas)
    
    # 4. 【关键优化】自适应二值化 + 膨胀
    # 将淡淡的线条变成强烈的黑白反差
    binary = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    
    # 膨胀 (Dilation)：让线条变粗，创造更多角点供 AKAZE 识别
    # 使用 3x3 的核进行 2 次膨胀
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # dilated = cv2.dilate(binary, kernel, iterations=2)
    
    return binary

def calculate_feature_match_score(pil_a: Image.Image, pil_b: Image.Image, target_hw=(512, 192)) -> dict:
    """
    优化的特征匹配算法：
    - 针对线条图进行了加粗预处理
    - 放宽了 RANSAC 阈值，容忍导线伸缩导致的非刚性形变
    """
    img1 = _prep_gray_for_features(pil_a, target_hw)
    img2 = _prep_gray_for_features(pil_b, target_hw)

    # 1. 初始化 AKAZE
    # threshold: 降低阈值以检测更多微弱特征 (默认约 0.001)
    detector = cv2.AKAZE_create(threshold=0.0003)

    # 2. 检测
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    # 兜底：如果特征点实在太少
    if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
        return {
            "score": 0.0, "matches": 0, "inliers": 0, 
            "kp1": len(kp1), "kp2": len(kp2), 
            "debug_matches": [], "debug_mask": None, 
            "imgs": (img1, img2), "kps": (kp1, kp2)
        }

    # 3. 匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    # 4. 过滤 (Ratio Test)
    # 【优化】放宽比例到 0.85 (原0.75)，允许更多重复纹理（如并排的CT）保留下来
    good_matches = []
    for m, n in matches:
        if m.distance < 0.85 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        return {
            "score": 0.0, "matches": 0, "inliers": 0, 
            "kp1": len(kp1), "kp2": len(kp2), 
            "debug_matches": [], "debug_mask": None,
            "imgs": (img1, img2), "kps": (kp1, kp2)
        }

    # 5. 几何校验 (RANSAC)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    inliers_count = 0
    mask = None
    try:
        # 【关键优化】ransacReprojThreshold 从 5.0 提高到 15.0
        # 这允许匹配点有 15 像素的“错位”，极大提高了对“导线拉长”等非刚性形变的容忍度
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 15.0)
        if mask is not None:
            inliers_count = np.sum(mask)
    except:
        inliers_count = len(good_matches)

    # 6. 算分
    # 使用 max(len) 作为分母过于严苛，改用 min(len)
    # 只要 A 是 B 的子集（截图），分数就应该高
    denominator = min(len(kp1), len(kp2))
    
    # 增加一个 log 惩罚，避免特征点极少时分数虚高
    # 如果总特征点少于 20，分数打折
    completeness = min(1.0, denominator / 50.0) 
    
    raw_score = inliers_count / (denominator + 1e-6)
    score = min(1.0, raw_score * 0.9 + completeness * 0.1) # 稍微平滑一下

    return {
        "score": float(score),
        "matches": len(good_matches),
        "inliers": int(inliers_count),
        "kp1": len(kp1),
        "kp2": len(kp2),
        "debug_matches": good_matches,
        "debug_mask": mask,
        "imgs": (img1, img2),
        "kps": (kp1, kp2)
    }
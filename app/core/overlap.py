import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

# ==========================================
# 🔧 核心参数配置 (调参看这里)
# ==========================================
CONFIG = {
    # 1. 预处理
    "TARGET_HW": (1024, 512),   # 统一分辨率 (高, 宽)
    "BINARY_THRESH": 200,       # 亮度阈值 (大于此值变黑)
    "MIN_AREA": 10,             # 去噪：小于此面积的连通域删掉
    "LINE_THICKNESS": 2,        # 最终线条统一膨胀的厚度 (iterations)
    
    # 2. 特征提取 (SIFT)
    "SIFT_NFEATURES": 3000,     # 最大特征点数
    "SIFT_EDGE_THRESH": 10,     # 边缘阈值 (越小越严，只抓拐角)
    "SIFT_CONTRAST": 0.03,      # 对比度阈值
    
    # 3. 几何校验 (RANSAC)
    "RANSAC_THRESH": 20.0,      # 允许的像素位移误差 (越小越严)
    
    # 4. 评分网格
    "GRID_SIZE": 40,            # 网格大小 (像素)
    "GRID_BLUR": 1,             # 网格容错半径 (1表示特征点会影响周围3x3个格子)
    
    # 5. 评分曲线
    "SCORE_POW": 0.7,           # 提分指数 (0.5-1.0, 越小提分越猛)
    "MIN_INLIERS": 5            # 最少匹配点数，少于此直接0分
}

def _prep_standardized_lines(pil_img: Image.Image) -> np.ndarray:
    """
    标准化预处理：
    1. 归一化 -> 二值化 -> 去噪
    2. 【关键】骨架化 (Skeletonize)：把所有线条变成 1 像素宽
    3. 【关键】定宽膨胀：统一把线条变粗
    """
    target_h, target_w = CONFIG["TARGET_HW"]
    
    # Resize & Pad
    img = pil_img.convert("L")
    w, h = img.size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    
    canvas = Image.new("L", (target_w, target_h), 255) 
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(img_resized, (paste_x, paste_y))
    
    arr = np.array(canvas)
    
    # 1. 强力二值化
    arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
    _, binary = cv2.threshold(arr, CONFIG["BINARY_THRESH"], 255, cv2.THRESH_BINARY_INV)
    
    # 2. 连通域去噪
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean_binary = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > CONFIG["MIN_AREA"]: 
            clean_binary[labels == i] = 255
            
    # 3. 骨架化 (先把线条变细到极致)
    # skeletonize 需要 bool 输入，输出也是 bool
    skeleton = skeletonize(clean_binary > 0)
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)
    
    # 4. 统一膨胀 (再把线条变粗到一致)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    final_img = cv2.dilate(skeleton_uint8, kernel, iterations=CONFIG["LINE_THICKNESS"])
    
    return final_img

def _update_grid_with_tolerance(grid, kps, radius):
    """
    带容错的网格统计。
    特征点不仅点亮当前格子，还会点亮周围 radius 范围内的格子。
    """
    h_grid, w_grid = grid.shape
    grid_px = CONFIG["GRID_SIZE"]
    
    for kp in kps:
        cx, cy = int(kp.pt[0] / grid_px), int(kp.pt[1] / grid_px)
        
        # 容错扩散
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h_grid and 0 <= nx < w_grid:
                    grid[ny, nx] = 1
    return grid

def calculate_feature_match_score(pil_a: Image.Image, pil_b: Image.Image, target_hw=None) -> dict:
    # 忽略传入的 target_hw，使用全局配置
    target_hw = CONFIG["TARGET_HW"]
    
    try:
        img1 = _prep_standardized_lines(pil_a)
        img2 = _prep_standardized_lines(pil_b)

        sift = cv2.SIFT_create(
            nfeatures=CONFIG["SIFT_NFEATURES"], 
            edgeThreshold=CONFIG["SIFT_EDGE_THRESH"], 
            contrastThreshold=CONFIG["SIFT_CONTRAST"]
        )
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # 初始化 Grid 尺寸
        h, w = target_hw
        grid_h = h // CONFIG["GRID_SIZE"]
        grid_w = w // CONFIG["GRID_SIZE"]

        if des1 is None or des2 is None or len(kp1) < 3 or len(kp2) < 3:
            return _empty_result(len(kp1) if kp1 else 0, len(kp2) if kp2 else 0, img1, img2)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:int(len(matches) * 0.85)]

        if len(good_matches) < 4:
            return _empty_result(len(kp1), len(kp2), img1, img2)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        inliers_count = 0
        mask = None
        try:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, CONFIG["RANSAC_THRESH"]) 
            if mask is not None:
                inliers_count = np.sum(mask)
        except:
            inliers_count = 0

        # === Grid 计算 (带容错) ===
        
        # 1. 活跃网格 (分母)
        grid_Q = np.zeros((grid_h + 1, grid_w + 1), dtype=np.uint8)
        grid_R = np.zeros((grid_h + 1, grid_w + 1), dtype=np.uint8)
        
        grid_Q = _update_grid_with_tolerance(grid_Q, kp1, CONFIG["GRID_BLUR"])
        grid_R = _update_grid_with_tolerance(grid_R, kp2, CONFIG["GRID_BLUR"])
        
        max_active_blocks = max(np.sum(grid_Q), np.sum(grid_R))
        
        # 2. 匹配网格 (分子)
        matched_grid_Q = np.zeros((grid_h + 1, grid_w + 1), dtype=np.uint8)
        matched_grid_R = np.zeros((grid_h + 1, grid_w + 1), dtype=np.uint8)
        
        # 提取 Inlier 点
        matched_kps_Q = []
        matched_kps_R = []
        
        if mask is not None:
            matchesMask = mask.ravel().tolist()
            for i, (m, inlier) in enumerate(zip(good_matches, matchesMask)):
                if inlier:
                    matched_kps_Q.append(kp1[m.queryIdx])
                    matched_kps_R.append(kp2[m.trainIdx])
        
        matched_grid_Q = _update_grid_with_tolerance(matched_grid_Q, matched_kps_Q, CONFIG["GRID_BLUR"])
        matched_grid_R = _update_grid_with_tolerance(matched_grid_R, matched_kps_R, CONFIG["GRID_BLUR"])
                        
        # 分子：我们关心的是 Query 图有多少被覆盖了
        # 或者取两者的 Max/Mean 也可以，这里取 Query 侧的匹配覆盖
        # 但为了和分母 Max 对齐，这里取 Max 也是合理的
        matched_blocks = max(np.sum(matched_grid_Q), np.sum(matched_grid_R))
        
        # 3. 算分
        if max_active_blocks == 0:
            score = 0.0
        else:
            raw_ratio = matched_blocks / (max_active_blocks + 1e-6)
            raw_ratio = min(1.0, raw_ratio)
            score = pow(raw_ratio, CONFIG["SCORE_POW"])

        if inliers_count < CONFIG["MIN_INLIERS"]: 
            score = 0.0

        return {
            "score": float(score),
            "matches": len(good_matches),
            "inliers": int(inliers_count),
            "kp1_count": len(kp1),
            "kp2_count": len(kp2),
            "debug_matches": good_matches,
            "debug_mask": mask,
            "imgs": (img1, img2),
            "kps": (kp1, kp2),
            "debug_grids": {
                "grid_size": CONFIG["GRID_SIZE"],
                "q_active": grid_Q,
                "r_active": grid_R,
                "q_matched": matched_grid_Q,
                "r_matched": matched_grid_R,
                "stats_matched": int(matched_blocks),
                "stats_total": int(max_active_blocks)
            }
        }
    except Exception as e:
        print(f"[Overlap Error] {e}")
        dummy = np.zeros((target_hw[0], target_hw[1]), dtype=np.uint8)
        return _empty_result(0, 0, dummy, dummy, target_hw)

def _empty_result(n1, n2, im1, im2, target_hw=(1024, 512)):
    # 为了防崩，也得构建一个空的 grid 结构
    h, w = target_hw
    gs = CONFIG["GRID_SIZE"]
    gh, gw = h//gs, w//gs
    dummy_grid = np.zeros((gh+1, gw+1), dtype=np.uint8)
    
    return {
        "score": 0.0, "matches": 0, "inliers": 0, 
        "kp1_count": n1, "kp2_count": n2, 
        "debug_matches": [], "debug_mask": None, 
        "imgs": (im1, im2), "kps": ([], []),
        "debug_grids": {
            "grid_size": gs, "q_active": dummy_grid, "r_active": dummy_grid, 
            "q_matched": dummy_grid, "r_matched": dummy_grid,
            "stats_matched": 0, "stats_total": 0
        }
    }
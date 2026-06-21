import cv2
import numpy as np
import requests
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from sklearn.cluster import KMeans
import uvicorn

app = FastAPI(title="FrameLore Image Analyzer v4.2 (Chroma-Isolated Engine)")

# =======================================================================
# ⚙️ 1. ENGINE CONFIGURATION (全域参数化，消灭硬编码)
# =======================================================================
class EngineConfig:
    GRADIENT_RADIAL_DIST = 35.0
    GRADIENT_VAR_LIMIT = 60.0
    GRADIENT_LINEAR_LIMIT = 100.0
    
    DARK_MODE_LUMINANCE_MAX = 85.0   
    COLOR_ACCENT_LUM_THRESHOLD = 100 
    
    MIN_COMPONENT_SIZE = 12           # 全局盲扫最小边长 (px)
    MIN_COMPONENT_AREA_RATIO = 0.003  # 少于视口 0.3% 且无子节点 → 噪声
    CANNY_SIGMA = 0.33

    # KMeans 惯量/像素 → 色彩置信度
    KMEANS_INERTIA_MEDIUM = 500.0    # 惯量/像素 > 此值 → medium
    KMEANS_INERTIA_LOW = 1500.0      # 惯量/像素 > 此值 → low

    # peak_accent 聚类内标准差 → 置信度
    PEAK_STD_HIGH = 15.0             # 峰值聚类 std ≤ 此值 → high
    PEAK_STD_MEDIUM = 30.0           # 峰值聚类 std ≤ 此值 → medium
    PEAK_MIN_PIXELS = 10             # 最少有彩像素数               

    # 自适应色彩孤立空间阈值
    CHROMA_SAT_MIN = 40              # 过滤无彩灰/黑/白的最小饱和度门槛 (0-255)
    CHROMA_VAL_MIN = 30              # 过滤死黑背景的最小明度门槛 (0-255)
    CHROMA_AREA_MIN_RATIO = 0.005     # 触发孤立聚类的最小有彩像素面积占比 (0.5%)

def load_image(source: str) -> np.ndarray:
    try:
        if source.startswith("http://") or source.startswith("https://"):
            resp = requests.get(source, timeout=10)
            image_bytes = np.frombuffer(resp.content, np.uint8)
            return cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        elif source.startswith("data:image") or len(source) > 200:
            if "," in source: source = source.split(",")[1]
            image_bytes = base64.b64decode(source)
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(source)
            if img is None: raise FileNotFoundError()
            return img
    except Exception:
        raise HTTPException(status_code=400, detail="图像加载失败。")

# --- 圆角精准度量（轮廓曲率分析） ---
def parse_geometry_with_confidence(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0, [0, 0, roi.shape[1], roi.shape[0]], "low", 1.0
    
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)
    main_area = cv2.contourArea(main_contour)
    roi_area = roi.shape[0] * roi.shape[1]
    
    # 面积合理性校验：暗黑模式下 OTSU 可能将 #111 卡片与 #000 背景混淆
    if main_area < roi_area * 0.5 and float(np.mean(gray)) < 85.0:
        # 自适应阈值重试
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, 31, 4)
        adapt_contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if adapt_contours:
            adapt_main = max(adapt_contours, key=cv2.contourArea)
            adapt_area = cv2.contourArea(adapt_main)
            if adapt_area > main_area * 1.5:  # 自适应找到了更大的轮廓 → 更可靠
                main_contour = adapt_main
                x, y, w, h = cv2.boundingRect(main_contour)
    
    # --- 轮廓曲率分析：判断是否存在圆角 ---
    peri = cv2.arcLength(main_contour, True)
    approx = cv2.approxPolyDP(main_contour, 0.015 * peri, True)
    
    if len(approx) <= 4:
        # 多边形顶点 ≤ 4：尖锐直角，无圆角
        return 0, [x, y, w, h], "high", 0.0
    
    # 顶点数 > 4：存在曲线段 → 对四角区域做最小二乘圆拟合
    corner_radii, errors = _fit_corner_circles(main_contour, x, y, w, h, peri)
    
    if not corner_radii:
        # 四角均无可拟合圆 → 形状非圆角矩形，报 low
        return 0, [x, y, w, h], "low", 1.0
    
    avg_radius = int(np.mean(corner_radii))
    avg_rmse = float(np.mean(errors))
    
    # RMSE → 置信度 (per SKILL.md spec)
    if avg_rmse <= 0.5:
        conf = "high"
    elif avg_rmse <= 1.5:
        conf = "medium"
    else:
        conf = "low"
    
    return avg_radius, [x, y, w, h], conf, round(avg_rmse, 4)


def _fit_corner_circles(contour, x, y, w, h, peri):
    """对轮廓四角区域分别拟合圆，返回半径列表和 RMSE 列表。"""
    corner_radii = []
    errors = []
    # 自适应 zone：弧长偏离度越大 → 圆角占比越大 → zone 越宽
    rect_peri = 2 * (w + h)
    deviation = max(0.0, peri - rect_peri) / max(1.0, rect_peri)
    zone_ratio = max(0.10, min(0.40, deviation * 1.2))
    corner_zone = int(min(w, h) * zone_ratio)
    corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    
    for cx, cy in corners:
        pts = []
        for pt in contour:
            px, py = pt[0]
            if abs(px - cx) <= corner_zone and abs(py - cy) <= corner_zone:
                pts.append([px, py])
        
        if len(pts) < 5:
            continue
        
        pts_arr = np.array(pts, dtype=np.float64)
        # 最小二乘圆拟合: min Σ[(x-a)²+(y-b)² - r²]²
        A = np.column_stack([2 * pts_arr[:, 0], 2 * pts_arr[:, 1], np.ones(len(pts_arr))])
        B = pts_arr[:, 0] ** 2 + pts_arr[:, 1] ** 2
        
        try:
            sol, residuals, _rank, _sv = np.linalg.lstsq(A, B, rcond=None)
            a, b_val, c = sol
            r = np.sqrt(max(0.0, c + a ** 2 + b_val ** 2))
            if 1.0 <= r <= min(w, h):
                corner_radii.append(r)
                rmse = float(np.sqrt(residuals[0] / len(pts_arr))) if len(residuals) > 0 else 0.0
                errors.append(rmse)
        except np.linalg.LinAlgError:
            continue
    
    return corner_radii, errors

# --- 阴影/发光高斯差分金字塔检测 (DoG) ---
def _detect_shadow_dog(roi_gray):
    """高斯差分金字塔：多尺度 σ 探测边缘弥散半径。
    对渐变背景、大弥散半径 (>20px) 和新拟态均鲁棒，不依赖固定 margin。"""
    h, w = roi_gray.shape
    if min(h, w) < 20:
        return None, "low"
    
    sigmas = [2.0, 4.0, 8.0, 16.0, 32.0]
    
    # 外环 + 内侧参考
    outer_ring_mask = np.zeros_like(roi_gray, dtype=np.uint8)
    cv2.rectangle(outer_ring_mask, (4, 4), (w-5, h-5), 255, -1)
    ring_pixels = (outer_ring_mask == 0)
    if not ring_pixels.any():
        return None, "low"
    
    inner_ring = roi_gray[8:-8, 8:-8] if h > 16 and w > 16 else None
    inner_mean = float(np.mean(inner_ring)) if inner_ring is not None and inner_ring.size > 0 else float(np.median(roi_gray))
    outer_mean = float(np.mean(roi_gray[ring_pixels]))
    base_diff = abs(inner_mean - outer_mean)
    
    if base_diff < 5:
        return None, "high"  # 内外几乎无差异，确实无阴影
    
    prev_blur = roi_gray.astype(np.float64)
    edge_responses = []
    
    for sigma in sigmas:
        ksize = max(3, int(sigma * 6) | 1)
        blurred = cv2.GaussianBlur(roi_gray, (ksize, ksize), sigma).astype(np.float64)
        diff = np.abs(blurred - prev_blur)
        ring_response = float(np.mean(diff[ring_pixels]))
        edge_responses.append((sigma, ring_response))
        prev_blur = blurred
    
    best_sigma, best_response = max(edge_responses, key=lambda x: x[1])
    
    if best_response < 1.5:
        return None, "high"
    
    blur_px = round(best_sigma * 0.7, 1)
    
    if best_response > 5.0 and base_diff > 15:
        conf = "high"
    elif best_response > 2.5:
        conf = "medium"
    else:
        conf = "low"
    
    return blur_px, conf

# --- 渐变拟合（核心重构：自适应前置色彩孤立） ---
def _estimate_gradient_angle(roi_gray):
    """Sobel 梯度方向加权直方图 → 真实渐变角度（对齐 15° 步长）"""
    gx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
    
    mag = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx) * 180 / np.pi % 180  # [0, 180)
    
    mag_flat = mag.flatten()
    angle_flat = angle.flatten()
    threshold = np.percentile(mag_flat, 70)  # 取梯度幅值 top 30%
    mask = mag_flat >= threshold
    
    if not mask.any():
        return "180deg"  # 无显著梯度 → 默认垂直
    
    angles = angle_flat[mask]
    hist, bins = np.histogram(angles, bins=36, range=(0, 180))
    dominant = bins[np.argmax(hist)] + 2.5  # bin center
    snapped = round(dominant / 15) * 15
    if snapped >= 180: snapped -= 180
    return f"{snapped}deg"

def analyze_gradient_with_confidence(roi, accent_ratio_threshold: float = 0.3):
    h, w, _ = roi.shape
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pixels = roi_rgb.reshape(-1, 3)
    total_roi_pixels = pixels.shape[0]
    
    # 修复一：将色彩孤立前置！先找到画面里真正带颜色的像素
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv_pixels = hsv_roi.reshape(-1, 3)
    chromatic_mask = (hsv_pixels[:, 1] > EngineConfig.CHROMA_SAT_MIN) & (hsv_pixels[:, 2] > EngineConfig.CHROMA_VAL_MIN)
    chromatic_pixels = pixels[chromatic_mask]
    
    # 确定靶向计算基准
    has_isolated_chroma = chromatic_pixels.shape[0] > max(50, int(total_roi_pixels * EngineConfig.CHROMA_AREA_MIN_RATIO))
    target_pixels = chromatic_pixels if has_isolated_chroma else pixels
    
    # 计算空间方差用于判断是否为 Solid
    row_means = np.mean(roi_rgb, axis=1)
    col_means = np.mean(roi_rgb, axis=0)
    row_var = np.var(row_means, axis=0).sum()
    col_var = np.var(col_means, axis=0).sum()
    
    center_roi = roi_rgb[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
    center_mean = np.mean(center_roi, axis=(0,1)) if center_roi.size > 0 else np.mean(roi_rgb, axis=(0,1))
    edge_mean = (np.mean(roi_rgb[:max(1, int(h*0.1)), :], axis=(0,1)) + np.mean(roi_rgb[max(1, int(h*0.9)):, :], axis=(0,1))) / 2
    radial_dist = np.linalg.norm(center_mean - edge_mean)
    
    grad_type = "solid"
    direction = ""
    color_confidence = "high"
    
    if radial_dist > EngineConfig.GRADIENT_RADIAL_DIST and row_var < EngineConfig.GRADIENT_VAR_LIMIT and col_var < EngineConfig.GRADIENT_VAR_LIMIT:
        grad_type = "radial"
    elif row_var > EngineConfig.GRADIENT_LINEAR_LIMIT or col_var > EngineConfig.GRADIENT_LINEAR_LIMIT:
        grad_type = "linear"
        direction = _estimate_gradient_angle(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
        
    if grad_type == "solid":
        # 修复一验证：如果是纯色，只对提纯后的色彩像素求平均，拒绝死黑背景稀释
        mean_c = np.mean(target_pixels, axis=0)
        hex_c = '#{:02x}{:02x}{:02x}'.format(int(mean_c[0]), int(mean_c[1]), int(mean_c[2]))
        return {"type": "solid", "gradient_css": hex_c, "stops": [{"hex": hex_c, "pos": 0}], "confidence": "high"}

    # 如果是渐变，则进入聚类
    n_clusters = min(3 if has_isolated_chroma else 4, len(np.unique(target_pixels, axis=0)))
    
    luminance_array = 0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]
    is_dark_mode = np.median(luminance_array) < EngineConfig.DARK_MODE_LUMINANCE_MAX
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
    kmeans.fit(target_pixels)
    
    if kmeans.inertia_ / len(target_pixels) > EngineConfig.KMEANS_INERTIA_LOW: color_confidence = "low"
    elif kmeans.inertia_ / len(target_pixels) > EngineConfig.KMEANS_INERTIA_MEDIUM: color_confidence = "medium"
        
    centers = kmeans.cluster_centers_
    # 空间坐标：非孤立模式用 2D labels，孤立模式需重建
    if has_isolated_chroma:
        coords_2d_chromatic = np.column_stack([np.argwhere(chromatic_mask)[:, 0] // w, np.argwhere(chromatic_mask)[:, 0] % w])  # 1D 展开索引 → 2D (row, col)
        labels_2d = None  # 使用 1D labels + coords_2d_chromatic
    else:
        labels_2d = kmeans.labels_.reshape(h, w)
        coords_2d_chromatic = None
    
    # 渐变方向向量（用于线性定位投影）
    ang_str = grad_type == "linear" and direction.replace("deg", "") or "180"
    try:
        ang = float(ang_str) * np.pi / 180
    except ValueError:
        ang = np.pi  # 180deg 默认
    d_row, d_col = -np.cos(ang), np.sin(ang)  # y 向下
    
    stops = []
    
    for i in range(n_clusters):
        # 获取该聚类的空间坐标
        if coords_2d_chromatic is not None:
            mask_i = kmeans.labels_ == i
            coords = coords_2d_chromatic[mask_i]
        else:
            coords = np.argwhere(labels_2d == i)
        
        if len(coords) == 0: continue
        
        if grad_type == "linear":
            # 空间质心投影到渐变方向
            centroid = np.mean(coords, axis=0)
            proj = centroid[0] * d_row + centroid[1] * d_col
            # 同类所有像素的投影值范围做归一化
            proj_all = coords[:, 0] * d_row + coords[:, 1] * d_col
            p_min, p_max = proj_all.min(), proj_all.max()
            if p_max - p_min > 0:
                pos = (proj - p_min) / (p_max - p_min) * 100
            else:
                pos = 50  # 单色团回退到中间
        else:  # radial
            cy, cx = h / 2, w / 2
            dists = np.sqrt((coords[:, 0] - cy)**2 + (coords[:, 1] - cx)**2)
            pos = np.mean(dists) / (np.sqrt(cy**2 + cx**2) or 1) * 100
        
        cluster_size = len(coords)
        
        color = centers[i]
        if is_dark_mode:
            lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            ratio = cluster_size / total_roi_pixels
            if lum > EngineConfig.COLOR_ACCENT_LUM_THRESHOLD and ratio < accent_ratio_threshold:
                color_np = np.array([[color]], dtype=np.uint8)
                hsv = cv2.cvtColor(color_np, cv2.COLOR_RGB2HSV)
                
                # 修复二：按 SKILL.md 饱和度约束 — 仅 S>70% 时 0.7x 衰减，最终 S 钳位 ≤60%
                sat = hsv[0, 0, 1]
                if sat > 70:  # 仅高饱和度触发衰减
                    sat = int(sat * 0.7)
                hsv[0, 0, 1] = min(sat, 153)  # 153 = 60% of 255, 最终 S 严禁超过 60%
                
                color_adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                color = color_adjusted[0, 0]
                
        hex_c = '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2]))
        stops.append({"hex": hex_c, "pos": int(pos)})
        
    stops = sorted(stops, key=lambda x: x["pos"])
    stop_strings = [f"{s['hex']} {s['pos']}%" for s in stops]
    css = f"linear-gradient({direction}, {', '.join(stop_strings)})" if grad_type == "linear" else f"radial-gradient(circle, {', '.join(stop_strings)})"
    return {"type": grad_type, "gradient_css": css, "stops": stops, "confidence": color_confidence}

def peak_accent_extraction(roi):
    """提取 ROI 中饱和度 Top 5% 的纯净强调色，忽略背景噪声"""
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv_flat = hsv.reshape(-1, 3)
    rgb_flat = roi_rgb.reshape(-1, 3)
    
    # 过滤低饱和度 + 低明度噪声
    mask = (hsv_flat[:, 1] > 40) & (hsv_flat[:, 2] > 30)
    candidate_rgb = rgb_flat[mask]
    candidate_sat = hsv_flat[mask, 1]
    
    if len(candidate_rgb) < EngineConfig.PEAK_MIN_PIXELS:
        # 没有足够有彩像素，回退到均值
        mean = np.mean(rgb_flat, axis=0)
        return '#{:02x}{:02x}{:02x}'.format(int(mean[0]), int(mean[1]), int(mean[2])), "low"
    
    # 取饱和度 top 5%
    threshold = np.percentile(candidate_sat, 95)
    peak_mask = candidate_sat >= threshold
    peak_pixels = candidate_rgb[peak_mask]
    
    if len(peak_pixels) == 0:
        return '#{:02x}{:02x}{:02x}'.format(int(candidate_rgb[0][0]), int(candidate_rgb[0][1]), int(candidate_rgb[0][2])), "low"
    
    # 对峰值像素做 1-2 聚类，取占比最大的中心
    n = min(2, len(np.unique(peak_pixels, axis=0)))
    kmeans = KMeans(n_clusters=n, n_init=3, random_state=42)
    kmeans.fit(peak_pixels)
    counts = np.bincount(kmeans.labels_)
    dominant = kmeans.cluster_centers_[np.argmax(counts)]
    
    # 置信度：基于聚类内标准差（色值分散程度）
    dominant_mask = (kmeans.labels_ == np.argmax(counts))
    cluster_std = float(np.std(peak_pixels[dominant_mask], axis=0).mean())
    if cluster_std <= EngineConfig.PEAK_STD_HIGH:
        confidence = "high"
    elif cluster_std <= EngineConfig.PEAK_STD_MEDIUM:
        confidence = "medium"
    else:
        confidence = "low"
    
    hex_c = '#{:02x}{:02x}{:02x}'.format(int(dominant[0]), int(dominant[1]), int(dominant[2]))
    return hex_c, confidence

class AnalyzeRegionRequest(BaseModel):
    image_source: str
    bbox: Optional[list[int]] = None  # palette 模式不需要
    mode: str = "kmeans"  # "kmeans" | "peak_accent" | "palette"
    accent_ratio_threshold: float = 0.3

class MeasureSpacingRequest(BaseModel):
    bboxes: list[list[int]]
    direction: str = "auto"  # "auto" = 双向, "x" = 仅水平, "y" = 仅垂直

class ScanGlobalRequest(BaseModel):
    image_path: str

@app.post("/analyze_region")
async def analyze_region(request: AnalyzeRegionRequest):
    img = load_image(request.image_source)
    h_img, w_img, _ = img.shape
    
    # 输入校验：非 palette 模式需校验 bbox
    if request.mode != "palette":
        if not request.bbox or len(request.bbox) != 4 or not all(0 <= v <= 1000 for v in request.bbox):
            raise HTTPException(status_code=400, detail="bbox must be [ymin, xmin, ymax, xmax] in [0, 1000]")
        ymin_norm, xmin_norm, ymax_norm, xmax_norm = request.bbox
    ymin = int(ymin_norm / 1000.0 * h_img)
    xmin = int(xmin_norm / 1000.0 * w_img)
    ymax = int(ymax_norm / 1000.0 * h_img)
    xmax = int(xmax_norm / 1000.0 * w_img)
    
    # palette 模式：全图分析，不需要 ROI
    if request.mode == "palette":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_f = hsv.reshape(-1, 3)
        chroma_mask = (hsv_f[:, 1] > 30) & (hsv_f[:, 2] > 25)
        if not chroma_mask.any():
            return {"status": "success", "mode": "palette", "palette": []}
        chroma_pixels = hsv_f[chroma_mask]
        n = min(6, len(np.unique(chroma_pixels, axis=0)))
        kmeans = KMeans(n_clusters=n, n_init=5, random_state=42)
        kmeans.fit(chroma_pixels)
        counts = np.bincount(kmeans.labels_)
        total = counts.sum()
        palette = []
        for i in range(n):
            h, s, v = kmeans.cluster_centers_[i]
            ratio = counts[i] / total
            if ratio >= 0.02:
                color_bgr = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]
                hex_c = '#{:02x}{:02x}{:02x}'.format(int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
                palette.append({"hex": hex_c, "ratio": round(ratio, 3)})
        return {"status": "success", "mode": "palette", "palette": palette}
    
    roi = img[max(0, ymin):min(ymax, h_img), max(0, xmin):min(xmax, w_img)]
    if roi.size == 0: raise HTTPException(status_code=400, detail="区域映射失败，ROI 为空。")
    
    # mode 路由：peak_accent 仅提取强调色，跳过全量分析
    if request.mode == "peak_accent":
        # 自动裁剪 ROI 中心 60% 区域，排除 LLM 框选边缘的相邻元素污染
        h_roi, w_roi = roi.shape[:2]
        cy, cx = h_roi // 2, w_roi // 2
        crop_h, crop_w = int(h_roi * 0.6), int(w_roi * 0.6)
        y_start = max(0, cy - crop_h // 2)
        y_end = min(h_roi, cy + crop_h // 2)
        x_start = max(0, cx - crop_w // 2)
        x_end = min(w_roi, cx + crop_w // 2)
        cropped = roi[y_start:y_end, x_start:x_end]
        if cropped.size == 0:
            cropped = roi
        peak_hex, peak_conf = peak_accent_extraction(cropped)
        return {
            "status": "success",
            "mode": "peak_accent",
            "style": {
                "peak_accent_hex": peak_hex,
                "color_type": "peak_accent"
            },
            "metrics_summary": {
                "color_gradient_confidence": peak_conf
            }
        }
    
    radius, geo_box, radius_conf, radius_err = parse_geometry_with_confidence(roi)
    color_info = analyze_gradient_with_confidence(roi, request.accent_ratio_threshold)
    shadow_blur, shadow_conf = _detect_shadow_dog(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
    
    return {
        "status": "success",
        "geometry": {
            "absolute_x": xmin + geo_box[0],
            "absolute_y": ymin + geo_box[1],
            "width": geo_box[2],
            "height": geo_box[3]
        },
        "style": {
            "border_radius_px": radius,
            "color_type": color_info["type"],
            "gradient_css": color_info["gradient_css"],
            "detected_shadow_blur_px": shadow_blur
        },
        "metrics_summary": {
            "border_radius_confidence": radius_conf,
            "border_radius_error_rate": radius_err,
            "color_gradient_confidence": color_info["confidence"],
            "shadow_confidence": shadow_conf
        }
    }

@app.post("/measure_spacing")
async def measure_spacing(request: MeasureSpacingRequest):
    boxes = request.bboxes
    direction = request.direction
    
    if len(boxes) < 2:
        return {"status": "success", "suggested_gap_x": 0, "suggested_gap_y": 0,
                "metrics_summary": {"gap_confidence": "high", "std_deviation_x": 0.0, "std_deviation_y": 0.0}}
    
    gaps_x = []
    gaps_y = []
    
    if direction in ("auto", "x"):
        bx = sorted(boxes, key=lambda b: b[1])
        for i in range(len(bx) - 1):
            gap = bx[i+1][1] - bx[i][3]
            if gap >= 0: gaps_x.append(gap)
    
    if direction in ("auto", "y"):
        by = sorted(boxes, key=lambda b: b[0])
        for i in range(len(by) - 1):
            gap = by[i+1][0] - by[i][2]
            if gap >= 0: gaps_y.append(gap)
        
    std_x = float(np.std(gaps_x)) if gaps_x else 0.0
    std_y = float(np.std(gaps_y)) if gaps_y else 0.0
    max_std = max(std_x, std_y)
    
    if max_std == 0.0: gap_confidence = "high"
    elif max_std <= 2.5: gap_confidence = "medium"
    else: gap_confidence = "low"

    return {
        "status": "success",
        "suggested_gap_x": int(np.median(gaps_x)) if gaps_x else 0,
        "suggested_gap_y": int(np.median(gaps_y)) if gaps_y else 0,
        "metrics_summary": {
            "gap_confidence": gap_confidence,
            "std_deviation_x": round(std_x, 2),
            "std_deviation_y": round(std_y, 2)
        }
    }

@app.post("/scan_global")
async def scan_global(request: ScanGlobalRequest):
    img = load_image(request.image_path)
    h_img, w_img, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    v = max(50.0, np.median(gray))
    lower = int(max(0, (1.0 - EngineConfig.CANNY_SIGMA) * v))
    upper = int(min(255, (1.0 + EngineConfig.CANNY_SIGMA) * v))
    edged = cv2.Canny(gray, lower, upper)
    
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detected_components = []
    if hierarchy is None: return {"status": "success", "components": [], "total_detected": 0, "filtered_count": 0}
    hierarchy = hierarchy[0]
    
    # 第一遍：收集所有候选（含噪声），记录原始索引 → 保留/丢弃决策
    candidates = []  # (orig_idx, parent_orig_idx, component_dict, keep_bool)
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if w > EngineConfig.MIN_COMPONENT_SIZE and h > EngineConfig.MIN_COMPONENT_SIZE:
            parent_orig_idx = int(hierarchy[i][3])
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.015 * peri, True)
            
            ymin_norm = int(y / h_img * 1000)
            xmin_norm = int(x / w_img * 1000)
            ymax_norm = int((y + h) / h_img * 1000)
            xmax_norm = int((x + w) / w_img * 1000)
            
            # 面积噪声判定
            area_norm = (ymax_norm - ymin_norm) * (xmax_norm - xmin_norm)
            area_ratio = area_norm / 1_000_000
            is_noise = (area_ratio < EngineConfig.MIN_COMPONENT_AREA_RATIO and parent_orig_idx == -1)
            
            # 启发式类型标注
            aspect = w / max(1, h) if w > 0 and h > 0 else 1.0
            if area_ratio >= 0.05:
                hint = "container"
            elif area_ratio >= 0.005:
                hint = "card" if 0.5 < aspect < 2.0 else "banner"
            elif 0.3 < aspect < 3.0:
                hint = "text" if area_ratio >= 0.001 else "icon"
            else:
                hint = "border"
            
            candidates.append({
                "orig_idx": i,
                "parent_orig": parent_orig_idx if parent_orig_idx != -1 else None,  # None = root
                "comp": {
                    "bbox": [ymin_norm, xmin_norm, ymax_norm, xmax_norm],
                    "maybe_rounded": len(approx) > 4,
                    "hint_type": hint
                },
                "keep": not is_noise
            })
    
    # 建立索引映射：原索引 → 新索引（仅保留项）
    keep_indices = {c["orig_idx"]: new_idx for new_idx, c in enumerate([x for x in candidates if x["keep"]])}
    filtered_count = sum(1 for c in candidates if not c["keep"])
    
    # 第二遍：重映射 parent_id
    for c in candidates:
        if not c["keep"]:
            continue
        if c["parent_orig"] is None:
            parent_id = "root"
        elif c["parent_orig"] in keep_indices:
            parent_id = f"comp_{keep_indices[c['parent_orig']]}"
        else:
            # 父节点被过滤 → 提升到 root
            parent_id = "root"
        
        detected_components.append({
            "id": f"comp_{keep_indices[c['orig_idx']]}",
            **c["comp"],
            "parent_id": parent_id
        })
            
    return {"status": "success", "total_detected": len(detected_components),
            "filtered_count": filtered_count, "components": detected_components}

# --- 字体大小估算（MSER 字符检测 + 空间聚类行分组，零 OCR 依赖） ---
class DetectTextRequest(BaseModel):
    image_source: str
    bbox: Optional[list[int]] = None


def _detect_text_mser(roi):
    """MSER 字符候选区提取 + 空间行分组 → 字号估算。
    对 Retina 缩放、中英文、暗黑模式均鲁棒，不再依赖形态学固定 kernel。"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h_roi, w_roi = gray.shape
    
    # 按分辨率自适应 MSER δ 参数（高 DPI → 大 δ）
    diag = np.sqrt(h_roi**2 + w_roi**2)
    mser_delta = max(2, min(8, int(diag / 200)))  # δ ∈ [2, 8]
    
    mser = cv2.MSER_create(delta=mser_delta, min_area=20, max_area=int(diag * 0.8))
    regions, _bboxes = mser.detectRegions(gray)
    
    if len(_bboxes) == 0:
        return {"heading_range": None, "body_range": None, "caption_range": None,
                "per_line_heights": [], "line_count": 0, "confidence": "estimated"}
    
    # 过滤：保留类文字几何特征的 bbox
    chars = []
    for b in _bboxes:
        x, y, w_c, h_c = b
        aspect = w_c / max(1, h_c)
        area = w_c * h_c
        # 中文方块 ≈ 1:1，英文字母 ≈ 0.3:1~1.2:1，排除极端扁平/细长 → 非文字
        if 0.15 < aspect < 3.0 and 20 < area < w_roi * h_roi * 0.3:
            chars.append((x, y, w_c, h_c, y + h_c // 2))  # 中心 y 用于行分组
    
    if len(chars) < 3:
        return {"heading_range": None, "body_range": None, "caption_range": None,
                "per_line_heights": [], "line_count": 0, "confidence": "estimated"}
    
    # 按中心 y 排序后分组为行（y 中心差 < 1.5 × 中位数高度）
    chars.sort(key=lambda c: c[4])  # 按 center_y 排序
    median_h = float(np.median([c[3] for c in chars]))
    
    lines = []
    current_line = [chars[0]]
    for ch in chars[1:]:
        if ch[4] - current_line[-1][4] <= median_h * 1.5:
            current_line.append(ch)
        else:
            lines.append(current_line)
            current_line = [ch]
    lines.append(current_line)
    
    # 每行统计
    per_line = []
    for line in lines:
        heights = [c[3] for c in line]
        per_line.append({
            "char_count": len(line),
            "median_height_px": int(np.median(heights)),
            "min_height_px": int(min(heights)),
            "max_height_px": int(max(heights))
        })
    
    all_heights = [l["median_height_px"] for l in per_line]
    all_heights_sorted = sorted(all_heights)
    median_line_h = float(np.median(all_heights_sorted))
    
    # 三层分级
    heading_range = None
    if len(all_heights_sorted) >= 2 and max(all_heights_sorted) >= median_line_h * 1.35:
        heading_range = f"{int(np.percentile(all_heights_sorted, 90))}-{int(max(all_heights_sorted))}px"
    body_range = f"{int(np.percentile(all_heights_sorted, 25))}-{int(np.percentile(all_heights_sorted, 75))}px"
    caption_range = None
    if len(all_heights_sorted) >= 2 and min(all_heights_sorted) <= median_line_h * 0.65:
        caption_range = f"{int(min(all_heights_sorted))}-{int(np.percentile(all_heights_sorted, 10))}px"
    
    return {
        "heading_range": heading_range,
        "body_range": body_range,
        "caption_range": caption_range,
        "per_line_heights": per_line,
        "line_count": len(lines),
        "median_height_px": int(median_line_h),
        "confidence": "estimated"
    }


@app.post("/detect_text")
async def detect_text(request: DetectTextRequest):
    img = load_image(request.image_source)
    h_img, w_img, _ = img.shape
    
    if request.bbox:
        ymin_norm, xmin_norm, ymax_norm, xmax_norm = request.bbox
        ymin = int(ymin_norm / 1000.0 * h_img)
        xmin = int(xmin_norm / 1000.0 * w_img)
        ymax = int(ymax_norm / 1000.0 * h_img)
        xmax = int(xmax_norm / 1000.0 * w_img)
        roi = img[max(0, ymin):min(ymax, h_img), max(0, xmin):min(xmax, w_img)]
    else:
        roi = img
    
    if roi.size == 0:
        raise HTTPException(status_code=400, detail="区域映射失败，ROI 为空。")
    
    result = _detect_text_mser(roi)
    return {"status": "success", "typography": result}


# --- 遮罩/弹窗检测（全局直方图统计矩分析） ---
class DetectOverlayRequest(BaseModel):
    image_source: str


@app.post("/detect_overlay")
async def detect_overlay(request: DetectOverlayRequest):
    """检测页面是否存在半透明遮罩/弹窗/骨架屏。
    使用全局灰度直方图的三阶统计矩（均值、方差、偏度），
    而非 5x5 网格硬阈值。对暗黑模式、Drawer、侧边弹窗均有效。"""
    img = load_image(request.image_source)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape
    
    # 全局统计矩
    mean_val = float(np.mean(gray))
    var_val = float(np.var(gray))
    skew_val = float(np.mean((gray - mean_val) ** 3)) / (max(1.0, var_val) ** 1.5)
    
    # 暗黑模式检测
    is_dark = mean_val < 85.0
    
    # 遮罩判定：方差被压缩（半透层抹平了对比度）+ 亮度偏移
    # 典型无遮罩 UI 方差 1500-4000，半透遮罩后方差 200-800
    overlay_present = False
    overlay_confidence = "high"
    
    if is_dark:
        # 暗黑模式：遮罩使画面进一步变暗 + 方差缩小
        if var_val < 600 and mean_val < 50:
            overlay_present = True
            overlay_confidence = "medium" if var_val > 300 else "high"
    else:
        # 亮色模式：遮罩拉低亮度 + 方差缩小
        if var_val < 1200 and mean_val < 160:
            overlay_present = True
            overlay_confidence = "medium" if var_val > 500 else "high"
    
    modal_bboxes = []
    if overlay_present:
        # 在遮罩区域扫描高对比度矩形（弹窗内容）
        # 将图像分块，找局部方差 > 全局方差 2x 的连通区块
        block_h, block_w = h_img // 8, w_img // 8
        local_var_map = np.zeros((8, 8), dtype=np.float32)
        for r in range(8):
            for c in range(8):
                patch = gray[r*block_h:(r+1)*block_h, c*block_w:(c+1)*block_w]
                if patch.size > 0:
                    local_var_map[r, c] = float(np.var(patch))
        
        # 找局部高方差块（弹窗内容特征）
        high_var = local_var_map > var_val * 2
        visited = np.zeros_like(high_var, dtype=bool)
        
        for r in range(8):
            for c in range(8):
                if high_var[r, c] and not visited[r, c]:
                    # BFS 找连通区域
                    region = []
                    queue = [(r, c)]
                    while queue:
                        cr, cc = queue.pop(0)
                        if 0 <= cr < 8 and 0 <= cc < 8 and high_var[cr, cc] and not visited[cr, cc]:
                            visited[cr, cc] = True
                            region.append((cr, cc))
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                queue.append((cr+dr, cc+dc))
                    
                    if len(region) >= 2:  # 至少 2 个高方差块
                        rows = [p[0] for p in region]
                        cols = [p[1] for p in region]
                        ymin = int(min(rows) * block_h / h_img * 1000)
                        xmin = int(min(cols) * block_w / w_img * 1000)
                        ymax = int((max(rows)+1) * block_h / h_img * 1000)
                        xmax = int((max(cols)+1) * block_w / w_img * 1000)
                        modal_bboxes.append({
                            "norm_bbox": [ymin, xmin, ymax, xmax],
                            "block_count": len(region)
                        })
    
    return {
        "status": "success",
        "overlay": {
            "present": overlay_present,
            "confidence": overlay_confidence,
            "global_stats": {
                "mean": round(mean_val, 1),
                "variance": round(var_val, 1),
                "skewness": round(skew_val, 3),
                "dark_mode": is_dark
            },
            "modal_bboxes": modal_bboxes,
            "modal_count": len(modal_bboxes)
        }
    }


# --- 组件树 IoA 拓扑修正 ---
class FixHierarchyRequest(BaseModel):
    components: list[dict]


def _bbox_intersection_area(a, b):
    i_ymin = max(a[0], b[0]); i_xmin = max(a[1], b[1])
    i_ymax = min(a[2], b[2]); i_xmax = min(a[3], b[3])
    if i_ymin >= i_ymax or i_xmin >= i_xmax:
        return 0
    return (i_ymax - i_ymin) * (i_xmax - i_xmin)


def _bbox_area(b):
    return max(0, (b[2] - b[0])) * max(0, (b[3] - b[1]))


@app.post("/fix_hierarchy")
async def fix_hierarchy(request: FixHierarchyRequest):
    """基于 IoA 修正 scan_global 的 parent_id。
    解决 OpenCV hierarchy 在视觉穿透场景下的错误（如 Badge 溢出按钮边界）。"""
    comps = request.components
    if not comps:
        return {"status": "success", "components": [], "tree": [], "corrections": 0}
    
    bboxes = [c["bbox"] for c in comps]
    corrections = 0
    
    for i, child in enumerate(comps):
        orig_parent = child.get("parent_id", "root")
        child_area = _bbox_area(bboxes[i])
        if child_area <= 0:
            continue
        
        best_ioa = 0.0
        best_parent_id = "root"
        for j, parent in enumerate(comps):
            if i == j:
                continue
            inter = _bbox_intersection_area(bboxes[i], bboxes[j])
            ioa = inter / child_area
            if ioa > best_ioa:
                best_ioa = ioa
                best_parent_id = parent["id"]
        
        if best_ioa >= 0.3 and best_parent_id != orig_parent:
            comps[i]["original_parent_id"] = orig_parent
            comps[i]["parent_id"] = best_parent_id
            comps[i]["ioa_score"] = round(best_ioa, 3)
            corrections += 1
        elif best_ioa < 0.3 and orig_parent != "root":
            comps[i]["original_parent_id"] = orig_parent
            comps[i]["parent_id"] = "root"
            comps[i]["ioa_score"] = round(best_ioa, 3)
            corrections += 1
    
    # 构建嵌套 children 树
    children_map = {}
    for c in comps:
        pid = c.get("parent_id", "root")
        children_map.setdefault(pid, []).append(c)
    
    def build_node(node_id):
        children = children_map.pop(node_id, [])
        for ch in children:
            if ch["id"] in children_map:
                ch["children"] = build_node(ch["id"])
            else:
                ch["children"] = []
        return children
    
    tree = build_node("root")
    for pid, nodes in children_map.items():
        for n in nodes:
            n["children"] = []
            tree.append(n)
    
    return {"status": "success", "corrections": corrections, "components": comps, "tree": tree,
            "correction_details": [{"id": c["id"], "original_parent": c.get("original_parent_id"),
                                    "new_parent": c["parent_id"], "ioa_score": c.get("ioa_score")}
                                   for c in comps if "original_parent_id" in c]}


# --- Set-of-Mark 打点图生成 ---
class DrawLabelsRequest(BaseModel):
    image_source: str
    components: list[dict]  # scan_global 的 components 数组
    max_labels: int = 50    # 标注数量上限


@app.post("/draw_labels")
async def draw_labels(request: DrawLabelsRequest):
    """在原图上画出 scan_global 组件的编号框图，返回 base64 PNG。
    VLM 看图报编号，后端按编号取精确 bbox。"""
    img = load_image(request.image_source)
    h_img, w_img, _ = img.shape
    
    overlay = img.copy()
    for i, comp in enumerate(request.components[:request.max_labels]):
        norm_bbox = comp["bbox"]
        ymin = int(norm_bbox[0] / 1000.0 * h_img)
        xmin = int(norm_bbox[1] / 1000.0 * w_img)
        ymax = int(norm_bbox[2] / 1000.0 * h_img)
        xmax = int(norm_bbox[3] / 1000.0 * w_img)
        
        # 画框（亮红色，2px）
        cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        # 标编号（红色大字，白色描边可读）
        label = str(i)
        pos = (xmin + 4, ymin + 24)
        cv2.putText(overlay, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
        cv2.putText(overlay, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    _, buf = cv2.imencode(".png", overlay)
    b64 = base64.b64encode(buf).decode()
    return {"status": "success", "image_base64": b64, "labeled_count": min(len(request.components), request.max_labels)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

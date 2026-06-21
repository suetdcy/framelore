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
    
    # --- 轮廓曲率分析：判断是否存在圆角 ---
    peri = cv2.arcLength(main_contour, True)
    approx = cv2.approxPolyDP(main_contour, 0.015 * peri, True)
    
    if len(approx) <= 4:
        # 多边形顶点 ≤ 4：尖锐直角，无圆角
        return 0, [x, y, w, h], "high", 0.0
    
    # 顶点数 > 4：存在曲线段 → 对四角区域做最小二乘圆拟合
    corner_radii, errors = _fit_corner_circles(main_contour, x, y, w, h)
    
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


def _fit_corner_circles(contour, x, y, w, h):
    """对轮廓四角区域分别拟合圆，返回半径列表和 RMSE 列表。"""
    corner_radii = []
    errors = []
    corner_zone = int(min(w, h) * 0.35)
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

# --- 渐变拟合（核心重构：自适应前置色彩孤立） ---
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
        direction = "180deg" if row_var > col_var else "90deg"
        
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
    labels_2d = kmeans.labels_ if not has_isolated_chroma else None
    stops = []
    
    for i in range(n_clusters):
        if has_isolated_chroma:
            pos = (i / max(1, n_clusters - 1)) * 100
            cluster_size = np.sum(kmeans.labels_ == i)
        else:
            coords = np.argwhere(labels_2d == i)
            if len(coords) == 0: continue
            if grad_type == "radial":
                cy, cx = h / 2, w / 2
                dists = np.sqrt((coords[:, 0] - cy)**2 + (coords[:, 1] - cx)**2)
                pos = np.mean(dists) / (np.sqrt(cy**2 + cx**2) or 1) * 100
            else:
                pos = np.mean(coords[:, 0]) / h * 100 if direction == "180deg" else np.mean(coords[:, 1]) / w * 100
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
    bbox: list[int]  
    mode: str = "kmeans"  # "kmeans" (full analysis) or "peak_accent" (top 5% saturation)
    accent_ratio_threshold: float = 0.3

class MeasureSpacingRequest(BaseModel):
    bboxes: list[list[int]]

class ScanGlobalRequest(BaseModel):
    image_path: str

@app.post("/analyze_region")
async def analyze_region(request: AnalyzeRegionRequest):
    img = load_image(request.image_source)
    h_img, w_img, _ = img.shape
    
    # 输入校验：bbox 必须为 [ymin, xmin, ymax, xmax] 且值在 [0, 1000]
    if len(request.bbox) != 4 or not all(0 <= v <= 1000 for v in request.bbox):
        raise HTTPException(status_code=400, detail="bbox must be [ymin, xmin, ymax, xmax] in [0, 1000]")
    ymin_norm, xmin_norm, ymax_norm, xmax_norm = request.bbox
    ymin = int(ymin_norm / 1000.0 * h_img)
    xmin = int(xmin_norm / 1000.0 * w_img)
    ymax = int(ymax_norm / 1000.0 * h_img)
    xmax = int(xmax_norm / 1000.0 * w_img)
    
    roi = img[max(0, ymin):min(ymax, h_img), max(0, xmin):min(xmax, w_img)]
    if roi.size == 0: raise HTTPException(status_code=400, detail="区域映射失败，ROI 为空。")
    
    # mode 路由：peak_accent 仅提取强调色，跳过全量分析
    if request.mode == "peak_accent":
        peak_hex, peak_conf = peak_accent_extraction(roi)
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
            "detected_shadow_blur_px": None  # [approximate] shadow analysis not yet implemented — reserved for future
        },
        "metrics_summary": {
            "border_radius_confidence": radius_conf,
            "border_radius_error_rate": radius_err,
            "color_gradient_confidence": color_info["confidence"]
        }
    }

@app.post("/measure_spacing")
async def measure_spacing(request: MeasureSpacingRequest):
    boxes = request.bboxes
    if len(boxes) < 2:
        return {"status": "success", "suggested_gap_x": 0, "suggested_gap_y": 0,
                "metrics_summary": {"gap_confidence": "high", "std_deviation_x": 0.0, "std_deviation_y": 0.0}}
    
    gaps_x = []
    gaps_y = []
    bx = sorted(boxes, key=lambda b: b[1])  
    by = sorted(boxes, key=lambda b: b[0])  
    
    for i in range(len(bx) - 1):
        gap = bx[i+1][1] - bx[i][3]  
        if gap >= 0: gaps_x.append(gap)
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
    if hierarchy is None: return {"status": "success", "components": []}
    hierarchy = hierarchy[0]
    
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if w > EngineConfig.MIN_COMPONENT_SIZE and h > EngineConfig.MIN_COMPONENT_SIZE:
            parent_idx = int(hierarchy[i][3])
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.015 * peri, True)
            
            ymin_norm = int(y / h_img * 1000)
            xmin_norm = int(x / w_img * 1000)
            ymax_norm = int((y + h) / h_img * 1000)
            xmax_norm = int((x + w) / w_img * 1000)
            
            # 面积噪声过滤：小于 0.3% 视口且无父容器 → 丢弃
            area_norm = (ymax_norm - ymin_norm) * (xmax_norm - xmin_norm)
            area_ratio = area_norm / 1_000_000
            if area_ratio < EngineConfig.MIN_COMPONENT_AREA_RATIO and parent_idx == -1:
                continue
            
            detected_components.append({
                "id": f"comp_{i}",
                "bbox": [ymin_norm, xmin_norm, ymax_norm, xmax_norm],  
                "parent_id": f"comp_{parent_idx}" if parent_idx != -1 else "root",
                "maybe_rounded": len(approx) > 4
            })
            
    return {"status": "success", "total_detected": len(detected_components), "components": detected_components}

# --- 字体大小估算（形态学 + 轮廓分析，无 OCR 依赖） ---
class DetectTextRequest(BaseModel):
    image_source: str
    bbox: Optional[list[int]] = None  # 可选局部区域 [ymin, xmin, ymax, xmax]


def _estimate_text_sizes(roi):
    """检测文本行并估算字号区间。返回按高度分组的层级表。"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # OTSU 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 形态学闭运算将文字聚合成行
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, gray.shape[1] // 30), 1))
    lines_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)
    
    contours, _ = cv2.findContours(lines_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    heights = []
    for cnt in contours:
        _x, _y, _w, h = cv2.boundingRect(cnt)
        if 6 <= h <= 200 and _w > h * 1.5:  # 宽高比 > 1.5 视为文字行
            heights.append(h)
    
    if not heights:
        return {"heading": None, "body": None, "caption": None, "all_heights": []}
    
    heights = sorted(heights)
    median_h = float(np.median(heights))
    
    # 按高度分三层
    tier = {}
    if max(heights) >= median_h * 1.3:
        tier["heading"] = f"{int(np.percentile(heights, 90))}-{int(max(heights))}px"
    else:
        tier["heading"] = None
    tier["body"] = f"{int(np.percentile(heights, 25))}-{int(np.percentile(heights, 75))}px"
    if min(heights) <= median_h * 0.7:
        tier["caption"] = f"{int(min(heights))}-{int(np.percentile(heights, 10))}px"
    else:
        tier["caption"] = None
    
    return {
        "heading_range": tier["heading"],
        "body_range": tier["body"],
        "caption_range": tier["caption"],
        "raw_heights_px": [int(h) for h in heights],
        "median_height_px": int(median_h),
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
    
    result = _estimate_text_sizes(roi)
    return {"status": "success", "typography": result}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

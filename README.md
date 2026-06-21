# FrameLore — UI/UX 设计提取技能

FrameLore 是一个基于多模态视觉模型 + 本地物理测量工具的 UI 设计提取技能。它从网页截图或在线页面中结构化提取视觉层（色值/字体/圆角/间距）和交互层（页面流程/弹窗/反馈），输出标准化的设计还原报告。

## 架构

```
┌──────────────────────────────────────────────────────┐
│              中枢推理层（LLM）                         │
│  qwen-vl-max / gpt-4o / doubao-vision               │
│  语义识别 · 布局意图分流 · 组件树构建 · UX 推断      │
└──────┬───────────────────────────────────────────────┘
       │ 调用 POST /analyze_region, /measure_spacing, /scan_global, /detect_text
       ▼
┌──────────────────────────────────────────────────────┐
│              外周工具层（image-analyzer）              │
│  本地 Python 服务 · OpenCV 像素级量化                │
│  色值 · 圆角 · 间距 · 字体 · 轮廓拓扑树              │
└──────────────────────────────────────────────────────┘
```

当外周工具不可用时，自动降级为纯 LLM 视觉推定，输出数据标注「视觉推定」。

## 前置依赖

### 1. 多模态视觉模型（至少配置一个）

| 服务商 | 环境变量 | 推荐模型 |
|--------|---------|---------|
| 阿里云百炼 | `DASHSCOPE_API_KEY` | qwen-vl-max |
| 火山引擎 | `DOUBAO_API_KEY` | doubao-vision |
| OpenAI | `OPENAI_API_KEY` | gpt-4o |

未配置时截图仅能输出结构清单，无法提取色值/字体/圆角等视觉样式。

### 2. 本地 image-analyzer 工具（推荐启动以获得最佳精度）

工具源码位于本目录下。

#### 安装

```bash
pip install fastapi uvicorn opencv-python numpy scikit-learn requests
```

#### 启动

```bash
python image_analyzer_v4.py
```

服务默认运行在 `http://127.0.0.1:8000`，提供六个端点：

| 端点 | 功能 | 返回数据 |
|------|------|---------|
| `POST /analyze_region` (mode=kmeans) | K-Means 背景/渐变提取 | 容器底色、渐变分布、圆角px、置信度 |
| `POST /analyze_region` (mode=peak_accent) | 强调色峰值提取 | 饱和度 Top 5% 极值强调色、置信度 |
| `POST /measure_spacing` | 间距度量 | 中位数间距、标准差、置信度 |
| `POST /scan_global` | 全局盲扫 | 轮廓拓扑树、`parent_id` 层级关系 |
| `POST /detect_text` | 字体大小估算 | 标题/正文/注释三级字号区间（px）、每行高度列表 |
| `POST /detect_overlay` | 遮罩/弹窗检测 | 半透明遮罩存在性、弹窗归一化 bbox 列表 |
| `POST /fix_hierarchy` | 组件树拓扑修正 | IoA 修正 parent_id、嵌套 children 树 |

#### 未启动时的行为

- 所有样式属性改为 LLM 视觉估算
- 数据标注「视觉推定」或「精度受限」
- 置信度校验、物理场优先等规则不可用

## 使用方法

在聊天中发送以下关键词即可激活：

- `UI 提取` / `UX 提取`
- `界面解析` / `截图分析`
- `设计还原` / `页面分析`

### 工作流

```
素材判断 → 布局意图分流 → 预处理 → 多轮推理融合 →
组件树构建 → 视觉提取(工具+LLM) → UI 提取 → UX 提取 →
Design System 提取 → 交叉校验 → CHECKPOINT → 写入文件
```

## 输出

- 报告文件：`framelore-output/FrameLore-{素材名}-{YYYYMMDD-HHmmss}.md`
- 可选 HTML：仅用于视觉验证（比对原图与报告数据），不作为主要交付物

## 关联

- `/vision` — 多模态视觉分析工具
- 可作为竞品 UI 分析、设计还原审查的前置提取步骤

---

## 常见问题与解决方案

### 1. PowerShell 调用 API 后中文乱码（问号/乱码字符）

**现象**：从 DashScope API 返回的中文变成 `??��`、`æ¥` 等乱码。

**原因**：在读取 API 响应时使用了编码转换：
```powershell
# ❌ 错误写法 — 经过 Default (GBK) 转换导致损坏
[System.Text.Encoding]::UTF8.GetString([System.Text.Encoding]::Default.GetBytes($text))
```

**解决**：直接取原始文本属性，不做任何编码转换：
```powershell
# ✅ 正确写法
$text = $response.output.choices[0].message.content[0].text
```

---

### 2. image-analyzer 报 `Internal Server Error` / `KMeans` 无 `cluster_centers`

**现象**：调用 `POST /analyze_region` 返回 500 错误。

**原因**：新版本 scikit-learn 将 `KMeans.cluster_centers`（属性）改为 `cluster_centers_`（末尾下划线）。

**解决**：修改 `image_analyzer_v4.py` 中对应行：
```python
# ❌ 旧版
centers = kmeans.cluster_centers()
# ✅ 新版
centers = kmeans.cluster_centers_
```

---

### 3. image-analyzer 无法加载中文路径图片

**现象**：`{"detail":"图像加载失败。"}` — `scan_global` 和 `analyze_region` 都失败。

**原因**：OpenCV 的 `cv2.imread()` 在 Windows 上不支持包含非 ASCII（中文）字符的文件路径。

**解决**：先将图片复制到无中文的临时路径（如 `%TEMP%`）再传入：
```powershell
$tmp = Join-Path $env:TEMP "extract.png"
Copy-Item "中文路径/图片.png" $tmp -Force
# 然后将 $tmp 作为 image_source / image_path 传入
```

---

### 4. 浏览器外壳文本污染（非激活标签页文字窜入页面）

**现象**：生成的 HTML `<title>` 错误地使用了浏览器非激活标签页中的文本（如「新手先用 — Agentic Infrastructure」），而非当前页面的实际标题。

**原因**：多模态模型在全局语义聚类时，误将灰色非激活标签页文本识别为页面元数据。

**解决**：在预处理时裁剪浏览器头部区域（外周物理层），辅以 SKILL.md 中的「报告输出红线」规则拦截。

---

### 5. 生成色值过于饱和/荧光，与原图不符

**现象**：还原的页面颜色比原图鲜艳刺眼，暗黑模式变成亮色。

**原因**：LLM 在无工具数据时调用预训练「典型模板」覆盖了原图实际色值。

**解决**：启动 image-analyzer 工具获取物理色值；启用 SKILL.md 中的饱和度约束（0.7x 衰减系数、HSL 中 S ≤ 60%）和 Alpha 通道硬约束（0.15~0.4）。

---

### 6. Retina 截图下坐标缩放偏差

**现象**：模型输出的像素坐标（如 1440px）与实际截图分辨率（如 2559px）不一致，导致标注位置偏移。

**原因**：多模态模型内部降采样处理 Retina 截图，输出的是逻辑视口尺寸而非物理像素。

**解决**：启用归一化坐标空间（`[0, 1000]` 千分位坐标），由 `image-analyzer` 根据实际分辨率反算物理像素。详见 SKILL.md「报告输出红线」第 1 条（The Gauge Invariant）。

---

### 7. 文本被语义补全/改写

**现象**：因原始文本乱码或 OCR 不清晰，LLM 根据上下文「脑补」了原文中没有的词（如将「日常工作流」补成「日常高频工作流」）。

**原因**：违反 Strict-Copy Protocol — 模型在文本不可读时选择了猜测而非标记错误。

**解决**：启用 Strict-Copy Protocol #3（乱码熔断与退避），遇到编码损坏字符时标记 `[ENCODING_ERROR]` 而非脑补。

---

## 已知精度限制

| 能力 | 状态 | 说明 |
|------|------|------|
| **阴影/发光检测** | ✅ 已实现 | 高斯差分金字塔 (DoG)，多尺度 σ 探测弥散半径 2-32px。对渐变背景/暗黑模式/新拟态鲁棒 |
| **字体大小估算** | ✅ 已实现 | MSER 字符检测 + 空间行分组。自适应图片分辨率（δ 参数），对 Retina/中文/暗黑均可靠。输出 `per_line_heights` 每行统计 |
| **渐变角度** | ✅ 高精度 | Sobel 梯度加权直方图，对齐 15° CSS 步长 |
| **圆角检测** | ✅ 高精度 | 轮廓曲率分析 + 四角最小二乘圆拟合，自适应 corner_zone |
| **强调色提取** | ✅ 高精度 | `peak_accent` 模式：饱和度 Top 5% + KMeans 聚类内标准差校验。暗黑模式 0.7x 衰减 |
| **遮罩/弹窗检测** | ✅ 已实现 | 全图直方图统计矩（均值/方差/偏度），8x8 块局部方差 BFS 扫描弹窗区域 |
| **组件树层级** | ⚠️ 部分自动化 | `scan_global` 返回扁平列表（含 `hint_type`）。`/fix_hierarchy` 基于 IoA 修正 parent_id 并输出嵌套树。微小组件仍可能因 Canny 边缘阈值漏扫 |
| **UX 交互层** | ⚠️ 依赖 VLM | 无工具侧交互语义检测（点击/跳转/校验逻辑依赖 LLM 视觉推理）。工具仅提供 overlay/modal 的结构化检测 |


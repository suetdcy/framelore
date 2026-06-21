---
name: framelore
description: 从网页、截图等界面素材中，结构化提取 UI 视觉层与 UX 交互层信息，输出高精度物理量化与语义对齐报告。
---

# Skill: FrameLore

## 🧠 认知模型（叙事层）

FrameLore 采用 **「VLM 语义锚定 + CV 物理探针」** 的双驱解耦架构。系统将大模型（VLM）视为具备设计常识的「语义切割器」，将底层 `image-analyzer` 视为纯粹客观的「物理量化仪」。
本 Skill 的终极交付物是结构化、可被机器与解析脚本无脑读取的高精度 Markdown 数据库，严禁输出任何发散性的自然语言散文或直接编写 HTML/CSS 渲染代码。

## 触发规则

- `UI 提取` / `UI 还原` / `UX 提取`
- `界面解析` / `界面提取` / `截图分析` / `截图提取`
- `设计还原` / `页面分析`

不激活：数据分析、文案总结、图片修图、视频剪辑。
边界：只提取不生成、不写渲染代码、不调素材、先判类型。

## 工具链依赖

| 工作类型 | 执行者 | 典型任务 |
|---------|--------|---------|
| 语义切割 | LLM 多模态推理 | 构建组件树、提供靶向千分位 BBox、分配组件语义角色 |
| 物理量化 | `image-analyzer` 工具 | 色值峰值提取、几何量化、物理坐标系标定 |

`image-analyzer v4.2`（`http://127.0.0.1:8000`）：

| 端点 | 核心输入 | 返回与功能描述 |
|------|---------|--------------|
| `POST /analyze_region` (mode=kmeans) | `{image_source, bbox, mode}` <br>⚠️ *`bbox` 为 [ymin, xmin, ymax, xmax] 归一化空间* | K-Means 聚类提取背景主色/渐变分布，适合容器底色大面积区域。|
| `POST /analyze_region` (mode=peak_accent) | 同上 | 饱和度 Top 5% 极值强调色提取，适合独立高亮控件（Badge/Tag/按钮）。|
| `POST /measure_spacing` | `{bboxes:[[ymin, xmin, ymax, xmax], ...]}` | 物理间距计算，返回中位数间距、标准差及置信度。|
| `POST /scan_global` | `{image_path}` | 获取全域组件的边缘轮廓与物理绝对坐标树。|

### 置信度数学定义

**圆角置信度 (Border Radius Confidence)** — 霍夫圆变换/轮廓曲率拟合 RMSE：

| RMSE | 置信度 | 含义 |
|:----:|:------:|------|
| ≤ 0.5px | High | 边缘清晰，像素级精准 |
| 0.5 ~ 1.5px | Medium | 存在反锯齿或轻微模糊，允许向标准 Token（4/8/12px）对齐 |
| > 1.5px | Low | 边缘严重混淆，强制输出区间值或 `[UNVERIFIED]` |

**间距标准差 (Spacing Std Deviation)** — measure_spacing 样本标准差 $\sigma = \sqrt{\frac{\sum(x_i - \mu)^2}{n}}$：

| $\sigma$ | 置信度 | 含义 |
|:--------:|:------:|------|
| ≤ 1.0 | High | 间距高度一致，直接信赖 |
| 1.0 ~ 2.5 | Medium | 存在微小排版抖动，可取整对齐 |
| > 2.5 | Low | 数据离散度极高，阻断单值，改用区间 |

---

## 🚨 微观语义切割契约 (Semantic Cropping Protocol)

大模型在调用 `POST /analyze_region` 获取色彩数据时，**绝对禁止**将包含文本、多个按钮、复杂插画的整张大卡片作为一个单一的 `bbox` 丢给工具。大模型必须像外科手术刀一样，基于视觉语义进行精准靶向切割调用：

1. **提取容器背景色/整体渐变时**：
   - 框选目标容器的**纯净空白区域**（避开复杂插图与密集的文本区域）。
   - 参数指令：强制传入 `"mode": "kmeans"`。

2. **提取独立高亮控件（如 Tag、按钮、Badge）时**：
   - 框选必须**极致缩小，仅仅包裹住该单一控件的物理边缘**（例如仅仅框住绿色标签，绝不能包含外部容器的底色）。
   - 参数指令：强制传入 `"mode": "peak_accent"`。工具将无视底噪，返回绝对纯正的单一强调色。

3. **面对多种强调色离散并存（如红按钮 + 绿标签）时**：
   - **严禁行为**：严禁将红绿按钮框在一起发起单次调用。
   - **正确执行**：必须发起**多次独立调用**。一次只精准框选红按钮（传 `peak_accent`），另一次只精确框选绿标签（传 `peak_accent`）。确保物理量化绝对不被跨色域污染。

4. **防御创意插画与弥散光干扰**：
   - 当背景存在柔和彩光、无边框弥散阴影或复杂插图时，大模型必须在视觉上绕过这些「创意污染区」，只针对存在明确 UI 交互意图的硬边缘控件发起微观 `bbox` 取色。

---

## 📐 布局意图判定 (Layout Prior)

在微观提取前必须率先判定排版制式，锁定后后续步骤不得随意推翻。

| 布局类型 | 判定条件 |
|---------|---------|
| **Grid** | 水平方向 ≥2 个重复子语义区块，统一 Y 轴基线 |
| **Stack** | 组件自上而下规律排列 |
| **Center** | 视觉核心居中，两侧松散留白 |
| **Freeform** | 故意打破网格、偏心错位 |
| **Hybrid** | 页面同时存在 ≥2 个主导布局体系 |

**Fail Safe（熔断重评估，最多一次）**：若后续推理出现系统性矛盾（parent_id 拓扑树持续冲突 / 80% 以上组件不符合当前 Layout Prior / 工具数据与 Layout Prior 产生冲突），允许重新评估 Layout Prior **最多一次**。第二次冲突时以首次判定为准，并在报告中标注「布局判定冲突」。

---

## 🛡️ 物理场事实零改写转录协议 (Zero-Heuristic Fact Transcription)

大模型在处理工具返回的冰冷物理事实时，严禁在写入 `.md` 报告前进行任何主观「渐变合成」、「分类」或融合脑补。报告必须是一面绝对诚实的镜子：

1. **色彩原样过账**：从 `/analyze_region` 获取的 `raw_peak_accents` 数组或 `raw_background_median`，无论其视觉表现如何，必须 100% 原样转录至 YAML 色彩池中。
2. **暗黑模式饱和度硬钳位约束（唯一允许的数学后处理）**：
   - 当原图判定为暗黑模式时，大模型在转录色彩池前，必须对提取的高亮色 Hex 进行 HSL 自检。
   - **执行限幅**：若 $S > 70\%$，必须人为执行 0.7x 饱和度衰减，且最终写入 YAML 报告的色值其饱和度严禁超过 60%（红/橙等功能性强警告色除外）。此步骤为纯直方图数学限幅，确保暗黑模式下的阅读舒适性。

---

## 🔒 报告输出红线：零幻觉与物理锚定

1. **坐标禁飞区（The Gauge Invariant）**：组件树中的任何节点必须携带 `norm_bbox: [ymin, xmin, ymax, xmax]`。没有任何组件可以凭空存在，严禁输出绝对像素坐标。
2. **零脑补熔断**：当工具链未返回特定属性时，严禁根据设计常识自行编造 CSS 属性。必须强制填入 `null` 或 `[UNVERIFIED]` 占位符。
3. **文本原像素拷贝 (Strict-Copy Protocol)**：所有从 UI 素材中提取的文本，必须执行字节级精确拷贝。严禁纠正拼写、严禁补全省略号、严禁翻译。遇到物理层编码损坏的乱码，强制标记为 `[ENCODING_ERROR]`。
4. **禁止发散性描述**：禁止使用任何自然语言散文描述 UI（如「这里有一个漂亮的红色按钮」）。必须全部收敛于 JSON 键值对。

---

## ⚖️ 冲突解决优先级链 (Conflict Resolution)

当多源数据冲突时，按以下顺序裁决：

**物理场（工具测量） > 拓扑树（scan_global） > 语义场（LLM 推理） > 设计意图（Layout Prior）**

具体规则：
- `scan_global` 扫出 N 个轮廓但 LLM 只识别了其中 M 个（M < N）→ 以工具轮廓数为准，触发二次核查
- `measure_spacing` 返回 $\sigma > 2.5$ → 阻断单值输出，改用区间值
- 组件坐标/色值以 `analyze_region` 返回值为最终值，LLM 估算仅作参考
- Layout Prior 不得覆盖工具的物理测量事实

---

## 🏁 终极交付物：高精度数据化 MD 报告模板

大模型在完成推理与工具调用后，**必须且只能**输出以下结构化格式的 Markdown 数据 Dump，不得删减层级或篡改数据架构：

```markdown
# FrameLore 物理量化提取报告

## 1. 探针遥测数据 (Telemetry)
- **分析器版本**: v4.2 Chroma-Isolated Engine
- **全局组件基数**: {scan_global 返回的轮廓总量}
- **Layout Prior**: {Grid/Stack/Center/Freeform/Hybrid} [Confidence: {High/Medium/Low}]

## 2. 物理场设计变量池 (Design Tokens)
```yaml
design_tokens:
  global_spacing:
    gap_x: "{suggested_gap_x}px"
    gap_y: "{suggested_gap_y}px"
    confidence: "{gap_confidence}"
  
  color_pool:
    container_background: "{raw_background_median 原始值 / 或 [UNVERIFIED]}"
    # 严格平铺由 peak_accent 独立提取的所有纯净峰值色值，严禁 LLM 擅自合并为渐变
    extracted_chroma_peaks:
      - hex: "{独立提取的 Hex 1}"
        role: "primary/main"
        confidence: "{confidence}"
      - hex: "{独立提取的 Hex 2 (若有)}"
        role: "secondary/badge"
        confidence: "{confidence}"
```

## 3. DOM 级高精度组件树 (Component Tree)
```json
// 必须严格输出嵌套层级，确保 parent_id 关系与 norm_bbox 物理包裹逻辑绝对吻合
[
  {
    "id": "node_01",
    "parent_id": "root",
    "semantic_tag": "Container",
    "norm_bbox": [ymin, xmin, ymax, xmax],
    "computed_style": {
      "border_radius": "{radius}px [Confidence: High/Medium/Low]",
      "background_color": "{匹配到的底色}"
    },
    "children": [
      {
        "id": "node_02",
        "parent_id": "node_01",
        "semantic_tag": "Button",
        "norm_bbox": [ymin, xmin, ymax, xmax],
        "content_text": "确 认" // Strict-Copy 严格保留原始空格或特殊符号
      }
    ]
  }
]
```

## 4. 规则度量统计 (Rule Telemetry)
```yaml
triggered_rules:
  - {根据执行链真实记录触发的隐含规则，如 Semantic-Cropping, Strict-Copy, Dark-Mode-Clamp}
```
```

---

## 失败模式与降级策略

| 场景 | 物理动作 | 语义降级处理 |
|------|---------|-------------|
| 工具离线/超时 | 停止 `/analyze_region` 调用 | `[CRITICAL_WARNING: PHYSICAL_FIELD_OFFLINE]`，产物降级为无样式的纯结构 JSON 清单 |
| 创意弥散光背景 | 规避背景框选，只切核心文字 | 容器背景标记 `[UNVERIFIED: Creative Mesh Gradient]` |
| 工具置信度持续 Low | 阻断绝对单值输出 | 启用区间值（如 `8-12px`），仍无法确认则标记 `[UNVERIFIED]` |

## 关联环境与接口

- **多模态核心**：推荐配置 `gpt-4o`, `qwen-vl-max`, 或 `doubao-vision` 级模型确保 BBox 靶向切割精度。
- **物理执行层**：强依赖本地 `http://127.0.0.1:8000` 提供图像张量计算。

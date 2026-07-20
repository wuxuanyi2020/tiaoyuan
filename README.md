# 立定跳远自动检测系统

> 代码检视日期：2026-07-20。本文档根据当前 `main.py` 与 `src/` 下实际代码逻辑整理。命令行默认参数以 `python main.py --help` 为准。

本项目基于 **OpenCV + MediaPipe Pose** 自动识别立定跳远视频中的垫子、人体关键点、起跳帧和落地帧，输出成绩、犯规判断和可视化图片。距离精修支持两条可选路径：

- `--diff`：MOG2 背景差分 + 边缘/投影极值修正，最终标注为 `diff-mog2-fixed`。
- `--yolo VERSION SCALE`：YOLO 实例分割修正，模型文件放在 `yolo_model/` 下。

如果同时传入 `--diff` 和 `--yolo`，代码会优先启用 YOLO，`--diff` 自动不生效。

## 目录结构

```text
tiaoyuan/
├── main.py                         # CLI 入口：单视频/批量处理
├── README.md
├── src/
│   ├── config.py                   # JumpConfig、视频源解析、结果目录创建
│   ├── core/
│   │   └── jump_system.py          # 主状态机、起跳/落地、修正结果采用、输出保存
│   ├── inference/
│   │   ├── mat_calibration.py      # 垫子自动/手动标定、HDR/强光适配、mask/grid 输出
│   │   ├── pose_estimator.py       # MediaPipe PoseLandmarker / Legacy Pose 封装
│   │   ├── diff_detector.py        # MOG2 差分与 YOLO 分割距离修正、阶段图输出
│   │   ├── shoe_detector.py        # 传统鞋边缘检测模块（当前主流程未作为核心修正方式）
│   │   └── shadow_remover.py       # HomoFormer 阴影移除封装（当前 CLI 未直接接入）
│   ├── rules/
│   │   └── foul_detection.py       # 犯规规则检测
│   └── visualization/
│       └── rendering.py            # 标注绘制、中文文字、中文路径安全写图
├── videos/                         # 本地视频目录，默认被 .gitignore 忽略
├── yolo_model/                     # YOLO seg 模型目录，*.pt 默认被 .gitignore 忽略
└── result/                         # 运行输出目录，默认被 .gitignore 忽略
```

## 环境安装

推荐 Python 3.10+。当前本地开发环境为 Python 3.12。

```bash
conda create -n tiaoyuan python=3.12 -y
conda activate tiaoyuan

pip install opencv-python numpy mediapipe Pillow -i https://pypi.tuna.tsinghua.edu.cn/simple

# 如果需要 YOLO 实例分割修正
pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
```

首次运行 MediaPipe PoseLandmarker 时，程序会尝试自动下载 `pose_landmarker_heavy.task` 到用户目录 `~/.mediapipe/`。如果自动下载失败，请手动下载模型并放到对应目录。

YOLO 模型请放到项目根目录 `yolo_model/`：

| 命令 | 期望模型文件 |
|---|---|
| `--yolo 8 x` | `yolo_model/yolov8x-seg.pt` |
| `--yolo 11 x` | `yolo_model/yolo11x-seg.pt` |
| `--yolo 26 x` | `yolo_model/yolo26x-seg.pt` |

支持版本：`8`、`11`、`26`；支持尺度：`n`、`s`、`m`、`l`、`x`。

## 快速开始

### 单视频

```bash
# 骨骼关键点基础流程
python main.py --video videos/跳远1-9-1080p.mp4 --no-display

# MOG2 差分修正，输出 score 图中的 diff-mog2-fixed 点/线
python main.py --video videos/跳远1-9-1080p.mp4 --diff --no-display

# YOLOv26x 实例分割修正
python main.py --video videos/跳远1-16.mp4 --yolo 26 x --no-display

# 强光/HDR 场景同时输出垫子 mask 与网格图，便于检查标定范围
python main.py --video videos/跳远1-16.mp4 --enable-mat-output --test-grid --no-display
```

### 批量处理

```bash
# 默认批量处理 videos/跳远1-1.mp4 到 videos/跳远1-9.mp4
python main.py --batch --diff

# 指定视频列表
python main.py --videos videos/跳远1-9-1080p.mp4 videos/跳远1-16.mp4 --diff
```

批量处理会在 `result/summary_YYYYMMDD_HHMMSS.csv` 中保存汇总。

## 常用参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--video` | `0` | 视频路径或摄像头索引 |
| `--save` | `result.json` | CLI 保留参数；当前主流程仍固定写入每次运行目录下的 `result.json` |
| `--no-display` | 关闭 | 不显示 OpenCV 预览窗口 |
| `--debug-dir` | 无 | CLI 保留参数，配置中可传入调试目录 |
| `--record` | 无 | 保存带标注的输出视频 |
| `--mat-length-cm` | `338.0` | 垫子物理长度 |
| `--mat-width-cm` | `100.0` | 垫子物理宽度 |
| `--takeoff-line-cm` | `31.0` | 起跳线物理 X 坐标 |
| `--takeoff-offset-cm` | `3.0` | 骨骼起跳点固定补偿 |
| `--landing-offset-cm` | `-5.0` | 骨骼落地点鞋跟补偿，负值会缩短距离 |
| `--trigger-move-cm` | `32.0` | 起跳触发位移阈值 |
| `--trigger-frames` | `2` | 起跳触发连续帧数 |
| `--min-flight-frames` | `5` | 最少飞行帧数 |
| `--max-jump-frames` | `120` | 最大跳跃帧数兜底 |
| `--manual-calib` | 关闭 | 手动四点标定 |
| `--no-foul-detection` | 关闭 | 禁用犯规检测 |
| `--debug` | 关闭 | 输出起跳/落地触发条件到日志 |
| `--diff` | 关闭 | 启用 MOG2 背景差分修正 |
| `--yolo VERSION SCALE` | 关闭 | 启用 YOLO 实例分割修正 |
| `--enable-mat-output` | 关闭 | 输出垫子识别 mask 图 |
| `--test-grid` | 关闭 | 输出垫子网格检查图 |

## 核心流程

```text
读取视频/摄像头
  ↓
自动或手动标定绿色跳远垫，建立 H_img2mat / H_mat2img 透视矩阵
  ↓
MediaPipe 检测人体关键点，支持多人关键点列表
  ↓
IDLE：等待人体进入垫子并靠近起跳线
  ↓
READY：记录稳定站立基线，检测脚尖、脚踝、髋部突变
  ↓
JUMPING：确认起跳，保存起跳帧，等待落地或超时兜底
  ↓
LANDED：骨骼法计算基础起跳点、落地点、成绩
  ↓
可选 YOLO 或 MOG2 修正起跳/落地 X
  ↓
使用最终采用的起跳/落地点重新计算成绩与犯规
  ↓
保存 result.json、score.jpeg、takeoff.jpeg、landed.jpeg、阶段调试图和日志
```

## 垫子标定逻辑

`src/inference/mat_calibration.py` 负责垫子范围识别与透视矩阵构建。当前逻辑包括：

1. HSV 绿色区域与强光低饱和高亮区域提案。
2. 形态学闭运算/开运算补全垫子边角、圆角和锯齿缺失。
3. 多轮廓合并、四边形候选评分、边缘线拟合。
4. HDR/强光适配：检测画面过亮且饱和度偏低时，使用更严格的 hue 下界重检，避免垫子范围外扩。
5. 构建 `H_img2mat`（图像坐标 → 垫子 cm 坐标）与 `H_mat2img`（垫子 cm 坐标 → 图像坐标）。

建议在强光或 HDR 原视频中使用：

```bash
python main.py --video videos/跳远1-16.mp4 --enable-mat-output --test-grid --no-display
```

重点检查输出的 `images/test_grid.jpeg` 是否贴合垫子边界。`mat_mask_hsv.jpeg` 只是颜色可见区域，中间正常但边角略缺失不一定代表最终标定错误；最终以 `mat_mask_quad.jpeg` 和 `test_grid.jpeg` 为准。

## 距离计算与修正优先级

### 骨骼基础成绩

基础流程使用 MediaPipe 关键点检测：

- 起跳：以前脚脚尖/脚踝/髋部变化确定起跳帧，并应用 `takeoff_offset_cm`。
- 落地：以脚后跟触地位置为主，并应用 `landing_offset_cm`。
- 基础成绩：`landing_x_cm - takeoff_x_cm`。

### YOLO 实例分割修正

启用 `--yolo VERSION SCALE` 后，`src/inference/diff_detector.py` 会加载对应 `*-seg.pt` 模型：

```text
YOLO person mask → 脚部 ROI → 切除上部干扰 → 清理轮廓 → 投影到垫子坐标 → 取 toe/heel 物理 X 极值
```

YOLO 修正值会覆盖骨骼基础值。输出目录为 `images/yolo/`。

### MOG2 差分修正（`--diff`）

启用 `--diff` 后，程序会在垫子标定完成且垫子范围内无人时捕获基准帧，然后在起跳/落地帧的脚部 ROI 中提取鞋子 mask：

```text
基准帧 ROI + 当前帧 ROI
  ↓
灰度差分 + MOG2 前景 + Sobel 边缘差分支持
  ↓
形态学清理、轮廓实心填充 SolidMask
  ↓
切除 ROI 上部 20% 干扰区域
  ↓
投影 SolidMask 到 Mat_Projection 俯视图
  ↓
起跳 toe 取俯视图最右侧边界像素，落地 heel 取俯视图最左侧边界像素
  ↓
得到 diff-mog2-fixed 起跳/落地 X，并覆盖骨骼基础值计算成绩
```

关键点：

- `score.jpeg` 中会画出紫色 `diff-mog2-fixed` 起跳点/线和落地点/线。
- `score.jpeg` 左上角在 `--diff` 模式下显示 `diff-mog2-fixed`，不会显示 `yolo`。
- 成绩使用最终采用的差分修正点计算：`diff_mog2_landing_x - diff_mog2_takeoff_x`。
- Stage3 调试图中的 `Mat_Projection` 黄点与投影 mask 的最右/最左边界一致，避免原图透视变换取整造成视觉偏差。
- ROI 可能左右脚重叠，代码会记录最终选中的脚，只在对应脚的 Stage3 行上画 toe/heel 点。

MOG2 输出目录为 `images/diff/`，主要文件包括：

| 文件 | 说明 |
|---|---|
| `diff-Stage1-base.jpeg` | 垫子内无人基准帧 |
| `diff-Stage2-roi-takeoff.jpeg` / `diff-Stage2-roi-landing.jpeg` | 起跳/落地脚部 ROI |
| `diff-Stage3-edge-takeoff.jpeg` / `diff-Stage3-edge-landing.jpeg` | Raw、GrayDiff、MOG2/Edge、PreClean、SolidMask、Mat_Projection |
| `diff-Stage4-takeoff.jpeg` / `diff-Stage4-landing.jpeg` / `diff-Stage4-combined.jpeg` | 最终 mask 叠加图 |

## 输出结果

每次运行会创建：

```text
result/<视频名>/<视频名>_YYYYMMDD_HHMMSS/
├── result.json
├── images/
│   ├── takeoff.jpeg
│   ├── landed.jpeg
│   ├── score.jpeg
│   ├── test_grid.jpeg                 # 开启 --test-grid 时
│   ├── mat_mask_quad.jpeg             # 开启 --enable-mat-output 时
│   ├── mat_mask_hsv.jpeg              # 开启 --enable-mat-output 时
│   ├── diff/                          # 开启 --diff 时
│   └── yolo/                          # 开启 --yolo 时
└── logs/
    └── run.log
```

`result.json` 字段：

| 字段 | 说明 |
|---|---|
| `score` / `distance_cm` | 最终成绩，单位 cm |
| `valid` | 是否有效，犯规则为 `false` |
| `foul_reason` | 犯规原因，无犯规则为 `null` |
| `takeoff_x_cm` | 最终采用的起跳 X |
| `landing_x_cm` | 最终采用的落地 X |
| `yolo_infer_time_s` | YOLO 累计推理耗时；非 YOLO 模式通常为 0 |

## 犯规规则

`src/rules/foul_detection.py` 当前包含：

| 规则 | 检测概要 |
|---|---|
| 踩线 | 最终采用的起跳 X 超过起跳线容差 |
| 垫步/起跳前异常 | 起跳前脚尖历史位移过大 |
| 单脚起跳异常 | 双脚脚踝 X/Y 差异过大 |
| 多人入界 | 垫子范围内同时检测到多个人体关键点 |
| 出界 | 落地点 Y 超出垫子宽度范围 |
| 撑杆/手部辅助 | 手腕位置低于膝盖等异常姿态 |

关闭犯规检测：

```bash
python main.py --video videos/跳远1-9-1080p.mp4 --no-foul-detection --no-display
```

## 代码逻辑检视摘要

本次检视范围为项目主流程文件，不包含外部依赖源码目录 `HomoFormer-master/` 和 `mediapipe-*`。

| 文件 | 当前职责 | 检视结论 |
|---|---|---|
| `main.py` | CLI 参数、单视频/批量入口、CSV 汇总 | `--diff` 与 `--yolo` 互斥优先级清晰；YOLO 优先 |
| `src/config.py` | dataclass 配置、视频源解析、结果目录 | 配置字段与 CLI 基本对应，CLI 默认值为准 |
| `src/core/jump_system.py` | 状态机、成绩采用、score/takeoff/landed 输出 | `--diff` 后最终成绩使用 MOG2 修正点；score 图显示 `diff-mog2-fixed` |
| `src/inference/mat_calibration.py` | 垫子识别、HDR/强光适配、透视矩阵 | 支持强光/HDR 重检、mask/grid 输出 |
| `src/inference/pose_estimator.py` | MediaPipe PoseLandmarker 与回退 Pose | 支持本地模型缓存、视频/图片读取、多人关键点 |
| `src/inference/diff_detector.py` | YOLO/MOG2 修正与阶段图 | MOG2 从投影后的 SolidMask 取 toe/heel 极值，Stage3 黄点与俯视 mask 边界一致 |
| `src/rules/foul_detection.py` | 犯规规则 | 由主流程在最终修正点确定后重新判定踩线 |
| `src/visualization/rendering.py` | 绘制标注、中文文本、安全写图 | 支持中文路径 `imwrite_safe` 与中文文字渲染 |

基础语法检查命令：

```bash
python -m py_compile main.py src/config.py src/core/jump_system.py src/inference/diff_detector.py src/inference/mat_calibration.py src/inference/pose_estimator.py src/inference/shoe_detector.py src/inference/shadow_remover.py src/rules/foul_detection.py src/visualization/rendering.py
```

## 常见问题

### 1. PowerShell 中 `cd /d` 报错

`cd /d` 是 cmd 语法，PowerShell 请使用：

```powershell
Set-Location -LiteralPath 'D:\DeepLearning\跳远_mat\tiaoyuan\tiaoyuan'
```

cmd 中可以使用：

```cmd
cd /d D:\DeepLearning\跳远_mat\tiaoyuan\tiaoyuan
```

### 2. GitHub push 出现 `Connection was reset`

这通常是网络连接被重置，不代表仓库不存在。可检查代理、网络、GitHub 可访问性，稍后重试 `git push origin main`。

### 3. HDR 原视频和 SDR 截图标定不一致

HDR 视频经 OpenCV 解码后可能更亮、更低饱和。当前代码已加入强光/HDR 自适应；如果仍有偏差，请先输出 `test_grid.jpeg` 检查最终透视标定是否贴合。

### 4. `mat_mask_hsv.jpeg` 边角缺失是否一定有问题？

不一定。`mat_mask_hsv.jpeg` 是颜色阈值可见区域，边角会受圆角、反光、锯齿、遮挡影响。最终判断应看 `mat_mask_quad.jpeg` 和 `test_grid.jpeg`。

### 5. 找不到 YOLO 模型

确认模型位于 `yolo_model/` 且文件名与命令一致，例如 `--yolo 26 x` 需要 `yolo_model/yolo26x-seg.pt`。模型体积较大，默认不会提交到 Git。

### 6. Windows 终端中文乱码

代码和 README 使用 UTF-8。部分 Windows cmd 显示中文 help 时可能乱码，但不影响结果 JSON、图片和日志。建议使用支持 UTF-8 的终端。

## Git 忽略策略

`.gitignore` 当前会忽略：

- `result/`、`videos/`、`*.mp4`、`*.pt` 等大文件或运行产物。
- `HomoFormer-master/`、`mediapipe-*/`、`backup/` 等外部依赖/备份目录。
- `github_token.txt`、`.env` 等凭据文件。

提交前建议检查：

```bash
git status --short
git diff -- README.md src/core/jump_system.py src/inference/diff_detector.py
```

## License

MIT

# 立定跳远自动检测系统

基于 OpenCV + MediaPipe 骨骼关键点的立定跳远视频自动检测系统。程序会自动标定绿色跳远垫，建立图像坐标到垫子物理坐标（cm）的透视变换，识别起跳/落地事件，计算成绩，并可进行犯规判定。距离精修可选 MOG2 差分或 YOLO 实例分割。

> 当前 README 按 `main.py` 和 `src/` 下的实际代码整理。命令行默认值以 `main.py --help` 为准；`src/config.py` 中的 dataclass 默认值主要用于代码级调用。

## 主要功能

- **自动垫子标定**：默认使用 HSV 绿色/强光低饱和高亮区域提案 + 形态学补全 + 四边形候选评分 + 边缘线拟合，输出透视矩阵。支持 `--manual-calib` 手动四点标定。
- **HDR/强光适配**：针对 HDR 原视频比 SDR 截图更亮、饱和度更低的问题，检测到强光特征且四边形透视比例异常时，会自动用更严格的色相下界重检，避免垫子范围外扩。
- **MediaPipe 骨骼检测**：优先使用 PoseLandmarker Heavy（33 点骨骼，最多 5 人），失败时回退到 Legacy Pose。支持视频/摄像头输入，也支持单张图片输入用于调试。
- **起跳/落地检测**：通过脚尖、脚踝、脚后跟、髋部关键点在垫子坐标系中的变化判断 READY/JUMPING/LANDED 状态。
- **距离修正（可选）**：
  - `--yolo VERSION SCALE`：使用 YOLO 实例分割 person mask，在脚部 ROI 内提取脚尖/脚跟物理 X 极值修正距离。
  - `--diff`：使用 MOG2 背景差分作为备选修正方式。
  - 同时传入 `--yolo` 和 `--diff` 时，代码优先启用 YOLO，`--diff` 会被关闭。
- **犯规检测**：支持踩线、垫步/单脚异常、多人入界、出界、撑杆辅助等规则；可用 `--no-foul-detection` 关闭。
- **可视化输出**：保存起跳帧、落地帧、最终成绩图、犯规图、垫子 mask、测试网格、YOLO/MOG2 各阶段调试图和运行日志。
- **批量处理**：支持默认批量跑 `videos/跳远1-1.mp4` 到 `videos/跳远1-9.mp4`，或通过 `--videos` 指定列表，并生成汇总 CSV。

## 目录结构

```text
tiaoyuan/
├── main.py                         # CLI 入口；单视频/批量处理
├── README.md
├── src/
│   ├── config.py                   # JumpConfig、视频源解析、结果目录创建
│   ├── core/
│   │   └── jump_system.py          # 主状态机、成绩计算、结果保存
│   ├── inference/
│   │   ├── mat_calibration.py      # 垫子自动/手动标定与 mask 输出
│   │   ├── pose_estimator.py       # MediaPipe 骨骼检测封装
│   │   ├── diff_detector.py        # MOG2/YOLO 距离修正与阶段图
│   │   ├── shoe_detector.py        # 传统鞋边缘检测模块（当前主流程未作为核心修正方式）
│   │   └── shadow_remover.py       # HomoFormer 阴影移除封装（当前 CLI 未直接接入）
│   ├── rules/
│   │   └── foul_detection.py       # 犯规规则
│   └── visualization/
│       └── rendering.py            # 标注绘制、中文文字、图片保存
├── videos/                         # 本地视频目录（被 .gitignore 忽略）
├── yolo_model/                     # YOLO seg 模型目录（*.pt 被 .gitignore 忽略）
└── result/                         # 运行输出目录（被 .gitignore 忽略）
```

## 环境安装

推荐使用 Python 3.10+（当前开发环境使用 Python 3.12）。

```bash
conda create -n tiaoyuan python=3.12 -y
conda activate tiaoyuan

pip install opencv-python numpy mediapipe Pillow -i https://pypi.tuna.tsinghua.edu.cn/simple

# 如需 YOLO 实例分割修正
pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
```

首次运行 MediaPipe PoseLandmarker 时，程序会自动下载 `pose_landmarker_heavy.task` 到用户目录 `~/.mediapipe/`。如果自动下载失败，请手动下载该模型并放到对应目录。

YOLO 模型请放在项目根目录的 `yolo_model/` 下，文件名由版本和尺度决定：

| 命令 | 期望模型文件 |
|---|---|
| `--yolo 8 x` | `yolo_model/yolov8x-seg.pt` |
| `--yolo 11 x` | `yolo_model/yolo11x-seg.pt` |
| `--yolo 26 x` | `yolo_model/yolo26x-seg.pt` |

支持尺度：`n`、`s`、`m`、`l`、`x`。支持版本：`8`、`11`、`26`。

## 快速开始

### 单视频处理

```bash
# 骨骼关键点基础流程
python main.py --video videos/跳远1-1.mp4 --no-display

# 启用 YOLOv26x 实例分割距离修正
python main.py --video videos/跳远1-16.mp4 --yolo 26 x --no-display

# 启用 MOG2 差分修正
python main.py --video videos/跳远1-1.mp4 --diff --no-display

# 强光/HDR 场景：额外输出垫子 mask 与网格图，检查标定是否贴合垫子边界
python main.py --video videos/跳远1-16.mp4 --yolo 26 x --enable-mat-output --test-grid --no-display
```

### 批量处理

```bash
# 默认处理 videos/跳远1-1.mp4 ~ videos/跳远1-9.mp4
python main.py --batch --no-display

# 指定文件名（相对路径会自动从 videos/ 下查找）
python main.py --videos 跳远1-9-1080p.mp4 跳远1-14.mp4 跳远1-16.mp4 --yolo 26 x --enable-mat-output --test-grid --no-display

# 也可以传绝对路径
python main.py --videos D:/data/a.mp4 D:/data/b.mp4 --no-display
```

### 预览与录制

```bash
# 打开预览窗口，按 q 退出
python main.py --video videos/跳远1-1.mp4

# 保存带标注的视频
python main.py --video videos/跳远1-1.mp4 --record result/preview.mp4
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--video` | `0` | 视频文件路径或摄像头索引；不存在时会回退到摄像头 0 |
| `--save` | `result.json` | 兼容参数；当前通过 CLI 正常运行时结果会写到自动创建的结果目录中的 `result.json` |
| `--no-display` | 关闭 | 不显示 OpenCV 预览窗口；批量处理中单个视频固定不显示 |
| `--backend` | `mediapipe` | 骨骼检测后端参数，目前实际实现为 MediaPipe |
| `--debug-dir` | 空 | 兼容/预留参数，传入 `JumpConfig` |
| `--record` | 空 | 保存显示画面的 MP4 路径 |
| `--mat-length-cm` | `338.0` | 垫子长度，单位 cm |
| `--mat-width-cm` | `100.0` | 垫子宽度，单位 cm |
| `--trigger-move-cm` | `32.0` | 脚尖前移触发起跳的主要阈值，单位 cm |
| `--trigger-frames` | `2` | 起跳触发需要满足的帧数参数 |
| `--min-flight-frames` | `5` | 最短腾空/跳跃帧数约束 |
| `--max-jump-frames` | `120` | 最大跳跃帧数，超出后按超时落地处理 |
| `--takeoff-line-cm` | `31.0` | 起跳线在垫子坐标系中的 X 位置 |
| `--takeoff-offset-cm` | `3.0` | 起跳点显示/成绩计算偏移补偿 |
| `--manual-calib` | 关闭 | 手动点击 4 个角点进行垫子标定 |
| `--no-foul-detection` | 关闭 | 关闭犯规检测 |
| `--landing-offset-cm` | `-5.0` | 落地点补偿，默认负值用于缩短鞋跟厚度带来的偏差 |
| `--debug` | 关闭 | 输出更详细的 READY/JUMPING 触发日志 |
| `--diff` | 关闭 | 启用 MOG2 背景差分修正；若同时传 `--yolo`，则 YOLO 优先 |
| `--yolo VERSION SCALE` | 空 | 启用 YOLO 实例分割修正，如 `--yolo 26 x` |
| `--enable-mat-output` | 关闭 | 输出 `mat_mask_quad.jpeg` 和 `mat_mask_hsv.jpeg` |
| `--test-grid` | 关闭 | 输出 `test_grid.jpeg`，用于检查厘米坐标映射 |
| `--batch` | 关闭 | 批量处理默认视频 `跳远1-1` 到 `跳远1-9` |
| `--videos` | 空 | 批量处理指定视频列表 |

## 输出结果

每次运行都会创建带时间戳的结果目录：

```text
result/
└── <视频名>/
    └── <视频名>_<YYYYMMDD_HHMMSS>/
        ├── result.json
        ├── images/
        │   ├── takeoff.jpeg                 # 起跳帧标注图
        │   ├── landed.jpeg                  # 落地帧标注图
        │   ├── score.jpeg                   # 最终成绩图
        │   ├── foul-<timestamp>.jpeg        # 犯规截图（如触发）
        │   ├── mat_mask_quad.jpeg           # --enable-mat-output：最终实心四边形 mask
        │   ├── mat_mask_hsv.jpeg            # --enable-mat-output：四边形内可见颜色 mask
        │   ├── test_grid.jpeg               # --test-grid：垫子边框、起跳线、10cm 网格
        │   ├── yolo/                        # --yolo：YOLO 各阶段调试图
        │   └── diff/                        # --diff：MOG2 各阶段调试图
        └── logs/
            ├── run_<timestamp>.log          # 运行状态日志
            └── keypoints_<timestamp>.log    # 每帧关键点日志
```

批量模式还会生成：

```text
result/summary_<YYYYMMDD_HHMMSS>.csv
```

`result.json` 字段：

| 字段 | 含义 |
|---|---|
| `score` | 最终成绩，单位 cm，与 `distance_cm` 保持一致 |
| `valid` | 是否有效；存在犯规原因时为 `false` |
| `foul_reason` | 犯规原因；无犯规时为 `null` |
| `distance_cm` | 最终距离，单位 cm |
| `takeoff_x_cm` | 起跳点在垫子坐标系中的 X 坐标 |
| `landing_x_cm` | 落地点在垫子坐标系中的 X 坐标 |
| `yolo_infer_time_s` | YOLO 推理累计耗时；未启用 YOLO 时通常为 0 |

## 垫子标定逻辑

`src/inference/mat_calibration.py` 是垫子范围标定的核心模块。当前自动标定流程：

1. 对下半画面进行颜色提案，屏蔽画面上方 50% 以减少背景干扰。
2. HSV 中提取绿色区域，同时合并强光下低饱和高亮的垫子可见区域。
3. 使用开运算/闭运算清理噪点，并用较长的水平闭运算补齐边角、锯齿、反光、人体遮挡造成的断裂。
4. 从检测 mask 中找长条候选轮廓，按面积、长宽比、位置评分。
5. 对候选轮廓做凸包/四边形近似，必要时使用 `minAreaRect`。
6. 用 Canny + Hough 边缘线进一步细化四条边，再生成最终四边形。
7. 将最终四边形映射到 `mat_length_cm × mat_width_cm` 的物理坐标系。
8. 强光/HDR 帧中若出现“上边宽度接近下边宽度”的外扩形态，会用更严格 `hue_low=26` 重检，并在面积和透视比例满足约束时替换结果。

调试时建议同时打开：

```bash
python main.py --video videos/跳远1-16.mp4 --enable-mat-output --test-grid --no-display
```

检查重点：

- `mat_mask_quad.jpeg`：真正用于坐标标定的最终实心四边形。找最大矩形/坐标转换依赖的是这个最终 quad，而不是原始 HSV 可见 mask。
- `mat_mask_hsv.jpeg`：最终四边形内部的可见颜色区域，用于观察反光、缺角、低饱和区域是否被识别。
- `test_grid.jpeg`：最直观检查垫子边界、起跳线和 10cm 网格是否贴合真实垫子。

## 主流程说明

```text
读取视频/图片
  ↓
垫子自动或手动标定，建立 H_img2mat / H_mat2img
  ↓
MediaPipe 检测人体关键点
  ↓
IDLE：等待人体进入垫子并靠近起跳线
  ↓
READY：记录稳定站立基线，检测脚尖/髋部/脚踝突变
  ↓
JUMPING：确认起跳，保存起跳帧，等待脚后跟触底/超时
  ↓
落地后计算骨骼基础成绩
  ↓
可选 YOLO 或 MOG2 修正起跳/落地 X
  ↓
犯规判定、保存 result.json 和可视化图片
```

### 起跳判定（概要）

代码会综合以下信号，而不是只看单一阈值：

- 脚尖前移超过 `trigger_move_cm` 或保底阈值。
- 髋部明显前冲并伴随脚尖移动。
- 稳定站立帧数积累后突然失稳。
- 脚尖关键点连续丢失且位置变化符合离地过程。
- 脚踝抬升、脚尖回缩、髋部前移等离地爆发组合条件。

起跳点会尽量使用触发前的稳定基准帧，避免触发帧脚已经前移导致成绩偏短。

### 落地判定（概要）

落地主要通过脚后跟关键点的 Y 方向触底形态、连续帧确认和最大跳跃帧数兜底判断。最终距离计算会应用 `landing_offset_cm` 鞋跟补偿。

## YOLO 实例分割修正

启用 `--yolo VERSION SCALE` 后，`src/inference/diff_detector.py` 会加载 `yolo_model/` 下对应的 `*-seg.pt` 模型，并对起跳/落地帧进行 person mask 推理。核心思路：

```text
全图 YOLO 推理 → person mask → 脚部 ROI 裁剪 → 上部遮挡过滤/轮廓清理 → 像素投影到垫子坐标系 → 取物理 X 极值
```

输出目录 `images/yolo/` 中会包含：

- `yolo-Stage1-seg-takeoff.jpeg` / `yolo-Stage1-seg-landing.jpeg`
- `yolo-Stage2-roi-takeoff.jpeg` / `yolo-Stage2-roi-landing.jpeg`
- `yolo-Stage3-mask-takeoff.jpeg` / `yolo-Stage3-mask-landing.jpeg`
- `yolo-Stage4-takeoff.jpeg` / `yolo-Stage4-landing.jpeg` / `yolo-Stage4-combined.jpeg`

YOLO 修正值会覆盖骨骼基础取值；保存的标注图中会同时保留骨骼/修正线索，便于对比。

## MOG2 差分修正

启用 `--diff` 后，程序会在垫子标定完成且垫子内无人体关键点时捕获基准帧，然后对起跳/落地帧做背景差分和脚部 ROI 边缘提取。输出目录为 `images/diff/`，文件名前缀为 `diff-Stage*.jpeg`。

## 犯规规则

| 规则 | 当前检测方式概要 |
|---|---|
| 踩线 | 起跳点 X 超过起跳线容差 |
| 垫步/起跳前异常 | 起跳前脚尖历史位移过大 |
| 单脚起跳异常 | 双脚脚踝 X/Y 差异过大 |
| 多人入界 | 垫子范围内同时检测到多个人体关键点 |
| 出界 | 落地点 Y 超出垫子宽度范围 |
| 撑杆/手部辅助 | 手腕位置低于膝盖等异常姿态 |

如当前任务只需要测距、不需要规则判定，可使用：

```bash
python main.py --video videos/跳远1-1.mp4 --no-foul-detection --no-display
```

## 常见问题

### 1. HDR 原视频和 SDR 截图标定不一致

HDR 视频经 OpenCV 解码后可能更亮、更低饱和，导致普通 HSV 阈值把垫子外侧黄绿/灰白区域也合进来。当前代码已经加入强光/HDR 自适应；如果仍有偏差，请先输出 `test_grid.jpeg`，以最终网格是否贴合为准。

```bash
python main.py --video videos/跳远1-16.mp4 --enable-mat-output --test-grid --no-display
```

### 2. `mat_mask_hsv.jpeg` 边角缺失是否一定有问题？

不一定。`mat_mask_hsv.jpeg` 是可见颜色区域，边角受圆角、反光、锯齿、遮挡影响会有缺失；最终坐标标定看 `mat_mask_quad.jpeg` 和 `test_grid.jpeg`。

### 3. 找不到 YOLO 模型

确认模型文件位于项目根目录 `yolo_model/` 下，并与命令匹配。例如 `--yolo 26 x` 需要 `yolo_model/yolo26x-seg.pt`。模型文件体积较大，默认不会提交到 Git。

### 4. Windows 命令行中文乱码

代码和 README 均为 UTF-8。部分 Windows cmd 窗口显示 `main.py --help` 时可能乱码，但不影响日志文件和结果 JSON。可以尝试使用支持 UTF-8 的终端。

### 5. 结果目录或视频没有上传到仓库

`.gitignore` 会忽略 `result/`、`videos/`、`*.pt`、MediaPipe/HomoFormer 外部目录和备份目录，避免把大文件或本地依赖提交到 GitHub。

## 代码检视清单

本 README 对应的代码检视范围：

- `main.py`：CLI 参数、单视频/批量入口、结果目录与 CSV 汇总。
- `src/config.py`：运行配置、视频源解析、结果目录创建。
- `src/core/jump_system.py`：状态机、起跳/落地、YOLO/MOG2 修正接入、输出文件。
- `src/inference/mat_calibration.py`：垫子自动标定、HDR/强光适配、mask/grid 输出。
- `src/inference/pose_estimator.py`：MediaPipe 模型下载、视频/图片读取、多人关键点。
- `src/inference/diff_detector.py`：YOLO 模型路径、实例分割/MOG2 调试图和距离修正。
- `src/rules/foul_detection.py`：犯规规则。
- `src/visualization/rendering.py`：图像标注和中文路径安全保存。

## License

MIT

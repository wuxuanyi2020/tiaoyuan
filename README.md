# 立定跳远自动检测系统

基于 OpenCV + MediaPipe 骨骼关键点（Skeleton）的立定跳远成绩自动检测系统。支持视频输入，自动识别垫子区域、检测起跳/落地点、计算距离、判定犯规。

## 功能特性

- **自动标定**: 通过 HSV 颜色分割自动识别绿色跳远垫子，生成透视变换矩阵
- **骨骼关键点检测**: 基于 MediaPipe 33 点骨骼，通过脚尖位移、髋部移动和脚踝离地判定起跳，通过脚后跟 Y 坐标触底检测落地
- **差分法/实例分割距离修正** (可选): 支持两种修正模式——MOG2 背景建模 + 轮廓实心填充（`--diff`），或 YOLOv11-seg 实例分割（`--yolo`），从基准帧（无人体）与起跳/落地帧的脚部 ROI 精确提取鞋子边缘位置，减少不同鞋子尺寸带来的误差
- **犯规检测**: 踩线、垫步、单脚起跳、多人入界、出界、撑杆辅助
- **鞋子边缘修正**: Canny + ROI 局部检测，补偿骨骼关键点在鞋底位置上的偏差
- **批量处理**: 支持一次性跑多段视频并生成汇总 CSV
- **结果可视化**: 垫子标定图、犯规截图、落地标注图、差分过程图

## 环境要求

- Python 3.10+
- conda 环境（推荐）

## 安装

```bash
# 创建 conda 环境
conda create -n tiaoyuan python=3.12 -y
conda activate tiaoyuan

# 安装依赖（清华源）
pip install opencv-python numpy mediapipe Pillow -i https://pypi.tuna.tsinghua.edu.cn/simple
```

首次运行时会自动下载 MediaPipe 骨骼模型 (`pose_landmarker_heavy.task`，约 29MB)，缓存于 `~/.mediapipe/`。

## 快速开始

### 单视频处理

```bash
python main.py --video videos/跳远1-1.mp4 --no-display
```

### 批量处理

```bash
# 跑 videos/ 下跳远1-1 ~ 跳远1-9
python main.py --batch --no-display

# 跑指定视频
python main.py --videos 跳远1-1.mp4 跳远1-2.mp4 --no-display
```

### 预览模式

```bash
# 带窗口预览
python main.py --video videos/跳远1-1.mp4
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--video` | `0` | 视频路径或摄像头索引 |
| `--batch` | - | 批量处理 `videos/` 下跳远1-1 ~ 跳远1-9 |
| `--videos` | - | 批量处理指定视频列表 |
| `--no-display` | - | 不显示预览窗口 |
| `--mat-length-cm` | `340.0` | 垫子长度 (cm) |
| `--mat-width-cm` | `90.0` | 垫子宽度 (cm) |
| `--takeoff-line-cm` | `32.0` | 起跳线位置 (cm) |
| `--takeoff-offset-cm` | `0.0` | 起跳点偏移修正 (cm) |
| `--landing-offset-cm` | `-5.0` | 落地点修正 (cm，补偿鞋跟厚度) |
| `--manual-calib` | - | 手动四点标定（鼠标点击） |
| `--no-foul-detection` | - | 禁用犯规检测 |
| `--diff` | - | 启用 MOG2 背景差分法距离修正（默认关闭） |
| `--yolo` | - | 启用 YOLOv11-seg 实例分割距离修正（默认关闭） |
| `--enable-mat-output` | - | 输出垫子识别图 (mat_mask_quad/hsv) |
| `--debug` | - | 调试模式：每帧记录起跳/落地判定参数到日志 |
| `--record` | - | 输出视频录制路径 |

## 输出结构

```
result/
└── <视频名>/
    └── <视频名>_<时间戳>/
        ├── result.json                # 结构化结果
        ├── images/
        │   ├── mat_mask_quad.jpeg         # 垫子识别图（四边形拟合）
        │   ├── mat_mask_hsv.jpeg          # 垫子识别图（HSV 原始）
        │   ├── takeoff.jpeg               # 起跳帧标注图
        │   ├── landed.jpeg                # 落地帧标注图
        │   ├── foul-*.jpeg                # 犯规截图
        │   └── diff/
        │       ├── diff-Stage1-baseframe.jpeg    # 基准帧（无人体）
        │       ├── diff-Stage2-roi-takeoff.jpeg  # ROI 标注
        │       ├── diff-Stage3-edge-takeoff.jpeg # MOG2 前景/轮廓填充过程
        │       └── diff-Stage4-combined.jpeg     # 差分叠加结果
        └── logs/
            ├── run_<时间戳>.log         # 运行日志
            └── keypoints_<时间戳>.log   # 关键点帧数据
```

`result.json` 格式:

```json
{
  "score": 155.4,
  "valid": false,
  "foul_reason": "踩线 (Line Violation)",
  "distance_cm": 155.4,
  "takeoff_x_cm": 34.3,
  "landing_x_cm": 194.8
}
```

## 项目结构

```
tiaoyuan/
├── main.py                          # 主入口 & 批量处理
├── src/
│   ├── config.py                    # 配置数据结构 & 路径工具
│   ├── core/
│   │   └── jump_system.py           # 核心状态机 (IDLE→READY→JUMPING→LANDED)
│   ├── inference/
│   │   ├── mat_calibration.py       # MatCalibrator: 垫子标定 & 透视变换
│   │   ├── shoe_detector.py         # ShoeEdgeDetector: ROI 鞋子边缘检测
│   │   ├── diff_detector.py         # DiffDetector: 差分法距离修正
│   │   └── pose_estimator.py        # PoseEstimator: MediaPipe 姿态推理
│   ├── rules/
│   │   └── foul_detection.py        # FoulDetector: 犯规规则引擎
│   └── visualization/
│       └── rendering.py             # Renderer: 可视化绘制工具
└── videos/                          # 视频文件（不纳入版本管理）
```

## 检测原理

### Skeleton 骨骼关键点法

1. **垫子标定**: HSV 颜色分割自动识别绿色垫子，通过轮廓拟合四边形并计算透视变换矩阵
2. **人体检测**: 通过 MediaPipe PoseLandmarker 获取每帧 33 个骨骼关键点坐标（含脚趾、脚踝、脚后跟）
3. **站立稳定期**: 稳定站立阶段记录脚尖基线 X 和脚踝 Y 基线（EMA 平滑更新），稳定帧数积累
4. **起跳判定**（满足任一）:
   - 脚尖前移 > max(trigger_move_cm, 30.0)
   - 髋部前移 > 35cm 且脚尖前移 > 10cm
   - 脚踝离地 > 20px 且脚尖前移 > 3cm 且稳定帧数归零
   - 稳定期 > 35 帧后突然置零
   - 脚尖关键点连续缺失 ≥ 3 帧且脚尖前移 > −3cm
5. **起跳点取值**: 使用触发前一帧的数据倒推，避免触发帧脚已离地前移导致的误差
6. **落地检测**: 脚后跟 Y 坐标触底（V 型谷底模式）+ 连续帧阈值确认
7. **成绩计算**: landing_x − takeoff_x + landing_offset (−5cm 补偿鞋后跟厚度)

### DiffDetector 差分法距离修正（可选，默认关闭）

差分法作为骨骼法的补充测量，利用无人体时的基准帧与起跳/落地帧的对比，提取鞋子精确位置，不干预骨骼法的起跳/落地判断逻辑。支持两种模式：

**MOG2 背景差分** (`--diff`):
```
流程: 基准帧 + 起跳帧 → MOG2 前景 → 二值化 → 轮廓填充 → 脚尖 X
      基准帧 + 落地帧 → MOG2 前景 → 二值化 → 轮廓填充 → 脚后跟 X
```

**YOLOv11-seg 实例分割** (`--yolo`):
```
流程: 全图 YOLO 推理 → person 二值 Mask → 脚部关键点 ROI 裁剪 → 高度过滤(前40%置零) → 脚尖/脚跟 X
```

运行示例：
```bash
python main.py --yolo --video videos/跳远1-1.mp4 --no-display
python main.py --diff --video videos/跳远1-1.mp4 --no-display
```

## 犯规规则

| 规则 | 检测方式 |
|---|---|
| 踩线 | 起跳点 X > 起跳线 + 1cm |
| 垫步 | 起跳前脚尖前移 > 10cm / 双脚 X/Y 差 > 阈值 |
| 单脚起跳 | 双脚踝 X/Y 差异过大 |
| 多人入界 | 垫子内同时检测到 ≥ 2 人 |
| 出界 | 落地点 Y 超出垫子宽度 |
| 撑杆辅助 | 手腕 Y 低于膝盖 Y |

## License

MIT

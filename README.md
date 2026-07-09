# 立定跳远自动检测系统

基于 OpenCV + MediaPipe 的立定跳远成绩自动检测系统。支持视频输入，自动识别垫子区域、检测起跳/落地点、计算距离、判定犯规。

## 功能特性

- **自动标定**: 通过 HSV 颜色分割自动识别绿色跳远垫子，生成透视变换矩阵
- **双模式检测**:
  - `contour` (差分法) — 基于垫子色反色提取人体轮廓，通过轮廓面积骤降判定起跳
  - `skeleton` (骨骼关键点) — 基于 MediaPipe 33 点骨骼，通过脚尖位移和脚踝离地判定起跳
- **犯规检测**: 踩线、垫步、单脚起跳、多人入界、出界、撑杆辅助
- **鞋子边缘修正**: Canny + ROI 局部检测，补偿骨骼关键点在鞋底位置上的偏差
- **批量处理**: 支持一次性跑多段视频并生成汇总 CSV
- **结果可视化**: 垫子标定图、差分热力图、犯规截图、落地标注图

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
# 差分法（默认）
python main.py --video videos/跳远1-1.mp4 --no-display

# 骨骼关键点法
python main.py --video videos/跳远1-1.mp4 --no-display --detection-method skeleton
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
| `--detection-method` | `contour` | 检测方式: `contour` / `skeleton` |
| `--mat-length-cm` | `340.0` | 垫子长度 (cm) |
| `--mat-width-cm` | `90.0` | 垫子宽度 (cm) |
| `--takeoff-line-cm` | `32.0` | 起跳线位置 (cm) |
| `--takeoff-offset-cm` | `0.0` | 起跳点偏移修正 (cm) |
| `--landing-offset-cm` | `-5.0` | 落地点修正 (cm，补偿鞋跟厚度) |
| `--manual-calib` | - | 手动四点标定（鼠标点击） |
| `--no-foul-detection` | - | 禁用犯规检测 |
| `--debug` | - | 调试模式：每帧记录起跳/落地判定参数到日志 |
| `--record` | - | 输出视频录制路径 |

## 输出结构

```
result/
└── <视频名>/
    └── <视频名>_<时间戳>/
        ├── result.json          # 结构化结果
        ├── images/
        │   ├── mat_mask_quad.jpeg   # 垫子识别图（四边形拟合）
        │   ├── mat_mask_hsv.jpeg    # 垫子识别图（HSV 原始）
        │   ├── diff-takeoff-*.jpeg  # 起跳差分热力图
        │   ├── diff-landing-*.jpeg  # 落地差分热力图
        │   ├── foul-*.jpeg          # 犯规截图
        │   └── landed-*.jpeg        # 落地标注图
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
│   │   ├── contour_detector.py      # ContourDetector: 差分 & 人体轮廓
│   │   ├── shoe_detector.py         # ShoeEdgeDetector: ROI 鞋子边缘检测
│   │   └── pose_estimator.py        # PoseEstimator: MediaPipe 姿态推理
│   ├── rules/
│   │   └── foul_detection.py        # FoulDetector: 犯规规则引擎
│   └── visualization/
│       └── rendering.py             # Renderer: 可视化绘制工具
└── videos/                          # 视频文件（不纳入版本管理）
```

## 检测原理

### Contour 模式 (差分法)

1. 标定阶段记录干净垫子作为基线帧
2. 通过垫子绿色反色提取垫上人体 mask
3. 当人体轮廓面积骤降至基线 30% 以下时，判定起跳
4. 取前缘 X 坐标历史均值作为起跳点
5. 结合 MediaPipe 鞋子边缘检测获取精确落地点

### Skeleton 模式 (骨骼关键点)

1. 通过 MediaPipe 获取 33 点骨骼坐标
2. 稳定站立阶段记录脚尖基线 X 和脚踝 Y 基线（EMA 平滑更新）
3. 稳定期 > 10 帧后检测脚尖前移 / 髋部移动 / 脚踝离地 → 起跳
4. 稳定期 > 35 帧后突然置 0 也视为起跳触发
5. 起跳计数器 >= 1 即确认起跳
6. 检测脚后跟 Y 坐标触底（V 型谷底）→ 落地

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

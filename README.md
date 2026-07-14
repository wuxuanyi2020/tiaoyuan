# 立定跳远自动检测系统

基于 OpenCV + MediaPipe 骨骼关键点（Skeleton）的立定跳远成绩自动检测系统。支持视频输入，自动识别垫子区域、检测起跳/落地点、计算距离、判定犯规。可选 YOLO 实例分割距离修正，支持 YOLOv8/v11/v26 多种尺度的模型。

## 功能特性

- **自动标定**: 通过 HSV 颜色分割自动识别绿色跳远垫子，生成透视变换矩阵；支持手动四点标定
- **骨骼关键点检测**: 基于 MediaPipe 33 点骨骼，通过脚尖位移、髋部移动和脚踝离地判定起跳，通过脚后跟 Y 坐标触底检测落地
- **YOLO 实例分割距离修正** (可选): 支持 YOLOv8/v11/v26 多种尺度（n/s/m/l/x）的实例分割模型，从基准帧与起跳/落地帧的脚部 ROI 精确提取鞋子边缘位置，减少不同鞋子尺寸带来的误差
- **MOG2 差分法距离修正** (可选): 基于背景建模 + 轮廓实心填充的备选修正方案（`--diff`）
- **智能起跳判定**: 综合脚尖位移、髋部前移、脚踝抬升、关键点丢失、稳定期突变等多维度判定，区分真实起跳与蓄力动作
- **犯规检测**: 踩线、垫步、单脚起跳、多人入界、出界、撑杆辅助
- **鞋子边缘修正**: Canny + ROI 局部检测，补偿骨骼关键点在鞋底位置上的偏差
- **批量处理**: 支持一次性跑多段视频并生成汇总 CSV
- **结果可视化**: 垫子标定图、犯规截图、落地标注图、YOLO/差分过程图

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

# 如需 YOLO 实例分割功能
pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
```

首次运行时会自动下载 MediaPipe 骨骼模型 (`pose_landmarker_heavy.task`，约 29MB)，缓存于 `~/.mediapipe/`。

YOLO 模型（如 `yolo11x-seg.pt`）需下载后放入项目根目录的 `yolo_model/` 文件夹。

## 快速开始

### 单视频处理

```bash
# 纯骨骼关键点法
python main.py --video videos/跳远1-1.mp4 --no-display

# 启用 YOLOv11x-seg 距离修正
python main.py --yolo 11 x --video videos/跳远1-1.mp4 --no-display

# 启用 MOG2 差分法距离修正
python main.py --diff --video videos/跳远1-1.mp4 --no-display
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
# 带窗口预览（按 q 退出）
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
| `--trigger-move-cm` | `31.0` | 脚尖前移触发起跳阈值 (cm) |
| `--landing-offset-cm` | `-5.0` | 落地点修正 (cm，补偿鞋跟厚度) |
| `--manual-calib` | - | 手动四点标定（鼠标点击） |
| `--no-foul-detection` | - | 禁用犯规检测 |
| `--diff` | - | 启用 MOG2 背景差分法距离修正（默认关闭） |
| `--yolo VERSION SCALE` | - | 启用 YOLO 实例分割距离修正，指定版本和尺度，如 `--yolo 26 x`（版本: 8/11/26, 尺度: n/s/m/l/x） |
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
        │   ├── diff/                       # MOG2 差分过程图
        │   │   ├── diff-Stage1-baseframe.jpeg
        │   │   ├── diff-Stage2-roi-*.jpeg
        │   │   ├── diff-Stage3-edge-*.jpeg
        │   │   └── diff-Stage4-*.jpeg
        │   └── yolo/                        # YOLO seg 过程图
        │       ├── yolo-Stage1-seg-*.jpeg        # 起跳/落地帧人体分割覆盖图
        │       ├── yolo-Stage2-roi-*.jpeg        # ROI 标注
        │       ├── yolo-Stage3-mask-*.jpeg       # YOLO Mask 切片
        │       └── yolo-Stage4-*.jpeg            # 叠加结果
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
├── yolo_model/                      # YOLO 模型文件（*.pt，不纳入版本管理）
├── src/
│   ├── config.py                    # 配置数据结构 & 路径工具
│   ├── core/
│   │   └── jump_system.py           # 核心状态机 (IDLE→READY→JUMPING→LANDED)
│   ├── inference/
│   │   ├── mat_calibration.py       # MatCalibrator: 垫子标定 & 透视变换
│   │   ├── shoe_detector.py         # ShoeEdgeDetector: ROI 鞋子边缘检测
│   │   ├── diff_detector.py         # DiffDetector: 差分法/YOLO 距离修正
│   │   └── pose_estimator.py        # PoseEstimator: MediaPipe 姿态推理
│   ├── rules/
│   │   └── foul_detection.py        # FoulDetector: 犯规规则引擎
│   └── visualization/
│       └── rendering.py             # Renderer: 可视化绘制工具
└── videos/                          # 视频文件（不纳入版本管理）
```

## 检测原理

### Skeleton 骨骼关键点法

1. **垫子标定**: HSV 颜色分割自动识别绿色垫子，通过轮廓拟合四边形并计算透视变换矩阵。支持画面顶部 40% 区域屏蔽以滤除天空/树木等绿色噪声，形态学操作使用 9x9 开运算 + 5x5 闭运算去除毛刺
2. **人体检测**: 通过 MediaPipe PoseLandmarker 获取每帧 33 个骨骼关键点坐标（含脚趾、脚踝、脚后跟）
3. **站立稳定期**: 稳定站立阶段（脚尖位移 < 6cm）记录脚尖基线 X 和脚踝 Y 基线（EMA 指数平滑更新），稳定帧数积累
4. **起跳判定**（满足任一）:
   - **脚尖大幅前移**: `toe_moved > max(trigger_move_cm, 30.0)`
   - **重心前冲**: `hip_moved > 35cm 且 toe_moved > 10cm`
   - **稳定期突变**: `stable_before > 35` 后稳定帧数突然归零
   - **脚尖丢失**: 连续 3 帧以上脚尖关键点缺失且位置不移后
   - **离地爆发**(动态复合判定): 脚踝抬升 + 脚尖前移 > 3cm，或脚踝抬升 + 髋部前移 > 70cm + 脚尖因腾空后摆出现回缩 (< -2cm)
5. **起跳点取值**: 使用触发前一帧的数据倒推，避免触发帧脚已离地前移导致的误差
6. **落地检测**: 脚后跟 Y 坐标触底（V 型谷底模式）+ 连续帧阈值确认
7. **成绩计算**: `landing_x − takeoff_x + landing_offset`（默认 −5cm 补偿鞋后跟厚度）

### YOLO 实例分割距离修正（可选）

通过全图 YOLO 推理获取 person 二值 Mask，结合脚部关键点 ROI 裁剪，精确提取鞋子边缘位置：

```
流程: 全图 YOLO 推理 → person 二值 Mask → 脚部关键点 ROI 裁剪 → 脚尖/脚跟 X 提取
```

支持的模型组合：

| 版本 | 可选尺度 | 示例模型 |
|------|---------|---------|
| YOLOv8 | n/s/m/l/x | yolov8x-seg.pt |
| YOLOv11 | n/s/m/l/x | yolo11x-seg.pt |
| YOLOv26 | n/s/m/l/x | yolo26x-seg.pt |

运行示例：
```bash
# YOLOv11x-seg（默认推荐）
python main.py --yolo 11 x --video videos/跳远1-1.mp4 --no-display

# YOLOv26n-seg（轻量快速）
python main.py --yolo 26 n --video videos/跳远1-1.mp4 --no-display

# YOLOv8m-seg（平衡模式）
python main.py --yolo 8 m --video videos/跳远1-1.mp4 --no-display
```

### DiffDetector 差分法距离修正（可选）

利用无人体时的基准帧与起跳/落地帧的对比，提取鞋子精确位置：

```
流程: 基准帧 → MOG2 前景 → 二值化 → 轮廓填充 → 脚尖/脚后跟 X
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

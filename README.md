# 立定跳远自动检测系统

基于 OpenCV + MediaPipe 骨骼关键点（Skeleton）的立定跳远成绩自动检测系统。支持视频输入，自动识别垫子区域、检测起跳/落地点、计算距离、判定犯规。可选 YOLO 实例分割或 MOG2 差分法距离修正。

## 功能特性

- **自动标定**: 通过 HSV 颜色分割自动识别绿色跳远垫子，生成透视变换矩阵；支持手动四点标定
- **骨骼关键点检测**: 基于 MediaPipe 33 点骨骼，通过脚尖位移、髋部移动和脚踝离地判定起跳，通过脚后跟 Y 坐标触底检测落地
- **YOLO 实例分割距离修正** (可选): 支持 YOLOv8/v11/v26 多种尺度（n/s/m/l/x）的实例分割模型，从基准帧与起跳/落地帧的脚部 ROI 精确提取鞋子边缘位置。Stage3 输出三列可视化：Raw_ROI（原图ROI）、Mask_Overlay（原图+半透明绿色Mask叠加）、FinalMask（上部20%切割二值图，标注检测到的脚尖/脚跟位置）
- **MOG2 差分法距离修正** (可选): 基于背景建模 + 轮廓实心填充的备选修正方案（`--diff`）
- **ROI 分辨率自适应**: ROI 尺寸基于图像短边百分比计算，1080p 与 4K 视频的 ROI 覆盖物理区域一致，无需额外调参
- **智能起跳判定**: 综合脚尖位移、髋部前移、脚踝抬升、关键点丢失、稳定期突变等多维度判定，区分真实起跳与蓄力动作
- **犯规检测**: 踩线、垫步、单脚起跳、多人入界、出界、撑杆辅助
- **结果可视化**: 起跳帧/落地帧/成绩三张标注图、YOLO/差分过程图、垫子毫米格测试图
- **批量处理**: 支持一次性跑多段视频并生成汇总 CSV

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
# 纯骨骼关键点法（无距离修正）
python main.py --video videos/跳远1-1.mp4 --no-display

# 启用 YOLOv11x-seg 距离修正（与 --diff 互斥）
python main.py --yolo 11 x --video videos/跳远1-1.mp4 --no-display

# 启用 MOG2 差分法距离修正（与 --yolo 互斥）
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
| `--mat-length-cm` | `338.0` | 垫子长度 (cm) |
| `--mat-width-cm` | `100.0` | 垫子宽度 (cm) |
| `--takeoff-line-cm` | `31.0` | 起跳线位置 (cm) |
| `--takeoff-offset-cm` | `3.0` | 起跳点偏移修正 (cm) |
| `--trigger-move-cm` | `30.0` | 脚尖前移触发起跳阈值 (cm) |
| `--landing-offset-cm` | `-5.0` | 落地点修正 (cm，补偿鞋跟厚度) |
| `--manual-calib` | - | 手动四点标定（鼠标点击） |
| `--no-foul-detection` | - | 禁用犯规检测 |
| `--diff` | - | 启用 MOG2 背景差分法距离修正（与 `--yolo` 互斥） |
| `--yolo VERSION SCALE` | - | 启用 YOLO 实例分割距离修正（与 `--diff` 互斥），如 `--yolo 26 x`。YOLO 起跳点更靠近垫子边界（更保守）时自动覆盖骨骼修正值 |
| `--enable-mat-output` | - | 输出垫子识别图 (mat_mask_quad/hsv) |
| `--test-grid` | - | 输出垫子毫米格测试图（起跳线外每 10cm 画一条绿线） |
| `--debug` | - | 调试模式：每帧记录 IDLE/READY/JUMPING 各状态的全量判定参数到运行日志 |

## 输出结构

```
result/
└── <视频名>/
    └── <视频名>_<时间戳>/
        ├── result.json                # 结构化结果
        ├── images/
        │   ├── mat_mask_quad.jpeg         # 垫子识别图（四边形拟合）
        │   ├── mat_mask_hsv.jpeg          # 垫子识别图（HSV 原始）
        │   ├── test_grid.jpeg             # 垫子毫米格测试图
        │   ├── takeoff.jpeg               # 起跳帧标注图（limit + fixed corrected + yolo corrected）
        │   ├── landed.jpeg                # 落地帧标注图（limit + fixed corrected + yolo corrected）
        │   ├── score.jpeg                 # 成绩汇总图（起跳线+落地线+测量线）
        │   ├── foul-*.jpeg                # 犯规截图
        │   ├── diff/                       # MOG2 差分过程图（需 --diff）
        │   └── yolo/                        # YOLO seg 过程图（需 --yolo）
        └── logs/
            ├── run_<时间戳>.log         # 运行日志
            └── keypoints_<时间戳>.log   # 关键点帧数据
```

输出图像标注说明：

| 图像 | 内容 |
|------|------|
| `takeoff.jpeg` | `limit` 白色标准起跳线 + `fixed corrected` 黄色骨骼修正起跳点/线 + `yolo corrected` 绿色 YOLO 修正起跳点/线（`--yolo` 时） |
| `landed.jpeg` | `limit` 白色标准起跳线 + `fixed corrected` 红色骨骼修正落地点/线 + `yolo corrected` 绿色 YOLO 修正落地点/线（`--yolo` 时） |
| `score.jpeg` | 所有线（limit 白、fixed corrected 黄/红、yolo corrected 绿）+ 绿色测量连线 |

`result.json` 格式:

```json
{
  "score": 173.1,
  "valid": true,
  "foul_reason": null,
  "distance_cm": 173.1,
  "takeoff_x_cm": 30.8,
  "landing_x_cm": 203.9
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
2. **人体检测**: 通过 MediaPipe PoseLandmarker Heavy 模型（优先加载，自动从 CDN 下载 `pose_landmarker_heavy.task`，本地安装于 `mediapipe-0.10.35/` 目录；失败时回退到 Legacy Pose）获取每帧 33 个骨骼关键点坐标。支持多人检测（最多 5 人）。关键点索引：脚踝(27,28)、脚后跟(29,30)、脚尖(31,32)
3. **两阶段入垫检测**:
   - 阶段一：脚尖进入垫子范围（`in_mat()`，垫子内 `0 ≤ x ≤ mat_length`，`0 ≤ y ≤ mat_width`）→ 输出 "检测到人体在垫内" 日志（仅一次）
   - 阶段二：**双脚**脚尖均距起跳线 ≤ 5cm → 输出 "检测预备起跳" 日志，切换至 READY 状态
4. **站立稳定期**: 稳定站立阶段（脚尖位移 < 6cm）记录脚尖基线 X 和脚踝 Y 基线（EMA 指数平滑更新），稳定帧数积累
5. **起跳判定**（满足任一）:
   - **脚尖大幅前移**: `toe_moved > max(trigger_move_cm, 30.0)`
   - **重心前冲**: `hip_moved > 35cm 且 toe_moved > 10cm`
   - **稳定期突变**: `stable_before > 35` 后稳定帧数突然归零
   - **脚尖丢失**: 连续 3 帧以上脚尖关键点缺失且位置不移后
   - **离地爆发**(动态复合判定): 脚踝抬升 + 脚尖前移 > 3cm，或脚踝抬升 + 髋部前移 > 70cm + 脚尖因腾空后摆出现回缩 (< -2cm)
6. **起跳点取值**: 使用触发前一帧的数据倒推，避免触发帧脚已离地前移导致的误差；取基准帧的脚尖 X（垫子坐标 cm）加上 `takeoff_offset_cm` 作为最终起跳点 (`takeoff_x_cm = takeoff_x + takeoff_display_offset_cm`)。在启用 `--yolo` 时，保存骨骼修正值备份至 `_skeleton_takeoff_x_cm`（仅用于图像标注），最终起跳点 `takeoff_x_cm` 被 YOLO 分割结果覆盖
7. **落地检测**: 脚后跟 Y 坐标触底（V 型谷底模式）+ 连续帧阈值确认
8. **成绩计算**: `final_distance = max(0, (landing_x_for_dist + landing_offset_cm) - (takeoff_x + takeoff_display_offset_cm))`。其中 `landing_offset_cm`（默认 -5cm）为鞋后跟厚度补偿。启用 `--diff` 时起跳/落地点改用 MOG2 差分值；启用 `--yolo` 时起跳/落地点以 YOLO 实例分割结果为准

### YOLO 实例分割距离修正（可选）

通过全图 YOLO 推理获取 person 二值 Mask，结合脚部关键点 ROI 裁剪，精确提取鞋子边缘位置：

```
流程: 全图 YOLO 推理 → person 二值 Mask → 脚部关键点 ROI 裁剪 → 脚尖/脚跟 X 提取
```

YOLO 结果与骨骼关键点修正值自动对比，取更保守值：
- **起跳**：YOLO 脚尖 X 更靠近 0（垫子边界）→ 采用 YOLO 值作为最终起跳点
- **落地**：YOLO 脚跟 X 更靠近 `mat_length`（垫子末端）→ 采用 YOLO 值作为最终落地点
- 输出图像中分别用绿色 `yolo corrected` 和黄色/红色 `fixed corrected` 标注两条线

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

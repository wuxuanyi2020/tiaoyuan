"""运行配置模块：参数数据结构、视频源解析。"""
from dataclasses import dataclass, field
import os
from datetime import datetime


@dataclass
class JumpConfig:
    video_source: object = 0
    display: bool = True
    save_path: str = "result.json"
    debug_dir: str | None = None
    mat_length_cm: float = 340.0
    mat_width_cm: float = 100.0
    trigger_move_cm: float = 30.0
    trigger_frames: int = 2
    takeoff_backtrack_frames: int = 2  # 起跳触发帧倒推 N 帧作为正式起跳帧
    min_flight_frames: int = 10  # 起跳触发帧到最早落地候选的最小间隔；实际兜底落地需再晚 1 帧确认
    max_jump_frames: int = 30    # 30fps 下约 1s；超时仍无可靠后跟触地则不强行给成绩
    backend: str = "mediapipe"
    record_path: str | None = None
    takeoff_line_cm: float = 30.0
    jump_direction: str = "ltr"  # 跳跃方向: ltr=画面左到右(默认), rtl=画面右到左；rtl 时起跳线自动放在垫子右侧
    takeoff_offset_cm: float = 0.0
    manual_calib: bool = False
    result_dir: str | None = None  # 结果输出根目录，由 main 在运行时传入
    enable_foul_detection: bool = True
    landing_offset_cm: float = -5.0  # 落地点修正（鞋跟厚度补偿）
    enable_diff: bool = False  # 启用差分法距离修正（默认关闭）
    enable_mat_output: bool = False  # 输出垫子识别图 (mat_mask_quad/hsv)
    enable_test_grid: bool = False  # 输出垫子毫米格测试图
    enable_seg: bool = False  # 是否启用 YOLOv11-seg 实例分割替代背景差分
    yolo_version: str = "11"  # YOLO 版本号: "8", "11", "26" 等
    yolo_scale: str = "x"     # YOLO 模型尺度: "n", "s", "m", "l", "x"
    debug: bool = False  # 调试模式：起跳/落地时输出触发条件到日志
    stream_mode: bool = False  # 模拟流模式：一跳完成后不中断，等待垫内无人后可重新标定并继续下一跳
    stream_recalib_empty_frames: int = 15  # 垫内无人连续 N 帧后重新标定并锁定
    enable_ocr_time: bool = False  # 识别左上角模拟时间码
    ocr_roi: tuple[int, int, int, int] = (15, 18, 155, 30)  # 左上角时间 OCR 区域 x,y,w,h


def resolve_video_source(value):
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    source = str(value).strip().strip('"').strip("'")
    if not source:
        return 0
    if source.isdigit():
        return int(source)
    if os.path.exists(source):
        return source
    print(f"未找到视频文件: {source}，将使用摄像头 0")
    return 0


def make_result_dir(video_name: str) -> str:
    """创建带时间标签的结果文件夹（result/<视频名>/<视频名>_<时间戳>/）。"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join("result", video_name, f"{video_name}_{ts}")
    os.makedirs(base, exist_ok=True)
    os.makedirs(os.path.join(base, "images"), exist_ok=True)
    os.makedirs(os.path.join(base, "logs"), exist_ok=True)
    print(f">>> 结果目录: {base}")
    return base

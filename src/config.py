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
    mat_width_cm: float = 90.0
    trigger_move_cm: float = 30.0
    trigger_frames: int = 2
    min_flight_frames: int = 5
    max_jump_frames: int = 120
    backend: str = "mediapipe"
    record_path: str | None = None
    takeoff_line_cm: float = 32.0
    takeoff_offset_cm: float = 0.0
    manual_calib: bool = False
    result_dir: str | None = None  # 结果输出根目录，由 main 在运行时传入
    enable_foul_detection: bool = True
    landing_offset_cm: float = -5.0  # 落地点修正（鞋跟厚度补偿）
    detection_method: str = "contour"  # 检测方式: "contour"(差分) / "skeleton"(骨骼关键点)
    debug: bool = False  # 调试模式：起跳/落地时输出触发条件到日志


def resolve_video_source(value):
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    source = str(value).strip()
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

"""立定跳远核心系统：状态机、起跳判定、落地检测、日志记录。"""
import logging
import os
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from src.config import JumpConfig
from src.rules.foul_detection import FoulDetector
from src.inference.mat_calibration import MatCalibrator
from src.inference.shoe_detector import ShoeEdgeDetector
from src.inference.diff_detector import DiffDetector
from src.inference.pose_estimator import PoseEstimator
from src.inference.time_ocr import TimeOCR
from src.visualization.rendering import Renderer, imwrite_safe


class StandingLongJumpSystem:
    def __init__(self, config: JumpConfig):
        self.config = config
        self.jump_direction = self._normalize_jump_direction(getattr(config, "jump_direction", "ltr"))
        self.is_rtl = (self.jump_direction == "rtl")
        self.config.jump_direction = self.jump_direction
        self.pose_estimator = PoseEstimator(config.video_source, backend=config.backend)
        self.calibrator = MatCalibrator(
            mat_length_cm=config.mat_length_cm,
            mat_width_cm=config.mat_width_cm,
            manual_mode=config.manual_calib,
        )
        self.shoe_detector = ShoeEdgeDetector(self.calibrator)
        self.diff_detector = DiffDetector(self.calibrator, enable_seg=config.enable_seg,
                                          yolo_version=config.yolo_version,
                                          yolo_scale=config.yolo_scale,
                                          jump_direction=self.jump_direction)
        self.kpt_idx = {
            "l_hip": 23, "r_hip": 24,
            "l_ankle": 27, "r_ankle": 28,
            "l_heel": 29, "r_heel": 30,
            "l_big_toe": 31, "r_big_toe": 32,
            "l_wrist": 15, "r_wrist": 16,
            "l_knee": 25, "r_knee": 26,
        }
        self.renderer = Renderer(self.kpt_idx)
        self.foul_detector = FoulDetector(
            calibrator=self.calibrator,
            kpt_idx=self.kpt_idx,
            get_kpt=self._get_kpt,
            get_feet=self._get_feet,
            transform_to_mat_cm=self.calibrator.transform_to_mat_cm,
            enabled=config.enable_foul_detection,
        )

        # --- 结果目录 & 日志 ---
        self.result_dir = config.result_dir
        self.images_dir = os.path.join(self.result_dir, "images") if self.result_dir else None
        self.images_diff_dir = os.path.join(self.result_dir, "images", "diff") if self.result_dir else None
        self.images_yolo_dir = os.path.join(self.result_dir, "images", "yolo") if self.result_dir else None
        self.logs_dir = os.path.join(self.result_dir, "logs") if self.result_dir else None

        # 运行日志
        self.run_logger = logging.getLogger("run")
        self.run_logger.setLevel(logging.INFO)
        self.run_logger.handlers.clear()
        if self.logs_dir:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_fh = logging.FileHandler(os.path.join(self.logs_dir, f"run_{ts}.log"), encoding="utf-8")
            run_fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
            self.run_logger.addHandler(run_fh)

        # 关键点日志文件句柄
        self._kpts_log_fh = None
        if self.logs_dir:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._kpts_log_fh = open(os.path.join(self.logs_dir, f"keypoints_{ts}.log"), "w", encoding="utf-8")

        self._log("INIT", f"系统初始化完成, video_source={config.video_source}, 检测方式=骨骼关键点(skeleton), jump_direction={self.jump_direction}")

        # ── 调试模式 ──
        self.debug = config.debug
        if self.debug:
            self._log("DEBUG", "调试模式已开启，触发条件将记录到日志")

        # --- 状态变量 ---
        self.state = "IDLE"
        self.takeoff_pt_px = None
        self.landing_pt_px = None
        self.takeoff_pt_xy = None
        self.landing_pt_xy = None
        self.takeoff_x_cm = None
        self.landing_x_cm = None
        self.final_distance_cm = None
        self.takeoff_frame = None
        self.takeoff_trigger_frame = None  # 脚尖离地触发帧；正式起跳帧=触发帧倒推N帧
        self.landing_frame = None

        # ── 骨骼关键点法状态变量 ──
        self._skeleton_baseline_x_cm = None
        self._skeleton_baseline_hip_x = None
        self._skeleton_baseline_ankle_y = None
        self._skeleton_in_mat_logged = False  # 是否已输出"人体在垫内"日志
        self._skeleton_jump_trigger_counter = 0
        self._skeleton_pending_takeoff = None  # 延迟确认的起跳触发，避免把预抬脚/蓄力帧当触发帧
        self._skeleton_ready_stable = 0
        self._skeleton_front_toe_hist = deque(maxlen=30)
        self._skeleton_takeoff_candidate_hist = deque(maxlen=30)
        self._skeleton_toe_y_px_hist = deque(maxlen=12)
        self._skeleton_toe_missing_counter = 0
        self._skeleton_ready_invalid_counter = 0
        self._skeleton_last_hip_moved_cm = 0.0
        self._skeleton_jump_counter = 0
        self._skeleton_prev_takeoff_data = None  # 前一帧的起跳相关数据，用于倒推起跳点
        self._skeleton_foot_hist = {
            "l": {"y_px": deque(maxlen=3), "x_cm": deque(maxlen=3), "y_cm": deque(maxlen=3), "xy_px": deque(maxlen=3),
                  "frame_idx": deque(maxlen=3), "frame_img": deque(maxlen=3), "kpts": deque(maxlen=3), "all_kpts_list": deque(maxlen=3)},
            "r": {"y_px": deque(maxlen=3), "x_cm": deque(maxlen=3), "y_cm": deque(maxlen=3), "xy_px": deque(maxlen=3),
                  "frame_idx": deque(maxlen=3), "frame_img": deque(maxlen=3), "kpts": deque(maxlen=3), "all_kpts_list": deque(maxlen=3)},
        }

        self._last_kpts = None  # 上一帧骨架数据（用于犯规检测）
        self._takeoff_frame_img = None
        self._landing_frame_img = None
        self._landing_kpts = None
        self._landing_all_kpts_list = []
        self._prev_frame_img = None
        self._takeoff_saved = False
        self._landed_saved = False
        self._score_saved = False
        self._foul_saved = False
        self._debug_takeoff_toe_moved_saved = False
        self.record_writer = None

        # ── 差分法状态变量 ──
        self._diff_computed = False

        self.takeoff_display_offset_cm = float(config.takeoff_offset_cm)
        self.landing_offset_cm = float(config.landing_offset_cm)

        # 骨骼修正值（YOLO 覆盖前保存，供 raw 计算使用）
        self._skeleton_takeoff_x_cm = None
        self._skeleton_landing_x_cm = None

        # YOLO / MOG2 修正值标志
        self._yolo_takeoff_x_cm = None
        self._yolo_landing_x_cm = None
        self._mog_takeoff_x_cm = None
        self._mog_landing_x_cm = None

        # ── 模拟流 / OCR 时间码 ──
        self.stream_mode = bool(getattr(config, "stream_mode", False))
        self._stream_empty_counter = 0
        self._stream_recalib_done_for_empty_period = False
        self._stream_recalib_count = 0
        self._stream_jump_active = False
        self._stream_result_recorded = False
        self._jump_index = 0
        self._stream_results = []
        self._current_time_info = None
        self._time_info_by_frame = {}  # 1-based frame_idx -> OCR/frame time info，供倒推起跳帧/落地帧精确取时间
        self._last_ocr_log_text = None
        self._video_fps = 30.0
        self._video_total_frames = 0
        if self.pose_estimator.cap is not None:
            cap_fps = float(self.pose_estimator.cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if cap_fps > 1e-3:
                self._video_fps = cap_fps
            self._video_total_frames = int(self.pose_estimator.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        max_ocr_seconds = int(self._video_total_frames / self._video_fps) + 5 if self._video_total_frames else 3600
        self.ocr_reader = None
        if getattr(config, "enable_ocr_time", False):
            self.ocr_reader = TimeOCR(
                roi=getattr(config, "ocr_roi", (0, 0, 260, 80)),
                fps=self._video_fps,
                max_seconds=max_ocr_seconds,
                enabled=True,
            )
            self._log("OCR", f"左上角时间 OCR 已开启, roi={self.ocr_reader.roi}, fps={self._video_fps:.3f}, max_seconds={max_ocr_seconds}")
        if self.stream_mode:
            self._log("STREAM", f"模拟流模式已开启: empty_frames={getattr(config, 'stream_recalib_empty_frames', 15)}, fps={self._video_fps:.3f}")

        if config.display:
            cv2.namedWindow("Auto Long Jump", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Auto Long Jump", 1280, 720)
            if self.calibrator.manual_mode:
                cv2.setMouseCallback("Auto Long Jump", self.calibrator.mouse_callback)

    # ---------- 辅助方法 ----------
    def _log(self, tag, message):
        """写运行日志（同时输出到控制台）。"""
        print(f"[{tag}] {message}")
        if self.run_logger:
            self.run_logger.info(f"[{tag}] {message}")

    def _log_keypoints(self, frame_idx, kpts):
        """记录一帧的关键点数据到关键点日志。"""
        if self._kpts_log_fh is None or kpts is None:
            return
        parts = [f"frame={frame_idx}"]
        for i, pt in enumerate(kpts):
            if pt[0] > 0 and pt[1] > 0:
                parts.append(f"kpt_{i}=({pt[0]:.1f},{pt[1]:.1f})")
        self._kpts_log_fh.write("[" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "] " + " ".join(parts) + "\n")
        self._kpts_log_fh.flush()

    def _get_kpt(self, kpts, idx):
        if kpts is None or idx >= len(kpts):
            return None
        pt = kpts[idx]
        if pt[0] <= 0 or pt[1] <= 0:
            return None
        return float(pt[0]), float(pt[1])

    def _get_feet(self, kpts, kind):
        if kind == "toe" and len(kpts) > 22:
            left = self._get_kpt(kpts, self.kpt_idx["l_big_toe"])
            right = self._get_kpt(kpts, self.kpt_idx["r_big_toe"])
            if left is not None or right is not None:
                return {"l": left, "r": right}
            return {"l": None, "r": None}
        if kind == "heel" and len(kpts) > 22:
            left = self._get_kpt(kpts, self.kpt_idx["l_heel"])
            right = self._get_kpt(kpts, self.kpt_idx["r_heel"])
            if left is not None or right is not None:
                return {"l": left, "r": right}
            return {"l": None, "r": None}
        left = self._get_kpt(kpts, self.kpt_idx["l_ankle"])
        right = self._get_kpt(kpts, self.kpt_idx["r_ankle"])
        return {"l": left, "r": right}

    @staticmethod
    def _avg_points(points):
        valid = [p for p in points.values() if p is not None]
        if not valid:
            return None
        x = sum(p[0] for p in valid) / len(valid)
        y = sum(p[1] for p in valid) / len(valid)
        return x, y

    @staticmethod
    def _normalize_jump_direction(jump_direction):
        v = str(jump_direction or "ltr").strip().lower()
        if v in ("rtl", "right-to-left", "right2left", "r2l"):
            return "rtl"
        return "ltr"

    def _takeoff_line_x_cm(self):
        """标准起跳线在垫子坐标系中的实际 X。rtl 时自动放到右侧。"""
        line = float(getattr(self.config, "takeoff_line_cm", 30.0))
        if self.is_rtl:
            return float(self.calibrator.mat_length_cm) - line
        return line

    def _forward_delta(self, x_cm, ref_x_cm):
        if x_cm is None or ref_x_cm is None:
            return 0.0
        return (ref_x_cm - x_cm) if self.is_rtl else (x_cm - ref_x_cm)

    def _distance_between_x(self, takeoff_x_cm, landing_x_cm):
        return max(0.0, self._forward_delta(landing_x_cm, takeoff_x_cm))

    def _apply_takeoff_offset(self, x_cm):
        if x_cm is None:
            return None
        sign = -1.0 if self.is_rtl else 1.0
        return x_cm + sign * self.takeoff_display_offset_cm

    def _apply_landing_offset(self, x_cm):
        if x_cm is None:
            return None
        sign = -1.0 if self.is_rtl else 1.0
        return x_cm + sign * self.landing_offset_cm

    def _front_toe(self, toe_l_xy, toe_r_xy, toe_l_cm, toe_r_cm):
        if toe_l_cm is not None and toe_r_cm is not None:
            if self.is_rtl:
                return (toe_l_xy, toe_l_cm) if toe_l_cm[0] <= toe_r_cm[0] else (toe_r_xy, toe_r_cm)
            return (toe_l_xy, toe_l_cm) if toe_l_cm[0] >= toe_r_cm[0] else (toe_r_xy, toe_r_cm)
        if toe_l_cm is not None:
            return toe_l_xy, toe_l_cm
        if toe_r_cm is not None:
            return toe_r_xy, toe_r_cm
        return None, None

    def _reset_round_state(self):
        # 骨架法变量
        self._skeleton_baseline_x_cm = None
        self._skeleton_baseline_hip_x = None
        self._skeleton_baseline_ankle_y = None
        self._skeleton_in_mat_logged = False
        self._skeleton_jump_trigger_counter = 0
        self._skeleton_pending_takeoff = None
        self._skeleton_ready_stable = 0
        self._skeleton_front_toe_hist.clear()
        self._skeleton_takeoff_candidate_hist.clear()
        self._skeleton_toe_y_px_hist.clear()
        self._skeleton_toe_missing_counter = 0
        self._skeleton_ready_invalid_counter = 0
        self._skeleton_last_hip_moved_cm = 0.0
        self._skeleton_jump_counter = 0
        self._skeleton_prev_takeoff_data = None
        self.takeoff_frame = None
        self.takeoff_trigger_frame = None
        self.landing_frame = None
        for side in ["l", "r"]:
            for q in self._skeleton_foot_hist[side].values():
                q.clear()
        # 公共变量
        self.takeoff_x_cm = None
        self.takeoff_pt_px = None
        self.landing_x_cm = None
        self.landing_pt_px = None
        self.final_distance_cm = None
        self._takeoff_frame_img = None
        self._landing_frame_img = None
        self._landing_kpts = None
        self._landing_all_kpts_list = []
        self.takeoff_pt_xy = None
        self.landing_pt_xy = None
        self._takeoff_saved = False
        self._landed_saved = False
        self._score_saved = False
        self._foul_saved = False
        self._debug_takeoff_toe_moved_saved = False
        self.foul_detector.reset()
        self._diff_computed = False
        self._skeleton_takeoff_x_cm = None
        self._skeleton_landing_x_cm = None
        self._yolo_takeoff_x_cm = None
        self._yolo_landing_x_cm = None
        self._mog_takeoff_x_cm = None
        self._mog_landing_x_cm = None
        self.calibrator.mat_locked = True

    def _ensure_record_writer(self, display_img):
        if not self.config.record_path or self.record_writer is not None:
            return
        fps = 30.0
        if self.pose_estimator.cap is not None:
            cap_fps = float(self.pose_estimator.cap.get(cv2.CAP_PROP_FPS))
            if cap_fps > 1e-3:
                fps = cap_fps
        out_dir = os.path.dirname(self.config.record_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        h, w = display_img.shape[:2]
        self.record_writer = cv2.VideoWriter(self.config.record_path, fourcc, fps, (w, h))
        if self.record_writer is None or not self.record_writer.isOpened():
            self._log("RECORD", f"运行画面视频打开失败: {self.config.record_path}")
            self.record_writer = None
            return
        self._log("RECORD", f"运行画面视频开始录制: {self.config.record_path}, fps={fps:.3f}, size={w}x{h}")


    def _image_output_path(self, base_name):
        if not self.images_dir:
            return base_name
        if self.stream_mode and self._jump_index > 0:
            stem, ext = os.path.splitext(base_name)
            return os.path.join(self.images_dir, f"{stem}_jump_{self._jump_index:03d}{ext}")
        return os.path.join(self.images_dir, base_name)

    def _frame_time_info(self, frame_idx=None):
        if frame_idx is None:
            return self._current_time_info
        frame_idx = int(frame_idx)
        cached = self._time_info_by_frame.get(frame_idx)
        if cached is not None:
            return dict(cached)
        frame_idx0 = int(max(0, frame_idx - 1))
        if self.ocr_reader is not None:
            # 优先使用每帧 OCR 缓存；没有缓存时退回到帧号换算。
            return self.ocr_reader.format_frame_time(frame_idx0, seconds=None, ok=False, confidence=0.0)
        fps_i = max(1, int(round(self._video_fps)))
        sec = int(frame_idx0 // fps_i)
        ff = int(frame_idx0 % fps_i)
        h = sec // 3600
        m = (sec % 3600) // 60
        s = sec % 60
        text = f"{h:02d}:{m:02d}:{s:02d}"
        return {"ok": False, "text": text, "second": sec, "frame_in_second": ff,
                "timecode": f"{text}:{ff:02d}", "frame_idx0": frame_idx0, "confidence": 0.0}

    def _has_person_in_mat(self, kpts, margin_cm=8.0):
        if not self.calibrator.calibrated:
            return False
        people = []
        if self.pose_estimator.all_kpts_list:
            people.extend(self.pose_estimator.all_kpts_list)
        elif kpts is not None:
            people.append(kpts)
        indices = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        for person in people:
            if person is None:
                continue
            for idx in indices:
                if idx >= len(person):
                    continue
                x, y = float(person[idx][0]), float(person[idx][1])
                if x <= 0 or y <= 0:
                    continue
                cm = self.calibrator.transform_to_mat_cm((x, y))
                if cm is None:
                    continue
                cx, cy = cm
                if (-margin_cm <= cx <= self.calibrator.mat_length_cm + margin_cm and
                        -margin_cm <= cy <= self.calibrator.mat_width_cm + margin_cm):
                    return True
        return False

    def _reset_diff_detector_round(self, reset_base=False):
        if reset_base:
            self.diff_detector._base_frame_raw = None
            self.diff_detector._base_frame_gray = None
            self.diff_detector._base_frame_captured = False
        for name in [
            "takeoff_diff_mask", "landing_diff_mask", "_takeoff_edge_px", "_landing_edge_px",
            "_takeoff_edge_mat_px", "_landing_edge_mat_px", "_takeoff_edge_foot_label",
            "_landing_edge_foot_label", "takeoff_shoe_x_cm", "landing_shoe_x_cm",
            "_takeoff_frame", "_takeoff_kpts", "_landing_frame", "_landing_kpts",
        ]:
            if hasattr(self.diff_detector, name):
                setattr(self.diff_detector, name, None)

    def _stream_try_recalibrate_mat(self, frame_idx, frame, reason="empty"):
        if self.calibrator.manual_mode or not self.stream_mode:
            return False
        backup = {
            "mat_locked": self.calibrator.mat_locked,
            "calibrated": self.calibrator.calibrated,
            "_smooth_box": None if self.calibrator._smooth_box is None else self.calibrator._smooth_box.copy(),
            "_last_box_points": None if self.calibrator._last_box_points is None else self.calibrator._last_box_points.copy(),
            "H_img2mat": None if self.calibrator.H_img2mat is None else self.calibrator.H_img2mat.copy(),
            "H_mat2img": None if self.calibrator.H_mat2img is None else self.calibrator.H_mat2img.copy(),
            "jump_line_px": self.calibrator.jump_line_px,
            "px_per_cm": self.calibrator.px_per_cm,
        }
        self.calibrator.mat_locked = False
        self.calibrator.calibrated = False
        self.calibrator._smooth_box = None
        self.calibrator._last_box_points = None
        self.calibrator.H_img2mat = None
        self.calibrator.H_mat2img = None
        self.calibrator.jump_line_px = None
        ok = self.calibrator.update(frame)
        if ok:
            self.calibrator.mat_locked = True
            self._stream_recalib_count += 1
            self._reset_diff_detector_round(reset_base=True)
            self._log("STREAM_RECALIB", f"垫内无人连续 {self._stream_empty_counter} 帧，已重新标定并锁定: frame={frame_idx}, count={self._stream_recalib_count}, reason={reason}")
            if self.config.enable_mat_output and self.images_dir:
                mask_quad = self.calibrator.render_mask(frame)
                if mask_quad is not None:
                    imwrite_safe(os.path.join(self.images_dir, f"mat_mask_quad_stream_{self._stream_recalib_count:03d}.jpeg"), cv2.cvtColor(mask_quad, cv2.COLOR_GRAY2BGR))
            return True
        for key, value in backup.items():
            setattr(self.calibrator, key, value)
        self._log("STREAM_RECALIB", f"尝试重新标定失败，保留上一组垫子参数: frame={frame_idx}, reason={reason}")
        return False

    def _update_stream_recalibration(self, frame_idx, frame, kpts):
        if not self.stream_mode or self.state != "IDLE" or not self.calibrator.calibrated:
            return
        has_person = self._has_person_in_mat(kpts)
        if has_person:
            if self._stream_empty_counter > 0 and self.debug:
                self._log("STREAM", f"frame={frame_idx}: 垫内重新检测到人，empty_counter 清零")
            self._stream_empty_counter = 0
            self._stream_recalib_done_for_empty_period = False
            return
        self._stream_empty_counter += 1
        need = max(1, int(getattr(self.config, "stream_recalib_empty_frames", 15)))
        if self._stream_empty_counter >= need and not self._stream_recalib_done_for_empty_period:
            if self._stream_try_recalibrate_mat(frame_idx, frame):
                self._stream_recalib_done_for_empty_period = True
                self._stream_empty_counter = 0

    def _current_payload(self):
        payload = {
            "score": float(self.final_distance_cm or 0.0),
            "valid": False if self.foul_detector.reason else True,
            "foul_reason": self.foul_detector.reason,
            "distance_cm": float(self.final_distance_cm or 0.0),
            "jump_direction": self.jump_direction,
            "takeoff_line_x_cm": float(self._takeoff_line_x_cm()),
            "takeoff_x_cm": float(self.takeoff_x_cm or 0.0),
            "landing_x_cm": float(self.landing_x_cm or 0.0) if self.landing_x_cm is not None else None,
            "takeoff_frame": self.takeoff_frame,
            "takeoff_trigger_frame": self.takeoff_trigger_frame,
            "landing_frame": self.landing_frame,
            "takeoff_time": self._frame_time_info(self.takeoff_frame) if self.takeoff_frame is not None else None,
            "takeoff_trigger_time": self._frame_time_info(self.takeoff_trigger_frame) if self.takeoff_trigger_frame is not None else None,
            "landing_time": self._frame_time_info(self.landing_frame) if self.landing_frame is not None else None,
            "score_time": self._current_time_info,
            "yolo_infer_time_s": round(self.diff_detector.yolo_total_time, 3),
        }
        if self.stream_mode:
            payload["stream_mode"] = True
            payload["jump_index"] = int(self._jump_index)
            payload["stream_recalib_count"] = int(self._stream_recalib_count)
        return payload

    def _record_stream_result(self):
        if not self.stream_mode or self._stream_result_recorded or self._jump_index <= 0:
            return
        payload = self._current_payload()
        payload["score_image"] = os.path.join("images", os.path.basename(self._image_output_path("score.jpeg"))) if self.images_dir else None
        self._stream_results.append(payload)
        if self.result_dir:
            import json
            json_path = os.path.join(self.result_dir, "stream_results.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self._stream_results, f, ensure_ascii=False, indent=2)
            csv_path = os.path.join(self.result_dir, "stream_results.csv")
            with open(csv_path, "w", encoding="utf-8-sig") as f:
                f.write("jump_index,distance_cm,valid,foul_reason,takeoff_frame,takeoff_trigger_frame,landing_frame,takeoff_timecode,takeoff_trigger_timecode,landing_timecode,score_image\n")
                for item in self._stream_results:
                    def tc(key):
                        val = item.get(key) or {}
                        return val.get("timecode", "") if isinstance(val, dict) else ""
                    f.write(f"{item.get('jump_index','')},{item.get('distance_cm','')},{item.get('valid','')},{item.get('foul_reason') or ''},"
                            f"{item.get('takeoff_frame') or ''},{item.get('takeoff_trigger_frame') or ''},{item.get('landing_frame') or ''},"
                            f"{tc('takeoff_time')},{tc('takeoff_trigger_time')},{tc('landing_time')},{item.get('score_image') or ''}\n")
            self._log("STREAM_RESULT", f"第 {self._jump_index:03d} 跳成绩已写入 stream_results.json/csv: distance={self.final_distance_cm:.1f}cm, landing_time={payload.get('landing_time')}")
        self._stream_result_recorded = True

    def _reset_for_next_stream_jump(self):
        if not self.stream_mode:
            return
        self._reset_round_state()
        self.state = "IDLE"
        self.takeoff_frame = None
        self.takeoff_trigger_frame = None
        self.landing_frame = None
        self._stream_jump_active = False
        self._stream_result_recorded = False
        self._reset_diff_detector_round(reset_base=False)
        self._log("STREAM", "本跳完成，状态已重置为 IDLE，继续等待下一跳/无人重标定")

    def _save_payload(self):
        payload = self._current_payload()
        save_path = os.path.join(self.result_dir, "result.json") if self.result_dir else self.config.save_path
        with open(save_path, "w", encoding="utf-8") as f:
            import json
            json.dump(payload, f, ensure_ascii=False, indent=2)
        if self.stream_mode and self.result_dir and self._jump_index > 0:
            jump_path = os.path.join(self.result_dir, f"result_jump_{self._jump_index:03d}.json")
            with open(jump_path, "w", encoding="utf-8") as f:
                import json
                json.dump(payload, f, ensure_ascii=False, indent=2)
        self._log("SAVE", f"结果已保存到 {save_path}")

    def _save_foul_record(self, frame, kpts=None):
        if self._foul_saved or self.foul_detector.reason is None:
            return

        img = frame.copy()
        self.renderer.draw_mat_outline(img, self.calibrator)
        self.renderer.draw_x_line(img, self.calibrator.H_mat2img, self.calibrator.mat_width_cm,
                                  self._takeoff_line_x_cm(), (255, 255, 255), thickness=3)

        actual_takeoff_x = self.takeoff_x_cm
        if actual_takeoff_x is not None:
            color = (0, 0, 255) if "踩线" in str(self.foul_detector.reason) else (0, 255, 255)
            self.renderer.draw_x_line(img, self.calibrator.H_mat2img, self.calibrator.mat_width_cm,
                                      actual_takeoff_x, color, thickness=2, label="Takeoff")

        if self.pose_estimator.all_kpts_list:
            for pk in self.pose_estimator.all_kpts_list:
                self.renderer.draw_pose(img, pk, self.pose_estimator.mp_connections, color=(0, 255, 0))
                self.renderer.draw_feet(img, self._get_feet, pk)
        elif kpts is not None:
            self.renderer.draw_pose(img, kpts, self.pose_estimator.mp_connections, color=(0, 255, 0))
            self.renderer.draw_feet(img, self._get_feet, kpts)

        # 在犯规图片上叠加成绩（如有）
        score_text = f"成绩: {self.final_distance_cm:.1f} cm" if self.final_distance_cm is not None else "成绩: 无"
        takeoff_text = f"起跳点: {self.takeoff_x_cm:.1f} cm" if self.takeoff_x_cm is not None else ""
        landing_text = f"落地点: {self.landing_x_cm:.1f} cm" if self.landing_x_cm is not None else ""
        img = self.renderer.put_text_chinese(img, f"犯规: {self.foul_detector.reason}", (50, 80), (0, 0, 255), size=50)
        img = self.renderer.put_text_chinese(img, score_text, (50, 140), (0, 255, 255), size=40)
        if takeoff_text:
            img = self.renderer.put_text_chinese(img, takeoff_text, (50, 190), (255, 255, 0), size=30)
        if landing_text:
            img = self.renderer.put_text_chinese(img, landing_text, (50, 230), (255, 255, 0), size=30)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self.images_dir, f"foul-{ts}.jpeg") if self.images_dir else f"foul-{ts}.jpeg"
        imwrite_safe(filename, img)
        self._log("FOUL", f"犯规图片已保存: {filename}")
        self._foul_saved = True

    def _recalc_results_with_current_mat(self):
        if not self.calibrator.calibrated:
            return
        # 启用 YOLO/DIFF 时由修正后的起跳线统一判断踩线，此处跳过
        if self.takeoff_x_cm is not None and not self.config.enable_seg and not self.config.enable_diff:
            self.foul_detector.check_line_violation(self.takeoff_x_cm, self._takeoff_line_x_cm(), self.jump_direction)
        if self.final_distance_cm is not None and self.landing_x_cm is not None:
            ld_cm = (self.landing_x_cm, 0.0)
            self.foul_detector.check_out_of_bounds(ld_cm)

    # ═══════════════════════════════════════════════
    # 骨骼关键点法处理函数
    # ═══════════════════════════════════════════════


    @staticmethod
    def _skeleton_detect_contact(hist):
        """脚后跟 Y 轴触地检测（骨架法）。"""
        if len(hist["y_px"]) < 3 or len(hist["x_cm"]) < 3:
            return None, None, None, None, []
        y0, y1, y2 = list(hist["y_px"])
        x1 = list(hist["x_cm"])[1]
        if (y1 >= y0) and (y1 >= y2) and x1 > 20.0:
            xy_mid = list(hist["xy_px"])[1] if len(hist["xy_px"]) == 3 else None
            frame_mid = list(hist["frame_idx"])[1] if len(hist.get("frame_idx", [])) == 3 else None
            frame_img_mid = list(hist["frame_img"])[1] if len(hist.get("frame_img", [])) == 3 else None
            kpts_mid = list(hist["kpts"])[1] if len(hist.get("kpts", [])) == 3 else None
            all_kpts_mid = list(hist["all_kpts_list"])[1] if len(hist.get("all_kpts_list", [])) == 3 else []
            return x1, xy_mid, frame_mid, frame_img_mid, (kpts_mid, all_kpts_mid)
        return None, None, None, None, []

    def _skeleton_enter_ready(self, ankle_cm, front_toe_cm):
        if front_toe_cm is None or not self.calibrator.in_mat(front_toe_cm):
            return
        self.state = "READY"
        self._log("STATE", "IDLE -> READY")
        self._reset_round_state()
        self._skeleton_baseline_x_cm = front_toe_cm[0]
        self._log("READY", f"骨架法基线X={self._skeleton_baseline_x_cm:.1f}cm")


    def _save_skeleton_takeoff_candidate(self, frame_idx, kpts, front_toe_cm, front_toe_xy, toe_xy, ankle_xy):
        """保存可用于倒推起跳的候选帧。"""
        if front_toe_cm is None:
            return
        self._skeleton_prev_takeoff_data = {
            "front_toe_cm": front_toe_cm,
            "front_toe_xy": front_toe_xy,
            "toe_xy": toe_xy,
            "ankle_xy": ankle_xy,
            "frame_idx": frame_idx,
            "kpts": kpts,
            "all_kpts_list": list(self.pose_estimator.all_kpts_list) if self.pose_estimator.all_kpts_list else [],
            "frame_img": self._prev_frame_img.copy() if self._prev_frame_img is not None else None,
            "synthetic": False,
        }
        self._skeleton_takeoff_candidate_hist.append(dict(self._skeleton_prev_takeoff_data))

    def _save_skeleton_synthetic_takeoff_candidate(self, frame_idx, kpts, ankle_xy=None):
        """关键点短暂缺失时，把上一帧可靠脚尖位置绑定到当前画面，补齐倒推候选。"""
        prev = self._skeleton_prev_takeoff_data
        if prev is None or prev.get("front_toe_cm") is None:
            return False
        self._skeleton_prev_takeoff_data = {
            "front_toe_cm": prev.get("front_toe_cm"),
            "front_toe_xy": prev.get("front_toe_xy"),
            "toe_xy": prev.get("toe_xy"),
            "ankle_xy": ankle_xy if ankle_xy is not None else prev.get("ankle_xy"),
            "frame_idx": frame_idx,
            # synthetic 候选帧只换画面，不采用当前乱飞/越界关键点；犯规检测和骨架绘制沿用上一帧可靠姿态。
            "kpts": prev.get("kpts") if prev.get("kpts") is not None else kpts,
            "all_kpts_list": prev.get("all_kpts_list", []),
            "frame_img": self._prev_frame_img.copy() if self._prev_frame_img is not None else prev.get("frame_img"),
            "synthetic": True,
        }
        self._skeleton_takeoff_candidate_hist.append(dict(self._skeleton_prev_takeoff_data))
        return True

    def _select_skeleton_takeoff_candidate(self, trigger_frame_idx):
        """先锁定起跳触发帧，再选择 trigger-N 的正式起跳帧。"""
        backtrack = max(0, int(getattr(self.config, "takeoff_backtrack_frames", 2)))
        target_frame = trigger_frame_idx - backtrack
        candidates = [c for c in self._skeleton_takeoff_candidate_hist
                      if c is not None and c.get("front_toe_cm") is not None and c.get("frame_idx") is not None]
        if self._skeleton_prev_takeoff_data is not None:
            prev = self._skeleton_prev_takeoff_data
            if prev.get("front_toe_cm") is not None and prev.get("frame_idx") is not None:
                candidates.append(prev)
        if not candidates:
            return None, target_frame, backtrack
        exact = [c for c in candidates if c.get("frame_idx") == target_frame]
        if exact:
            return exact[-1], target_frame, backtrack
        earlier = [c for c in candidates if c.get("frame_idx") <= target_frame]
        if earlier:
            earlier.sort(key=lambda c: c.get("frame_idx", -10**9))
            return earlier[-1], target_frame, backtrack
        candidates.sort(key=lambda c: abs(c.get("frame_idx", trigger_frame_idx) - target_frame))
        return candidates[0], target_frame, backtrack

    def _confirm_skeleton_takeoff(self, frame_idx, kpts, front_toe_cm, front_toe_xy, toe_xy, ankle_xy, takeoff_reason):
        """确认骨架法起跳，统一执行倒推、犯规检测、图像保存。"""
        # 先精准判断触发帧(frame_idx)，再按配置倒推 N 帧作为正式起跳帧/修正图。
        prev, target_frame, backtrack = self._select_skeleton_takeoff_candidate(frame_idx)
        if prev is not None and prev["front_toe_cm"] is not None:
            takeoff_x = prev["front_toe_cm"][0]
            takeoff_frame = prev["frame_idx"]
            takeoff_pt_px = (prev["front_toe_xy"] if prev["front_toe_xy"] is not None
                             else (prev["toe_xy"] if prev["toe_xy"] is not None else prev["ankle_xy"]))
            takeoff_kpts = prev["kpts"]
            takeoff_img = prev["frame_img"]
            takeoff_all_kpts = prev.get("all_kpts_list", [])
        elif front_toe_cm is not None:
            # 无缓存时才回退到当前帧；正常视频不应走到这里。
            takeoff_x = front_toe_cm[0]
            takeoff_frame = frame_idx
            takeoff_pt_px = front_toe_xy if front_toe_xy is not None else (toe_xy if toe_xy else ankle_xy)
            takeoff_kpts = kpts
            takeoff_img = self._prev_frame_img.copy() if self._prev_frame_img is not None else None
            takeoff_all_kpts = list(self.pose_estimator.all_kpts_list) if self.pose_estimator.all_kpts_list else []
        else:
            self._log("JUMP", f"起跳触发帧{frame_idx}但没有可用脚尖候选，忽略本次触发")
            self._skeleton_jump_trigger_counter = 0
            return
        source_note = f"target={target_frame}, backtrack={backtrack}"
        if takeoff_frame != target_frame:
            source_note += f", used_nearest={takeoff_frame}"
        if prev is not None and prev.get("synthetic"):
            source_note += ", synthetic"
        if self.debug:
            self._log("DEBUG_TAKEOFF", f"帧{takeoff_frame}(<-{frame_idx}, {source_note}): 起跳触发 | reason={takeoff_reason} | takeoff_x={takeoff_x:.1f}cm | stable={self._skeleton_ready_stable}帧 | trigger={self._skeleton_jump_trigger_counter}/1帧")
        self._log("JUMP", f"起跳成功(骨架)！触发帧={frame_idx}, 起跳帧={takeoff_frame}(倒推{backtrack}帧), 起跳点={takeoff_x:.1f}cm")

        self.foul_detector.check_step_jump(self._skeleton_front_toe_hist, self._skeleton_baseline_x_cm, self.jump_direction)
        self.foul_detector.check_single_leg_takeoff(takeoff_kpts)
        self.foul_detector.check_prop_assistance(takeoff_kpts)
        # 启用 YOLO/DIFF 时延后踩线检测（等修正后的起跳线）
        if not self.config.enable_seg and not self.config.enable_diff:
            self.foul_detector.check_line_violation(takeoff_x, self._takeoff_line_x_cm(), self.jump_direction)
        if self.foul_detector.reason:
            self._log("FOUL", f"起跳时检测到犯规: {self.foul_detector.reason}")

        if self.stream_mode and not self._stream_jump_active:
            self._jump_index += 1
            self._stream_jump_active = True
            self._stream_result_recorded = False
            self._log("STREAM", f"开始第 {self._jump_index:03d} 跳: takeoff_trigger_frame={frame_idx}, takeoff_frame={takeoff_frame}")
        self.state = "JUMPING"
        self._log("STATE", "READY -> JUMPING")
        self._skeleton_jump_counter = 0
        self.takeoff_frame = takeoff_frame
        self.takeoff_trigger_frame = frame_idx
        self.takeoff_pt_px = takeoff_pt_px
        self.takeoff_pt_xy = self.takeoff_pt_px
        self.takeoff_x_cm = self._apply_takeoff_offset(takeoff_x)
        self._skeleton_takeoff_x_cm = self.takeoff_x_cm  # 骨骼修正值（YOLO 覆盖前保存）
        takeoff_img_to_save = takeoff_img if takeoff_img is not None else self._prev_frame_img
        self._save_takeoff_image(takeoff_img_to_save, takeoff_kpts, takeoff_all_kpts)
        self._takeoff_frame_img = takeoff_img_to_save.copy() if takeoff_img_to_save is not None else None
        self._takeoff_kpts = takeoff_kpts
        self._takeoff_all_kpts_list = takeoff_all_kpts
        # 差分法/YOLO：保存倒推后的正式起跳帧
        if self.config.enable_seg or self.diff_detector.has_base_frame:
            self.diff_detector.save_takeoff_frame(takeoff_img_to_save, takeoff_kpts)
    def _handle_idle_skeleton(self, frame_idx, ankle_cm, front_toe_cm, toe_l_cm=None, toe_r_cm=None):
        self.foul_detector.reset()
        # 第一阶段：脚尖进入垫子区域 → 提示人体已踏上垫子（仅首次输出）
        if front_toe_cm is not None and self.calibrator.in_mat(front_toe_cm):
            if not self._skeleton_in_mat_logged:
                self._skeleton_in_mat_logged = True
                self._log("IDLE", f"检测到人体在垫内(骨架), toe=({front_toe_cm[0]:.1f},{front_toe_cm[1]:.1f})cm")
        # 第二阶段：双脚脚尖都距起跳线 ≤ 5cm → 预备起跳，切换至 READY
        if ankle_cm is not None and toe_l_cm is not None and toe_r_cm is not None:
            toe_l_to_line = abs(toe_l_cm[0] - self._takeoff_line_x_cm())
            toe_r_to_line = abs(toe_r_cm[0] - self._takeoff_line_x_cm())
            if self.debug:
                ankle_str = f"({ankle_cm[0]:.1f},{ankle_cm[1]:.1f})cm" if ankle_cm is not None else "None"
                ft_str = f"({front_toe_cm[0]:.1f},{front_toe_cm[1]:.1f})cm" if front_toe_cm is not None else "None"
                self._log("DEBUG_IDLE_PARAMS", f"帧{frame_idx}: ankle={ankle_str} | front_toe={ft_str} | "
                          f"toe_l_to_line={toe_l_to_line:.1f}cm | toe_r_to_line={toe_r_to_line:.1f}cm | "
                          f"in_mat={self.calibrator.in_mat(front_toe_cm) if front_toe_cm else False}")
            if toe_l_to_line <= 10.0 and toe_r_to_line <= 10.0:
                self._log("IDLE", f"检测预备起跳(骨架), ankle=({ankle_cm[0]:.1f},{ankle_cm[1]:.1f})cm")
                self._skeleton_enter_ready(ankle_cm, front_toe_cm)

    def _handle_ready_skeleton(self, frame_idx, kpts, ankle_cm, ankle_xy, front_toe_cm, front_toe_xy, toe_xy, takeoff_signal):
        hip_l = self._get_kpt(kpts, self.kpt_idx["l_hip"])
        hip_r = self._get_kpt(kpts, self.kpt_idx["r_hip"])
        hip_xy = self._avg_points({"l": hip_l, "r": hip_r})
        hip_cm = self.calibrator.transform_to_mat_cm(hip_xy) if hip_xy else None
        curr_ankle_y_px = ankle_xy[1] if ankle_xy else 99999

        # 有些视频在真正离地前会先出现“脚踝抬起/脚尖回缩”的预抬脚帧。
        # 这里允许把这类候选延迟 1~2 帧再正式确认触发帧；倒推起跳图仍从最终触发帧回推。
        if self._skeleton_pending_takeoff is not None:
            pending = self._skeleton_pending_takeoff
            confirm_frame = pending.get("confirm_frame", frame_idx)
            if frame_idx >= confirm_frame:
                self._skeleton_jump_trigger_counter = 1
                takeoff_reason = f"{pending.get('reason', '起跳候选')}，延迟确认{pending.get('delay', 0)}帧"
                self._skeleton_pending_takeoff = None
                if self.debug:
                    self._log("DEBUG_TAKEOFF_PARAMS", f"帧{frame_idx}: pending_takeoff到期 | trigger=1/1 | reason={takeoff_reason}")
                self._confirm_skeleton_takeoff(frame_idx, kpts, front_toe_cm, front_toe_xy, toe_xy, ankle_xy, takeoff_reason)
                return
            if front_toe_cm is not None and front_toe_xy is not None and self.calibrator.in_mat(front_toe_cm):
                self._save_skeleton_takeoff_candidate(frame_idx, kpts, front_toe_cm, front_toe_xy, toe_xy, ankle_xy)
            else:
                self._save_skeleton_synthetic_takeoff_candidate(frame_idx, kpts, ankle_xy)
            if self.debug:
                self._log("DEBUG_TAKEOFF_PARAMS", f"帧{frame_idx}: pending_takeoff等待到{confirm_frame}确认")
            return

        if ankle_cm is None:
            self._skeleton_ready_invalid_counter += 1
            invalid_hip_moved = (self._forward_delta(hip_cm[0], self._skeleton_baseline_hip_x)
                                 if (hip_cm and self._skeleton_baseline_hip_x is not None) else self._skeleton_last_hip_moved_cm)
            if (self._skeleton_ready_stable >= 40
                    and self._skeleton_ready_invalid_counter >= 7
                    and self._skeleton_prev_takeoff_data is not None
                    and invalid_hip_moved > 70.0):
                self._skeleton_jump_trigger_counter = 1
                takeoff_reason = (f"READY后关键点连续无效{self._skeleton_ready_invalid_counter}帧，"
                                  f"使用上一帧稳定脚尖倒推")
                if self.debug:
                    self._log("DEBUG_TAKEOFF_PARAMS", f"帧{frame_idx}: ankle_cm=None | invalid_ready={self._skeleton_ready_invalid_counter} | "
                              f"trigger=1/1 | stable={self._skeleton_ready_stable} | reason={takeoff_reason}")
                self._confirm_skeleton_takeoff(frame_idx, kpts, None, None, toe_xy, ankle_xy, takeoff_reason)
                return
            self._save_skeleton_synthetic_takeoff_candidate(frame_idx, kpts, ankle_xy)
            if self.debug:
                self._log("DEBUG_TAKEOFF_PARAMS", f"帧{frame_idx}: ankle_cm=None，跳过本帧 | invalid_ready={self._skeleton_ready_invalid_counter}")
            return

        if front_toe_cm is None or not self.calibrator.in_mat(front_toe_cm):
            self._skeleton_ready_invalid_counter += 1
            invalid_hip_moved = (self._forward_delta(hip_cm[0], self._skeleton_baseline_hip_x)
                                 if (hip_cm and self._skeleton_baseline_hip_x is not None) else self._skeleton_last_hip_moved_cm)
            if (self._skeleton_ready_stable >= 40
                    and self._skeleton_ready_invalid_counter >= 7
                    and self._skeleton_prev_takeoff_data is not None
                    and invalid_hip_moved > 70.0):
                self._skeleton_jump_trigger_counter = 1
                takeoff_reason = (f"READY后脚尖连续缺失/越界{self._skeleton_ready_invalid_counter}帧，"
                                  f"使用上一帧稳定脚尖倒推")
                if self.debug:
                    in_mat_str = "N/A" if front_toe_cm is None else f"(in_mat={self.calibrator.in_mat(front_toe_cm)})"
                    ft_str = "None" if front_toe_cm is None else f"({front_toe_cm[0]:.1f},{front_toe_cm[1]:.1f})cm"
                    self._log("DEBUG_TAKEOFF_PARAMS", f"帧{frame_idx}: front_toe_cm={ft_str} {in_mat_str} | invalid_ready={self._skeleton_ready_invalid_counter} | "
                              f"trigger=1/1 | stable={self._skeleton_ready_stable} | reason={takeoff_reason}")
                self._confirm_skeleton_takeoff(frame_idx, kpts, None, None, toe_xy, ankle_xy, takeoff_reason)
                return
            self._save_skeleton_synthetic_takeoff_candidate(frame_idx, kpts, ankle_xy)
            if self.debug:
                in_mat_str = "N/A" if front_toe_cm is None else f"(in_mat={self.calibrator.in_mat(front_toe_cm)})"
                ft_str = "None" if front_toe_cm is None else f"({front_toe_cm[0]:.1f},{front_toe_cm[1]:.1f})cm"
                self._log("DEBUG_TAKEOFF_PARAMS", f"帧{frame_idx}: front_toe_cm={ft_str} {in_mat_str}，跳过本帧 | invalid_ready={self._skeleton_ready_invalid_counter}")
            return  # 无脚尖位置数据，跳过本帧（不用脚踝替代起跳点）

        self._skeleton_ready_invalid_counter = 0

        cur_x = front_toe_cm[0]
        if self._skeleton_baseline_x_cm is None:
            self._skeleton_baseline_x_cm = cur_x
        if self._skeleton_baseline_hip_x is None and hip_cm:
            self._skeleton_baseline_hip_x = hip_cm[0]
        if self._skeleton_baseline_ankle_y is None:
            self._skeleton_baseline_ankle_y = curr_ankle_y_px

        baseline_x_for_vis = self._skeleton_baseline_x_cm
        toe_moved = self._forward_delta(cur_x, baseline_x_for_vis)
        raw_toe_moved = toe_moved  # 保存原始值（重置前）
        raw_hip_moved = self._forward_delta(hip_cm[0], self._skeleton_baseline_hip_x) if (hip_cm and self._skeleton_baseline_hip_x is not None) else 0  # 保存重置前的 hip_moved
        self._skeleton_last_hip_moved_cm = raw_hip_moved
        stable_before_update = self._skeleton_ready_stable
        stable_reset_after_ready = False
        jitter_outlier_after_ready = False
        if abs(toe_moved) < 6.0:
            self._skeleton_baseline_x_cm = (0.9 * self._skeleton_baseline_x_cm) + (0.1 * cur_x)
            if hip_cm and self._skeleton_baseline_hip_x:
                self._skeleton_baseline_hip_x = (0.9 * self._skeleton_baseline_hip_x) + (0.1 * hip_cm[0])
            if self._skeleton_baseline_ankle_y:
                self._skeleton_baseline_ankle_y = (0.9 * self._skeleton_baseline_ankle_y) + (0.1 * curr_ankle_y_px)
            self._skeleton_ready_stable += 1
        else:
            if not (self._skeleton_ready_stable >= 10 and toe_moved > 0):
                # A long-stable foot can momentarily jump backward because of pose jitter.
                # Only treat that stable break as takeoff when the body is clearly moving
                # forward and the toe jump is not an extreme backward outlier.
                stable_reset_after_ready = (
                    stable_before_update > 35
                    and raw_hip_moved > 50.0
                    and raw_toe_moved > -15.0
                )
                jitter_outlier_after_ready = (
                    stable_before_update > 35
                    and raw_toe_moved <= -15.0
                    and not stable_reset_after_ready
                )
                if stable_reset_after_ready or stable_before_update <= 35:
                    self._skeleton_ready_stable = 0
                    self._skeleton_baseline_x_cm = cur_x
                    if hip_cm:
                        self._skeleton_baseline_hip_x = hip_cm[0]
                    self._skeleton_baseline_ankle_y = curr_ankle_y_px
                    toe_moved = 0

        hip_moved = self._forward_delta(hip_cm[0], self._skeleton_baseline_hip_x) if (hip_cm and self._skeleton_baseline_hip_x is not None) else 0
        ankle_lifted_px = (self._skeleton_baseline_ankle_y - curr_ankle_y_px) if self._skeleton_baseline_ankle_y else 0
        ankle_lifted = ankle_lifted_px > 30.0

        # 防止单帧脚尖关键点突刺：例如右向左视频中脚尖实际没动，
        # 但某一帧 front_toe_cm 突然越过 10cm 阈值，且脚踝没有抬高。
        # 这类帧不应触发“hip_moved + toe_moved”起跳条件，也不进入倒推候选缓存。
        prev_toe_moved_max = 0.0
        if self._skeleton_front_toe_hist:
            prev_toe_moved_max = max(
                self._forward_delta(float(item[0]), baseline_x_for_vis)
                for item in list(self._skeleton_front_toe_hist)[-3:]
            )
        toe_spike_without_lift = (
            toe_moved > 10.0
            and prev_toe_moved_max < 6.0
            and (toe_moved - prev_toe_moved_max) > 6.0
            and ankle_lifted_px < 12.0
        )
        if toe_spike_without_lift:
            jitter_outlier_after_ready = True

        is_taking_off = False
        takeoff_reason = ""
        takeoff_candidate_mode = None
        takeoff_delay_frames = 0
        if stable_reset_after_ready:
            is_taking_off = True
            takeoff_reason = f"stable_before_update={stable_before_update}>35 后突然置0"
            takeoff_delay_frames = 1
            # 脚尖在真实起跳瞬间可能因抬脚发生轻微回缩；这种强回缩帧本身更接近离地瞬间。
            # 但此时脚尖坐标可能已回缩，所以只把帧号/画面推进到当前帧，测距点仍沿用上一帧稳定脚尖。
            if raw_toe_moved <= -8.0:
                takeoff_candidate_mode = "synthetic_current_frame"
        elif self._skeleton_toe_missing_counter >= 3 and toe_moved > -3.0:
            is_taking_off = True
            takeoff_reason = f"toe_missing={self._skeleton_toe_missing_counter}>=3, toe_moved={toe_moved:.1f}>-3.0"
        elif self._skeleton_ready_stable > 35 and hip_moved > 100.0 and ankle_lifted_px > 30.0:
            is_taking_off = True
            takeoff_reason = f"hip_moved={hip_moved:.1f}>100.0 and ankle_lifted_px={ankle_lifted_px:.1f}>30.0"
        if toe_moved > max(self.config.trigger_move_cm, 30.0):
            is_taking_off = True
            takeoff_reason = f"toe_moved={toe_moved:.1f} > max(trigger_move_cm={self.config.trigger_move_cm}, 30.0)"
        elif hip_moved > 35.0 and toe_moved > 10.0 and not toe_spike_without_lift:
            is_taking_off = True
            takeoff_reason = f"hip_moved={hip_moved:.1f}>35.0 and toe_moved={toe_moved:.1f}>10.0"
        elif hip_moved > 35.0 and toe_moved > 10.0 and toe_spike_without_lift:
            takeoff_reason = (f"忽略脚尖单帧突刺: toe_moved={toe_moved:.1f}, "
                              f"prev_max={prev_toe_moved_max:.1f}, ankle_lifted_px={ankle_lifted_px:.1f}<12.0")
        # 结合脚踝抬高、重心前冲，以及离地腾空时脚尖特有的"爆发性突变回缩"，过滤蓄力大动作
        elif ankle_lifted and (toe_moved > 3.0 or (hip_moved > 70.0 and -12.0 < toe_moved < -2.0)):
            is_taking_off = True
            if toe_moved > 3.0:
                takeoff_reason = f"检测到离地爆发：ankle_lifted={ankle_lifted_px:.1f}px 且 脚尖前移(toe_moved={toe_moved:.1f}>3.0)"
                takeoff_candidate_mode = "current"
                takeoff_delay_frames = 2
            else:
                takeoff_reason = f"检测到离地爆发：ankle_lifted={ankle_lifted_px:.1f}px，重心前冲(hip_moved={hip_moved:.1f}>70) 且 脚尖腾空回缩(-12.0<toe_moved={toe_moved:.1f}<-2.0)"

        trigger_count = self._skeleton_jump_trigger_counter
        stable_gate_passed = self._skeleton_ready_stable >= 10 or stable_reset_after_ready
        self._skeleton_jump_trigger_counter = (trigger_count + 1 if is_taking_off else 0) if stable_gate_passed else 0

        if self.debug:
            self._log("DEBUG_TAKEOFF_PARAMS", f"帧{frame_idx}: toe_moved={toe_moved:.1f}(raw={raw_toe_moved:.1f}) | hip_moved={hip_moved:.1f}(raw={raw_hip_moved:.1f}) | "
                      f"ankle_lifted_px={ankle_lifted_px:.1f}(lifted={ankle_lifted}) | toe_missing={self._skeleton_toe_missing_counter} | "
                      f"trigger={self._skeleton_jump_trigger_counter}/1 | stable={self._skeleton_ready_stable} | stable_before={stable_before_update} | "
                      f"stable_reset_after_ready={stable_reset_after_ready} | is_taking_off={is_taking_off} | reason={takeoff_reason or '无'}")
            if is_taking_off:
                self._save_debug_takeoff_toe_moved_image(
                    frame_idx, kpts, front_toe_xy, front_toe_cm, baseline_x_for_vis,
                    toe_moved, raw_toe_moved, hip_moved, raw_hip_moved,
                    ankle_lifted_px, takeoff_reason, stable_before_update, self._skeleton_ready_stable,
                )

        if (not takeoff_signal and not jitter_outlier_after_ready
                and front_toe_cm is not None and front_toe_xy is not None
                and self.calibrator.in_mat(front_toe_cm)):
            self._skeleton_front_toe_hist.append((float(front_toe_cm[0]), float(front_toe_cm[1]), float(front_toe_xy[0]), float(front_toe_xy[1])))

        if is_taking_off and takeoff_delay_frames > 0:
            if takeoff_candidate_mode == "current":
                self._save_skeleton_takeoff_candidate(frame_idx, kpts, front_toe_cm, front_toe_xy, toe_xy, ankle_xy)
            elif takeoff_candidate_mode == "synthetic_current_frame":
                self._save_skeleton_synthetic_takeoff_candidate(frame_idx, kpts, ankle_xy)
            else:
                self._save_skeleton_takeoff_candidate(frame_idx, kpts, front_toe_cm, front_toe_xy, toe_xy, ankle_xy)
            self._skeleton_pending_takeoff = {
                "first_frame": frame_idx,
                "confirm_frame": frame_idx + takeoff_delay_frames,
                "delay": takeoff_delay_frames,
                "reason": takeoff_reason,
            }
            self._skeleton_jump_trigger_counter = 0
            if self.debug:
                self._log("DEBUG_TAKEOFF_PARAMS", f"帧{frame_idx}: 起跳候选延迟确认 | delay={takeoff_delay_frames} | confirm_frame={frame_idx + takeoff_delay_frames} | reason={takeoff_reason}")
            return

        if self._skeleton_jump_trigger_counter < 1:
            # 未起跳：保存本帧数据作为下一帧起跳时的倒推候选
            self._save_skeleton_takeoff_candidate(frame_idx, kpts, front_toe_cm, front_toe_xy, toe_xy, ankle_xy)
            return

        if takeoff_candidate_mode == "current":
            self._save_skeleton_takeoff_candidate(frame_idx, kpts, front_toe_cm, front_toe_xy, toe_xy, ankle_xy)
        elif takeoff_candidate_mode == "synthetic_current_frame":
            self._save_skeleton_synthetic_takeoff_candidate(frame_idx, kpts, ankle_xy)
        self._confirm_skeleton_takeoff(frame_idx, kpts, front_toe_cm, front_toe_xy, toe_xy, ankle_xy, takeoff_reason)


    def _handle_jumping_skeleton(self, frame_idx, heel_l_xy, heel_r_xy, heel_l_cm, heel_r_cm, kpts=None):
        self._skeleton_jump_counter += 1
        if heel_l_xy is not None or heel_r_xy is not None:
            all_kpts = list(self.pose_estimator.all_kpts_list) if self.pose_estimator.all_kpts_list else []
            frame_img = self._prev_frame_img.copy() if self._prev_frame_img is not None else None
            for side, heel_xy, heel_cm in [("l", heel_l_xy, heel_l_cm), ("r", heel_r_xy, heel_r_cm)]:
                if heel_xy is None or heel_cm is None:
                    continue
                self._skeleton_foot_hist[side]["y_px"].append(heel_xy[1])
                self._skeleton_foot_hist[side]["x_cm"].append(self._forward_delta(heel_cm[0], (self.takeoff_x_cm or 0.0)))
                self._skeleton_foot_hist[side]["y_cm"].append(heel_cm[1])
                self._skeleton_foot_hist[side]["xy_px"].append((float(heel_xy[0]), float(heel_xy[1])))
                self._skeleton_foot_hist[side]["frame_idx"].append(frame_idx)
                self._skeleton_foot_hist[side]["frame_img"].append(frame_img)
                self._skeleton_foot_hist[side]["kpts"].append(kpts)
                self._skeleton_foot_hist[side]["all_kpts_list"].append(all_kpts)

        detected_landing = False
        landing_xy_candidate = None
        landing_frame_candidate = None
        landing_frame_img_candidate = None
        landing_kpts_candidate = None
        landing_all_kpts_candidate = []
        landing_reason = ""
        contact_l = contact_r = None  # 当前接触距离，debug 日志使用

        # 用户标注的“滞空帧数”以脚尖离地触发帧 -> 脚后跟首次触地帧计算，
        # 而正式起跳图仍为触发帧倒推 N 帧；所以最早落地帧要基于 trigger_frame。
        landing_base_frame = self.takeoff_trigger_frame
        if landing_base_frame is None and self.takeoff_frame is not None:
            landing_base_frame = self.takeoff_frame + max(0, int(getattr(self.config, "takeoff_backtrack_frames", 2)))
        min_landing_frame = (landing_base_frame + self.config.min_flight_frames
                             if landing_base_frame is not None else None)
        contact_l, contact_l_xy, contact_l_frame, contact_l_img, contact_l_pose = self._skeleton_detect_contact(self._skeleton_foot_hist["l"])
        contact_r, contact_r_xy, contact_r_frame, contact_r_img, contact_r_pose = self._skeleton_detect_contact(self._skeleton_foot_hist["r"])

        if min_landing_frame is not None:
            if contact_l is not None and (contact_l_frame is None or contact_l_frame < min_landing_frame):
                contact_l = None
                contact_l_xy = None
                contact_l_frame = None
                contact_l_img = None
                contact_l_pose = None
            if contact_r is not None and (contact_r_frame is None or contact_r_frame < min_landing_frame):
                contact_r = None
                contact_r_xy = None
                contact_r_frame = None
                contact_r_img = None
                contact_r_pose = None

        if contact_l is not None or contact_r is not None:
            detected_landing = True
            if contact_l is not None and contact_r is not None:
                use_left = contact_l <= contact_r
                landing_reason = f"双脚触地: L_x={contact_l:.1f}cm, R_x={contact_r:.1f}cm, 取min={min(contact_l, contact_r):.1f}cm"
            else:
                use_left = contact_l is not None
                landing_reason = f"单脚触地: {'左脚' if use_left else '右脚'}"
            contact_side = "l" if use_left else "r"
            if use_left:
                landing_xy_candidate = contact_l_xy
                landing_frame_candidate = contact_l_frame
                landing_frame_img_candidate = contact_l_img
                landing_kpts_candidate, landing_all_kpts_candidate = contact_l_pose or (None, [])
            else:
                landing_xy_candidate = contact_r_xy
                landing_frame_candidate = contact_r_frame
                landing_frame_img_candidate = contact_r_img
                landing_kpts_candidate, landing_all_kpts_candidate = contact_r_pose or (None, [])

            # 左向右样本中，local-peak 三帧法在最小滞空帧处偶尔会把“触地前一帧”当作峰值；
            # 若正好是允许的最早落地帧，则保守使用当前确认帧，避免半空帧被提前作为落地图。
            if (not self.is_rtl and min_landing_frame is not None
                    and landing_frame_candidate == min_landing_frame
                    and frame_idx == min_landing_frame + 1):
                hist = self._skeleton_foot_hist[contact_side]
                if hist["frame_idx"] and hist["frame_idx"][-1] == frame_idx:
                    landing_xy_candidate = hist["xy_px"][-1]
                    landing_frame_candidate = frame_idx
                    landing_frame_img_candidate = hist["frame_img"][-1]
                    landing_kpts_candidate = hist["kpts"][-1]
                    landing_all_kpts_candidate = hist["all_kpts_list"][-1]
                    landing_reason += "，最早帧local-peak改用当前确认帧"

        # 对当前这批视频，落地帧定义为“起跳后达到最短滞空间隔后，第一次可靠后跟出现”。
        # 这样不会早于 min_flight_frames，也避免原先局部 Y 峰值法把落地推迟数帧。
        if (not detected_landing and min_landing_frame is not None and frame_idx >= min_landing_frame + 1):
            latest_candidates = []
            for side in ("l", "r"):
                hist = self._skeleton_foot_hist[side]
                if not hist["frame_idx"]:
                    continue
                latest_frame_idx = hist["frame_idx"][-1]
                # 到达用户标注的最短滞空帧时，MediaPipe 可能正好丢 1 帧脚跟点；
                # 允许使用上一帧可靠脚跟位置绑定到当前落地图，避免把落地推迟到下一帧。
                use_synthetic_min_frame = (frame_idx == min_landing_frame and latest_frame_idx >= frame_idx - 1)
                if latest_frame_idx != frame_idx and not use_synthetic_min_frame:
                    continue
                rel_x = hist["x_cm"][-1] if hist["x_cm"] else None
                xy = hist["xy_px"][-1] if hist["xy_px"] else None
                if rel_x is None or xy is None or rel_x <= 20.0:
                    continue
                frame_img_candidate = (self._prev_frame_img.copy()
                                       if use_synthetic_min_frame and self._prev_frame_img is not None
                                       else hist["frame_img"][-1])
                latest_candidates.append((rel_x, side, xy, frame_img_candidate, hist["kpts"][-1], hist["all_kpts_list"][-1]))
            if latest_candidates:
                latest_candidates.sort(key=lambda item: item[0])
                rel_x, side, xy, img0, kpts0, all0 = latest_candidates[0]
                detected_landing = True
                landing_xy_candidate = xy
                landing_frame_candidate = frame_idx
                landing_frame_img_candidate = img0
                landing_kpts_candidate = kpts0
                landing_all_kpts_candidate = all0
                landing_reason = f"最早可靠后跟帧: {'左脚' if side == 'l' else '右脚'}_x={rel_x:.1f}cm, min_frame={min_landing_frame}"

        elapsed_from_takeoff = (frame_idx - landing_base_frame
                                if landing_base_frame is not None else self._skeleton_jump_counter)
        if not detected_landing and elapsed_from_takeoff >= self.config.max_jump_frames:
            self.foul_detector.reason = self.foul_detector.reason or "落地超时未检测到可靠后跟触地，成绩无效"
            self.state = "LANDED"
            self.landing_frame = frame_idx
            self.landing_pt_px = None
            self.landing_pt_xy = None
            self.landing_x_cm = self.takeoff_x_cm
            self._skeleton_landing_x_cm = self.landing_x_cm
            self.final_distance_cm = 0.0
            self._landing_frame_img = self._prev_frame_img.copy() if self._prev_frame_img is not None else None
            self._landing_kpts = kpts
            self._landing_all_kpts_list = list(self.pose_estimator.all_kpts_list) if self.pose_estimator.all_kpts_list else []
            self._log("LAND", f"落地超时无可靠后跟触地: elapsed={elapsed_from_takeoff} >= max_jump_frames={self.config.max_jump_frames}，本次成绩无效")
            return

        if self.debug:
            heel_l_x = self._skeleton_foot_hist["l"]["x_cm"][-1] if self._skeleton_foot_hist["l"]["x_cm"] else None
            heel_r_x = self._skeleton_foot_hist["r"]["x_cm"][-1] if self._skeleton_foot_hist["r"]["x_cm"] else None
            contact_l_info = f"L_contact={contact_l:.1f}@{contact_l_frame}" if contact_l is not None else "L_contact=None"
            contact_r_info = f"R_contact={contact_r:.1f}@{contact_r_frame}" if contact_r is not None else "R_contact=None"
            self._log("DEBUG_LANDING_PARAMS", f"帧{frame_idx}: jump_cnt={self._skeleton_jump_counter} | "
                      f"elapsed={elapsed_from_takeoff} | min_frame={min_landing_frame} | "
                      f"heel_L_x={heel_l_x}cm | heel_R_x={heel_r_x}cm | "
                      f"{contact_l_info} | {contact_r_info} | "
                      f"min_flight={self.config.min_flight_frames} | detected_landing={detected_landing} | reason={landing_reason or '无'}")

        if not detected_landing:
            return

        landing_x_for_dist = None
        if landing_xy_candidate is not None:
            temp_cm_pt = self.calibrator.transform_to_mat_cm(landing_xy_candidate)
            if temp_cm_pt is not None and 0 < temp_cm_pt[0] < (self.calibrator.mat_length_cm + 2.0):
                landing_x_for_dist = temp_cm_pt[0]

        if landing_x_for_dist is None:
            self._log("LAND", "无法获取骨架落地位置，继续等待可靠触地")
            return

        # 落地点 X：鞋跟厚度补偿；之后统一用于输出/复算/成绩
        landing_x_for_dist = self._apply_landing_offset(landing_x_for_dist)
        temp_dist = self._distance_between_x(self.takeoff_x_cm if self.takeoff_x_cm is not None else 0.0, landing_x_for_dist)

        if temp_dist < 50.0:
            self._log("LAND", f"疑似假落地(骨架, 距离{temp_dist:.1f}cm 过短)，继续等待")
            return

        self.state = "LANDED"
        landing_frame_used = landing_frame_candidate if landing_frame_candidate is not None else frame_idx
        if self.debug:
            self._log("DEBUG_LANDING", f"帧{landing_frame_used}(<-{frame_idx}): 落地触发 | reason={landing_reason} | landing_x={landing_x_for_dist:.1f}cm | jump_counter={self._skeleton_jump_counter}")
        self._log("LAND", f"落地触发(骨架), 确认帧={frame_idx}, 落地帧={landing_frame_used}, landing_x={landing_x_for_dist:.1f}, 距离={temp_dist:.1f}cm")
        self.landing_frame = landing_frame_used
        self.landing_pt_px = landing_xy_candidate
        self.landing_pt_xy = self.landing_pt_px
        self.landing_x_cm = landing_x_for_dist
        self._skeleton_landing_x_cm = self.landing_x_cm  # 骨骼修正值（YOLO 覆盖前保存）
        self.final_distance_cm = max(0.0, temp_dist)
        self._landing_frame_img = (landing_frame_img_candidate.copy()
                                   if landing_frame_img_candidate is not None else
                                   (self._prev_frame_img.copy() if self._prev_frame_img is not None else None))
        self._landing_kpts = landing_kpts_candidate if landing_kpts_candidate is not None else kpts
        self._landing_all_kpts_list = landing_all_kpts_candidate or (list(self.pose_estimator.all_kpts_list) if self.pose_estimator.all_kpts_list else [])
    def _save_debug_takeoff_toe_moved_image(self, frame_idx, kpts, front_toe_xy, front_toe_cm,
                                             baseline_x_cm, toe_moved, raw_toe_moved,
                                             hip_moved, raw_hip_moved, ankle_lifted_px,
                                             takeoff_reason, stable_before_update, stable_now):
        """debug 模式下保存起跳触发帧，直观显示 toe_moved 的基准点到当前 toe 点距离。"""
        if (not self.debug or not self.images_dir or self._debug_takeoff_toe_moved_saved
                or self._prev_frame_img is None or front_toe_xy is None or front_toe_cm is None
                or baseline_x_cm is None or self.calibrator.H_mat2img is None):
            return

        img = self._prev_frame_img.copy()
        H = self.calibrator.H_mat2img
        toe_y_cm = float(front_toe_cm[1])
        baseline_pt_mat = np.array([[[float(baseline_x_cm), toe_y_cm]]], dtype=np.float32)
        baseline_pt_img = cv2.perspectiveTransform(baseline_pt_mat, H).reshape(-1, 2)[0]
        p_base = (int(round(baseline_pt_img[0])), int(round(baseline_pt_img[1])))
        p_toe = (int(round(front_toe_xy[0])), int(round(front_toe_xy[1])))

        self.renderer.draw_mat_outline(img, self.calibrator)
        self.renderer.draw_x_line(img, H, self.calibrator.mat_width_cm,
                                  self._takeoff_line_x_cm(), (255, 255, 255), thickness=3, label="limit")
        self.renderer.draw_x_line(img, H, self.calibrator.mat_width_cm,
                                  baseline_x_cm, (255, 180, 0), thickness=2, label="baseline toe X")

        # 人体关键点：主检测 + 多人列表，便于排查关键点是否跳变。
        if kpts is not None:
            self.renderer.draw_pose(img, kpts, self.pose_estimator.mp_connections, color=(0, 255, 0))
            self.renderer.draw_feet(img, self._get_feet, kpts)
        for pk in (self.pose_estimator.all_kpts_list or []):
            self.renderer.draw_pose(img, pk, self.pose_estimator.mp_connections, color=(0, 180, 0))
            self.renderer.draw_feet(img, self._get_feet, pk)

        # toe_moved 可视化：同一垫子 Y 位置上，从 READY 基准 toe X 连到当前前脚尖。
        cv2.arrowedLine(img, p_base, p_toe, (0, 255, 255), 4, cv2.LINE_AA, tipLength=0.08)
        cv2.circle(img, p_base, 8, (255, 180, 0), -1, cv2.LINE_AA)
        cv2.circle(img, p_toe, 9, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(img, "baseline", (p_base[0] + 8, p_base[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 180, 0), 2, cv2.LINE_AA)
        cv2.putText(img, "front_toe", (p_toe[0] + 8, p_toe[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

        text_y = 70
        img = self.renderer.put_text_chinese(img, f"DEBUG 起跳触发帧: {frame_idx}", (40, text_y), (0, 255, 255), size=38)
        text_y += 46
        img = self.renderer.put_text_chinese(
            img,
            f"toe_moved={toe_moved:.1f}cm (raw={raw_toe_moved:.1f})  baseline_x={baseline_x_cm:.1f}cm  toe_x={front_toe_cm[0]:.1f}cm",
            (40, text_y), (0, 255, 255), size=26,
        )
        text_y += 34
        img = self.renderer.put_text_chinese(
            img,
            f"hip_moved={hip_moved:.1f}cm (raw={raw_hip_moved:.1f})  ankle_lifted_px={ankle_lifted_px:.1f}  stable={stable_now}  stable_before={stable_before_update}",
            (40, text_y), (0, 255, 255), size=24,
        )
        text_y += 32
        if takeoff_reason:
            img = self.renderer.put_text_chinese(img, f"reason={takeoff_reason}", (40, text_y), (0, 255, 255), size=24)

        filename = os.path.join(self.images_dir, "debug_takeoff_toe_moved.jpeg")
        imwrite_safe(filename, img)
        self._debug_takeoff_toe_moved_saved = True
        self._log("SAVE", f"debug起跳toe_moved可视化已保存: {filename}")

    def _save_takeoff_image(self, frame, kpts=None, all_kpts_list=None):
        """保存起跳帧图像（含骨架、垫子轮廓、起跳线标注）。"""
        if not self.images_dir or self._takeoff_saved:
            return
        all_kpts = all_kpts_list if all_kpts_list is not None else (self.pose_estimator.all_kpts_list or [])
        img = frame.copy()

        H = self.calibrator.H_mat2img
        mw = self.calibrator.mat_width_cm

        # 垫子轮廓
        self.renderer.draw_mat_outline(img, self.calibrator)

        # 标准起跳线（白色）— limit
        self.renderer.draw_x_line(img, H, mw, self._takeoff_line_x_cm(), (255, 255, 255), thickness=3, label="limit")

        # 骨架和脚点
        if kpts is not None:
            self.renderer.draw_pose(img, kpts, self.pose_estimator.mp_connections, color=(0, 255, 0))
            self.renderer.draw_feet(img, self._get_feet, kpts)
        for pk in all_kpts:
            self.renderer.draw_pose(img, pk, self.pose_estimator.mp_connections, color=(0, 255, 0))
            self.renderer.draw_feet(img, self._get_feet, pk)

        # fixed corrected 起跳线（黄色）— 骨骼关键点 + offset 修正
        fixed_corrected_x = self._skeleton_takeoff_x_cm or self.takeoff_x_cm
        if fixed_corrected_x is not None:
            self.renderer.draw_x_point(img, H, mw, fixed_corrected_x, (0, 255, 255), radius=8, label="fixed corrected", label_offset_y=20)
            self.renderer.draw_x_line(img, H, mw, fixed_corrected_x, (0, 255, 255), thickness=2)

        # yolo corrected 起跳线（绿色）— YOLO 实例分割修正
        if self._yolo_takeoff_x_cm is not None:
            self.renderer.draw_x_point(img, H, mw, self._yolo_takeoff_x_cm, (0, 255, 0), radius=8, label="yolo corrected", label_offset_y=-8)
            self.renderer.draw_x_line(img, H, mw, self._yolo_takeoff_x_cm, (0, 255, 0), thickness=2)

        # mog_corrected 起跳线（紫色）— MOG2 差分法修正
        if self._mog_takeoff_x_cm is not None:
            self.renderer.draw_x_point(img, H, mw, self._mog_takeoff_x_cm, (255, 0, 255), radius=8, label="mog_corrected", label_offset_y=-8)
            self.renderer.draw_x_line(img, H, mw, self._mog_takeoff_x_cm, (255, 0, 255), thickness=2)

        # 左上角文字
        img = self.renderer.put_text_chinese(img, "起跳帧", (50, 80), (0, 255, 255), size=50)
        img = self.renderer.put_text_chinese(img,
            f"fixed corrected: {fixed_corrected_x:.1f} cm (offset={self.takeoff_display_offset_cm:.1f})",
            (50, 140), (255, 255, 0), size=25)
        text_y = 175
        if self._yolo_takeoff_x_cm is not None:
            img = self.renderer.put_text_chinese(img,
                f"yolo corrected: {self._yolo_takeoff_x_cm:.1f} cm",
                (50, text_y), (0, 255, 0), size=25)
            text_y += 35
        if self._mog_takeoff_x_cm is not None:
            img = self.renderer.put_text_chinese(img,
                f"mog_corrected: {self._mog_takeoff_x_cm:.1f} cm",
                (50, text_y), (255, 0, 255), size=25)
            text_y += 35
        if self.foul_detector.reason:
            img = self.renderer.put_text_chinese(img, f"犯规: {self.foul_detector.reason}", (50, text_y), (0, 0, 255), size=40)

        filename = self._image_output_path("takeoff.jpeg")
        imwrite_safe(filename, img)
        self._log("SAVE", f"起跳图片已保存: {filename}")
        self._takeoff_saved = True

    def _save_landed_image(self, frame, kpts):
        if self.state != "LANDED" or self._landed_saved:
            return

        img = (self._landing_frame_img if self._landing_frame_img is not None else frame).copy()

        H = self.calibrator.H_mat2img
        mw = self.calibrator.mat_width_cm

        self.renderer.draw_mat_outline(img, self.calibrator)

        # 标准起跳线（白色）— limit
        self.renderer.draw_x_line(img, H, mw, self._takeoff_line_x_cm(), (255, 255, 255), thickness=3, label="limit")

        # 骨架：优先画真正落地候选帧的关键点，而不是后一帧确认帧。
        draw_kpts = self._landing_kpts if self._landing_kpts is not None else kpts
        draw_all_kpts = self._landing_all_kpts_list or []
        if draw_kpts is not None:
            self.renderer.draw_pose(img, draw_kpts, self.pose_estimator.mp_connections, color=(0, 255, 0))
        for pk in draw_all_kpts:
            self.renderer.draw_pose(img, pk, self.pose_estimator.mp_connections, color=(0, 255, 0))

        # fixed corrected 落地线（红色）— 骨骼关键点 + offset 修正
        fixed_corrected_x = self._skeleton_landing_x_cm or self.landing_x_cm
        if fixed_corrected_x is not None:
            self.renderer.draw_x_point(img, H, mw, fixed_corrected_x, (0, 0, 255), radius=8, label="fixed corrected", label_offset_y=20)
            self.renderer.draw_x_line(img, H, mw, fixed_corrected_x, (0, 0, 255), thickness=2)

        # yolo corrected 落地线（绿色）— YOLO 实例分割修正
        if self._yolo_landing_x_cm is not None:
            self.renderer.draw_x_point(img, H, mw, self._yolo_landing_x_cm, (0, 255, 0), radius=8, label="yolo corrected", label_offset_y=-8)
            self.renderer.draw_x_line(img, H, mw, self._yolo_landing_x_cm, (0, 255, 0), thickness=2)

        # mog_corrected 落地线（紫色）— MOG2 差分法修正
        if self._mog_landing_x_cm is not None:
            self.renderer.draw_x_point(img, H, mw, self._mog_landing_x_cm, (255, 0, 255), radius=8, label="mog_corrected", label_offset_y=-8)
            self.renderer.draw_x_line(img, H, mw, self._mog_landing_x_cm, (255, 0, 255), thickness=2)

        # 左上角文字
        img = self.renderer.put_text_chinese(img, "落地帧", (50, 80), (0, 255, 255), size=50)
        img = self.renderer.put_text_chinese(img,
            f"fixed corrected: {fixed_corrected_x:.1f} cm (offset={self.landing_offset_cm:.1f})",
            (50, 140), (0, 0, 255), size=25)
        text_y = 175
        if self._yolo_landing_x_cm is not None:
            img = self.renderer.put_text_chinese(img,
                f"yolo corrected: {self._yolo_landing_x_cm:.1f} cm",
                (50, text_y), (0, 255, 0), size=25)
            text_y += 35
        if self._mog_landing_x_cm is not None:
            img = self.renderer.put_text_chinese(img,
                f"mog_corrected: {self._mog_landing_x_cm:.1f} cm",
                (50, text_y), (255, 0, 255), size=25)
            text_y += 35
        if self.foul_detector.reason:
            img = self.renderer.put_text_chinese(img, f"INVALID: {self.foul_detector.reason}", (50, text_y), (0, 0, 255), size=50)

        filename = self._image_output_path("landed.jpeg")
        imwrite_safe(filename, img)
        self._log("SAVE", f"落地图片已保存: {filename}")
        self._landed_saved = True
        self._save_payload()

    def _save_score_image(self, frame):
        """保存成绩图像，标注修正后起跳点/落地线、标准起跳线及成绩文本。"""
        if not self.images_dir or self._score_saved:
            return
        if self.takeoff_x_cm is None or self.landing_x_cm is None:
            return

        img = frame.copy()
        H = self.calibrator.H_mat2img
        mw = self.calibrator.mat_width_cm

        self.renderer.draw_mat_outline(img, self.calibrator)

        # 标准起跳线（白色）— limit
        self.renderer.draw_x_line(img, H, mw, self._takeoff_line_x_cm(), (255, 255, 255), thickness=3, label="limit")

        # fixed corrected 起跳线（黄色）— 骨骼关键点 + offset 修正
        to_fixed = self._skeleton_takeoff_x_cm or self.takeoff_x_cm
        if to_fixed is not None:
            self.renderer.draw_x_point(img, H, mw, to_fixed, (0, 255, 255), radius=8, label="fixed corrected")
            self.renderer.draw_x_line(img, H, mw, to_fixed, (0, 255, 255), thickness=2)

        # yolo corrected 起跳线（绿色）— 仅 YOLO 模式显示；--diff 模式不显示 yolo 字样
        if not self.config.enable_diff and self._yolo_takeoff_x_cm is not None:
            self.renderer.draw_x_point(img, H, mw, self._yolo_takeoff_x_cm, (0, 255, 0), radius=8, label="yolo corrected", label_offset_y=48)
            self.renderer.draw_x_line(img, H, mw, self._yolo_takeoff_x_cm, (0, 255, 0), thickness=2)

        # diff-mog2-fixed 起跳线（紫色）— MOG2 差分修正最终采用点
        if self._mog_takeoff_x_cm is not None:
            self.renderer.draw_x_point(img, H, mw, self._mog_takeoff_x_cm, (255, 0, 255), radius=8, label="diff-mog2-fixed", label_offset_y=48)
            self.renderer.draw_x_line(img, H, mw, self._mog_takeoff_x_cm, (255, 0, 255), thickness=2)

        # fixed corrected 落地线（红色）— 骨骼关键点 + offset 修正
        ld_fixed = self._skeleton_landing_x_cm or self.landing_x_cm
        if ld_fixed is not None:
            self.renderer.draw_x_point(img, H, mw, ld_fixed, (0, 0, 255), radius=8, label="fixed corrected")
            self.renderer.draw_x_line(img, H, mw, ld_fixed, (0, 0, 255), thickness=2)

        # yolo corrected 落地线（绿色）— 仅 YOLO 模式显示；--diff 模式不显示 yolo 字样
        if not self.config.enable_diff and self._yolo_landing_x_cm is not None:
            self.renderer.draw_x_point(img, H, mw, self._yolo_landing_x_cm, (0, 255, 0), radius=8, label="yolo corrected", label_offset_y=48)
            self.renderer.draw_x_line(img, H, mw, self._yolo_landing_x_cm, (0, 255, 0), thickness=2)

        # diff-mog2-fixed 落地线（紫色）— MOG2 差分修正最终采用点
        if self._mog_landing_x_cm is not None:
            self.renderer.draw_x_point(img, H, mw, self._mog_landing_x_cm, (255, 0, 255), radius=8, label="diff-mog2-fixed", label_offset_y=48)
            self.renderer.draw_x_line(img, H, mw, self._mog_landing_x_cm, (255, 0, 255), thickness=2)

        # 测量连线（绿色）— 连接最终采用的起跳点和落地点；--diff 后这里就是 MOG2 修正成绩
        self.renderer.draw_measurement_line(img, H, mw, self.takeoff_x_cm, self.landing_x_cm)

        # 左上角文字
        score_text = f"成绩: {self.final_distance_cm:.1f} cm"
        if self._mog_takeoff_x_cm is not None or self._mog_landing_x_cm is not None:
            mog_to = f"{self._mog_takeoff_x_cm:.1f}" if self._mog_takeoff_x_cm is not None else "N/A"
            mog_ld = f"{self._mog_landing_x_cm:.1f}" if self._mog_landing_x_cm is not None else "N/A"
            to_text = f"起跳: {self.takeoff_x_cm:.1f} cm | fixed: {to_fixed:.1f} | diff-mog2-fixed: {mog_to}"
            ld_text = f"落地: {self.landing_x_cm:.1f} cm | fixed: {ld_fixed:.1f} | diff-mog2-fixed: {mog_ld}"
        elif self._yolo_takeoff_x_cm is not None or self._yolo_landing_x_cm is not None:
            yolo_to = f"{self._yolo_takeoff_x_cm:.1f}" if self._yolo_takeoff_x_cm is not None else "N/A"
            yolo_ld = f"{self._yolo_landing_x_cm:.1f}" if self._yolo_landing_x_cm is not None else "N/A"
            to_text = f"起跳: {self.takeoff_x_cm:.1f} cm | fixed: {to_fixed:.1f} | yolo: {yolo_to}"
            ld_text = f"落地: {self.landing_x_cm:.1f} cm | fixed: {ld_fixed:.1f} | yolo: {yolo_ld}"
        else:
            to_text = f"起跳: {self.takeoff_x_cm:.1f} cm | fixed: {to_fixed:.1f}"
            ld_text = f"落地: {self.landing_x_cm:.1f} cm | fixed: {ld_fixed:.1f}"
        img = self.renderer.put_text_chinese(img, score_text, (50, 80), (0, 255, 0), size=50)
        img = self.renderer.put_text_chinese(img, to_text, (50, 145), (255, 255, 0), size=22)
        img = self.renderer.put_text_chinese(img, ld_text, (50, 180), (0, 0, 255), size=22)
        if self.ocr_reader is not None or self.stream_mode:
            to_time = (self._frame_time_info(self.takeoff_frame) or {}).get("timecode", "") if self.takeoff_frame is not None else ""
            ld_time = (self._frame_time_info(self.landing_frame) or {}).get("timecode", "") if self.landing_frame is not None else ""
            img = self.renderer.put_text_chinese(img, f"time: takeoff={to_time} landing={ld_time}", (50, 215), (255, 255, 255), size=22)

        filename = self._image_output_path("score.jpeg")
        imwrite_safe(filename, img)
        self._log("SAVE", f"成绩图片已保存: {filename}")
        self._score_saved = True

    def _save_diff_image(self):
        """分步保存差分/YOLO 各阶段过程照片。

        传统差分 → images/diff/；YOLO seg → images/yolo/
        """
        if self.config.enable_seg:
            target_dir = self.images_yolo_dir
            if not target_dir:
                return
            os.makedirs(target_dir, exist_ok=True)
            stages = [
                ("base_takeoff",          "Stage1-seg-takeoff"),
                ("base_landing",          "Stage1-seg-landing"),
                ("roi_takeoff",           "Stage2-roi-takeoff"),
                ("roi_landing",           "Stage2-roi-landing"),
                ("mask_takeoff",          "Stage3-mask-takeoff"),
                ("mask_landing",          "Stage3-mask-landing"),
                ("takeoff",               "Stage4-takeoff"),
                ("landing",               "Stage4-landing"),
                ("combined",              "Stage4-combined"),
            ]
            prefix = "yolo"
        else:
            if not self.images_diff_dir or not self.diff_detector.has_base_frame:
                return
            target_dir = self.images_diff_dir
            os.makedirs(target_dir, exist_ok=True)
            stages = [
                ("base",                  "Stage1-baseframe"),
                ("roi_takeoff",           "Stage2-roi-takeoff"),
                ("roi_landing",           "Stage2-roi-landing"),
                ("diffmap_takeoff",       "Stage3-edge-takeoff"),
                ("diffmap_landing",       "Stage3-edge-landing"),
                ("takeoff",               "Stage4-takeoff"),
                ("landing",               "Stage4-landing"),
                ("combined",              "Stage4-combined"),
            ]
            prefix = "diff"

        for mode, suffix in stages:
            img = self.diff_detector.render_result_image(mode=mode)
            if img is None:
                self._log("DIFF", f"{prefix}照片[{suffix}] 生成失败，跳过")
                continue
            filename = os.path.join(target_dir, f"{prefix}-{suffix}.jpeg")
            imwrite_safe(filename, img)
        self._log("SAVE", f"{prefix.upper()}各阶段照片已保存: {target_dir}")

    # ---------- 显示合成 ----------
    def _compose_display(self, frame, display_img, kpts):
        if kpts is not None:
            self.renderer.draw_pose(display_img, kpts, self.pose_estimator.mp_connections, color=(0, 255, 0))
            self.renderer.draw_feet(display_img, self._get_feet, kpts)

        self.renderer.draw_mat_outline(display_img, self.calibrator)
        self.renderer.draw_x_line(display_img, self.calibrator.H_mat2img, self.calibrator.mat_width_cm,
                                  self._takeoff_line_x_cm(), (255, 255, 255), thickness=3, label="Limit")
        self.renderer.draw_measurement_line(display_img, self.calibrator.H_mat2img, self.calibrator.mat_width_cm,
                                            self.takeoff_x_cm, self.landing_x_cm)

        if self.takeoff_x_cm is not None:
            ct = (0, 0, 255) if self.foul_detector.reason == "踩线 (Line Violation)" else (0, 255, 255)
            self.renderer.draw_x_line(display_img, self.calibrator.H_mat2img, self.calibrator.mat_width_cm,
                                      self.takeoff_x_cm, ct)

        if self.landing_x_cm is not None:
            self.renderer.draw_x_line(display_img, self.calibrator.H_mat2img, self.calibrator.mat_width_cm,
                                      self.landing_x_cm, (0, 0, 255))
        if self.takeoff_pt_xy is not None:
            cv2.circle(display_img, (int(self.takeoff_pt_xy[0]), int(self.takeoff_pt_xy[1])), 8, (0, 255, 255), 2, lineType=cv2.LINE_AA)
        if self.landing_pt_xy is not None:
            cv2.circle(display_img, (int(self.landing_pt_xy[0]), int(self.landing_pt_xy[1])), 8, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        cv2.putText(display_img, f"State: {self.state}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if self.state == "READY" and self.calibrator.calibrated:
            cv2.putText(display_img, f"Stable: {self._skeleton_ready_stable}/{8}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if self.foul_detector.reason:
            display_img = self.renderer.put_text_chinese(display_img, f"犯规: {self.foul_detector.reason}", (20, 180), (0, 0, 255), size=40)
        elif self.final_distance_cm is not None:
            cv2.putText(display_img, f"RESULT: {self.final_distance_cm:.1f} cm", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        mv = self.renderer.render_mat_view(frame, self.calibrator, self._get_feet, kpts,
                                           self._takeoff_line_x_cm(), self.takeoff_x_cm, self.landing_x_cm)
        if mv is not None:
            h_m, w_m = mv.shape[:2]
            max_h, max_w = display_img.shape[:2]
            if w_m > max_w or h_m > max_h:
                s = min(max_w / float(w_m), max_h / float(h_m))
                nw, nh = max(1, int(round(w_m * s))), max(1, int(round(h_m * s)))
                mv = cv2.resize(mv, (nw, nh), interpolation=cv2.INTER_AREA)
                h_m, w_m = mv.shape[:2]
            display_img[0:h_m, max_w - w_m:max_w] = mv
        return display_img

    # ---------- 主循环 ----------
    def run(self):
        frame_idx = 0
        calib_frame = None
        _was_calibrated = False
        try:
            while True:
                if self.calibrator.manual_mode and not self.calibrator.mat_locked and calib_frame is not None:
                    ret, frame = True, calib_frame.copy()
                else:
                    ret, frame = self.pose_estimator.read_frame()
                    if ret and self.calibrator.manual_mode and not self.calibrator.mat_locked:
                        calib_frame = frame.copy()
                if not ret:
                    break

                frame_idx += 1
                if self.ocr_reader is not None:
                    self._current_time_info = self.ocr_reader.recognize(frame, frame_idx - 1, self._video_fps)
                    self._time_info_by_frame[frame_idx] = dict(self._current_time_info)
                    if len(self._time_info_by_frame) > 20000:
                        for old_idx in sorted(self._time_info_by_frame)[:2000]:
                            self._time_info_by_frame.pop(old_idx, None)
                    ocr_log_key = self._current_time_info.get("text")
                    if self.debug and ocr_log_key != self._last_ocr_log_text:
                        self._last_ocr_log_text = ocr_log_key
                        self._log("OCR", f"frame={frame_idx}, timecode={self._current_time_info.get('timecode')}, conf={self._current_time_info.get('confidence')}, ok={self._current_time_info.get('ok')}")
                display_img = frame.copy()

                # 手动标定提示
                if self.calibrator.manual_mode and not self.calibrator.mat_locked:
                    cv2.putText(display_img, f"Click 4 corners: {len(self.calibrator.manual_points)}/4",
                                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    self.renderer.draw_mat_outline(display_img, self.calibrator)

                self._ensure_record_writer(display_img)
                self._prev_frame_img = frame.copy()
                kpts = self.pose_estimator.infer_keypoints(frame)

                # 关键点日志
                self._log_keypoints(frame_idx, kpts)

                # 多人检测
                self.foul_detector.check_multi_person(self.pose_estimator.all_kpts_list)
                if self.foul_detector.reason and not self._foul_saved:
                    self._save_foul_record(frame, kpts)

                # 垫子标定
                self._update_stream_recalibration(frame_idx, frame, kpts)

                if self.state == "IDLE" and self.calibrator.update(frame):
                    if not self.calibrator.mat_locked:
                        self.calibrator.mat_locked = True
                    if not _was_calibrated:
                        _was_calibrated = True
                        if self.stream_mode:
                            self._log("CALIB", f"首次标定完成(流模式)，保留全局帧号 frame={frame_idx}")
                        else:
                            frame_idx = 0
                            self._log("CALIB", f"标定完成，帧计数器重置为0")
                        self._log("CALIB", "垫子标定完成")
                        # 保存两张垫子识别图（默认不输出）
                        if self.config.enable_mat_output and self.images_dir:
                            mask_quad = self.calibrator.render_mask(frame)
                            if mask_quad is not None:
                                imwrite_safe(os.path.join(self.images_dir, "mat_mask_quad.jpeg"), cv2.cvtColor(mask_quad, cv2.COLOR_GRAY2BGR))
                            # 输出四边形内的可见颜色区域，和 D:/DeepLearning/hsv/run_calib_test.py 保持一致
                            mask_hsv = self.calibrator.render_visible_mask(frame)
                            if mask_hsv is not None:
                                imwrite_safe(os.path.join(self.images_dir, "mat_mask_hsv.jpeg"), cv2.cvtColor(mask_hsv, cv2.COLOR_GRAY2BGR))
                            self._log("CALIB", "垫子识别图已保存: mat_mask_quad.jpeg (实心四边形), mat_mask_hsv.jpeg (四边形内可见颜色区域)")

                        # 垫子毫米格测试图（默认不输出）
                        if self.config.enable_test_grid and self.images_dir:
                            grid_img = frame.copy()
                            self.renderer.draw_mat_outline(grid_img, self.calibrator)
                            self.renderer.draw_x_line(grid_img, self.calibrator.H_mat2img,
                                                      self.calibrator.mat_width_cm,
                                                      self._takeoff_line_x_cm(), (255, 255, 255), thickness=3)
                            step = 10.0
                            if self.is_rtl:
                                x = self._takeoff_line_x_cm() - step
                                while x >= 0.0:
                                    self.renderer.draw_x_line(grid_img, self.calibrator.H_mat2img,
                                                              self.calibrator.mat_width_cm, x, (0, 255, 0), thickness=1)
                                    x -= step
                            else:
                                x = self._takeoff_line_x_cm() + step
                                while x <= self.calibrator.mat_length_cm:
                                    self.renderer.draw_x_line(grid_img, self.calibrator.H_mat2img,
                                                              self.calibrator.mat_width_cm, x, (0, 255, 0), thickness=1)
                                    x += step
                            imwrite_safe(os.path.join(self.images_dir, "test_grid.jpeg"), grid_img)
                            self._log("CALIB", "垫子毫米格测试图已保存: test_grid.jpeg")

                # ── 差分法：基准帧捕获（垫子上无人时） ──
                if (self.config.enable_diff
                        and self.calibrator.calibrated
                        and not self.diff_detector.has_base_frame
                        and self.state == "IDLE"):
                    has_person_in_mat = False
                    if kpts is not None:
                        ankles = self._get_feet(kpts, "ankle")
                        ankle_xy = self._avg_points(ankles)
                        ankle_cm = self.calibrator.transform_to_mat_cm(ankle_xy)
                        if ankle_cm is not None and self.calibrator.in_mat(ankle_cm):
                            has_person_in_mat = True
                    if not has_person_in_mat:
                        self.diff_detector.capture_base_frame(frame)
                        self._log("DIFF", f"基准帧已捕获（帧{frame_idx}，垫子标定范围无人体关键点）")

                # 保存上一帧骨架数据（供犯规检测使用）
                if kpts is not None:
                    self._last_kpts = kpts

                # ── 状态机：仅骨骼关键点法 ──
                if self.calibrator.calibrated:
                    # 需要关键点数据
                    if kpts is not None and self.state in ("IDLE", "READY"):
                        ankles = self._get_feet(kpts, "ankle")
                        ankle_xy = self._avg_points(ankles)
                        ankle_cm = self.calibrator.transform_to_mat_cm(ankle_xy)

                        toes = self._get_feet(kpts, "toe")
                        toe_l_xy = toes.get("l")
                        toe_r_xy = toes.get("r")
                        toe_l_cm = self.calibrator.transform_to_mat_cm(toe_l_xy) if toe_l_xy else None
                        toe_r_cm = self.calibrator.transform_to_mat_cm(toe_r_xy) if toe_r_xy else None
                        toe_xy = self._avg_points(toes)
                        front_toe_xy, front_toe_cm = self._front_toe(toe_l_xy, toe_r_xy, toe_l_cm, toe_r_cm)

                        toe_y_px = None
                        if front_toe_xy is not None:
                            toe_y_px = float(front_toe_xy[1])
                        elif toe_xy is not None:
                            toe_y_px = float(toe_xy[1])

                        if toe_y_px is None:
                            self._skeleton_toe_missing_counter += 1
                        else:
                            self._skeleton_toe_missing_counter = 0
                            self._skeleton_toe_y_px_hist.append(toe_y_px)

                        ground_toe_y_px = max(self._skeleton_toe_y_px_hist) if self._skeleton_toe_y_px_hist else None
                        toe_lifted = False
                        if toe_y_px is not None and ground_toe_y_px is not None:
                            toe_lifted = (ground_toe_y_px - toe_y_px) > 8.0
                        takeoff_signal = toe_lifted or (self._skeleton_toe_missing_counter >= 2)

                        if self.state == "IDLE":
                            self._handle_idle_skeleton(frame_idx, ankle_cm, front_toe_cm, toe_l_cm, toe_r_cm)
                        elif self.state == "READY":
                            self._handle_ready_skeleton(frame_idx, kpts, ankle_cm, ankle_xy,
                                                        front_toe_cm, front_toe_xy, toe_xy, takeoff_signal)

                if self.state == "JUMPING":
                    # 骨架法落地：需要脚后跟关键点
                    if kpts is not None:
                        heels = self._get_feet(kpts, "heel")
                        heel_l_xy = heels.get("l")
                        heel_r_xy = heels.get("r")
                        heel_l_cm = self.calibrator.transform_to_mat_cm(heel_l_xy) if heel_l_xy else None
                        heel_r_cm = self.calibrator.transform_to_mat_cm(heel_r_xy) if heel_r_xy else None
                    else:
                        heel_l_xy = heel_r_xy = heel_l_cm = heel_r_cm = None
                    self._handle_jumping_skeleton(frame_idx, heel_l_xy, heel_r_xy, heel_l_cm, heel_r_cm, kpts)
                    # 差分法：落地后保存落地帧（从主循环中调用，frame/kpts 有效）
                    if self.state == "LANDED" and (self.config.enable_seg or self.diff_detector.has_base_frame):
                        self.diff_detector.save_landing_frame(frame, kpts)

                if self.calibrator.calibrated:
                    self._recalc_results_with_current_mat()

                # ── 差分法：落地后计算差分距离 ──
                if self.state == "LANDED" and not self._diff_computed and (self.config.enable_seg or self.diff_detector.has_base_frame):
                    self._diff_computed = True
                    to_x, ld_x, dist = self.diff_detector.compute_combined_distance()
                    if to_x is not None and ld_x is not None:
                        log_tag = "YOLO" if self.config.enable_seg else "DIFF"
                        log_label = "YOLO 实例分割" if self.config.enable_seg else "差分法"
                        self._log(log_tag,
                                  f"{log_label}结果: 起跳={to_x:.1f}cm, 落地={ld_x:.1f}cm, "
                                  f"距离={dist:.1f}cm")
                        if self.config.enable_seg:
                            self._log("YOLO_TIME",
                                      f"[{self.diff_detector.yolo_model_label}] "
                                      f"YOLO 实例分割总用时: {self.diff_detector.yolo_total_time:.3f}s")

                        # 保存原始骨骼修正值（仅用于图像标注）
                        skeleton_takeoff_x = self.takeoff_x_cm
                        skeleton_landing_x = self.landing_x_cm
                        self._skeleton_takeoff_x_cm = skeleton_takeoff_x
                        self._skeleton_landing_x_cm = skeleton_landing_x

                        # 成绩以修正后的值为准（YOLO / DIFF），但 MOG2 必须与骨架结果保持物理一致性
                        if not self.config.enable_seg:
                            if skeleton_takeoff_x is not None and abs(to_x - skeleton_takeoff_x) > 14.0:
                                self._log(log_tag, f"差分法起跳结果偏离骨架值过大({abs(to_x - skeleton_takeoff_x):.1f}cm)，回退骨架值 {skeleton_takeoff_x:.1f}cm")
                                to_x = skeleton_takeoff_x
                            if skeleton_landing_x is not None and abs(ld_x - skeleton_landing_x) > 20.0:
                                self._log(log_tag, f"差分法落地结果偏离骨架值过大({abs(ld_x - skeleton_landing_x):.1f}cm)，回退骨架值 {skeleton_landing_x:.1f}cm")
                                ld_x = skeleton_landing_x

                        self.takeoff_x_cm = to_x
                        self.landing_x_cm = ld_x
                        self.final_distance_cm = self._distance_between_x(to_x, ld_x)
                        self._log(log_tag, f"采用{log_label}结果: 起跳={to_x:.1f}cm, 落地={ld_x:.1f}cm, 距离={self.final_distance_cm:.1f}cm")

                        # 用修正后的起跳线重新判断踩线犯规
                        self.foul_detector.check_line_violation(self.takeoff_x_cm, self._takeoff_line_x_cm(), self.jump_direction)

                        if self.config.enable_seg:
                            self._yolo_takeoff_x_cm = to_x
                            self._yolo_landing_x_cm = ld_x
                        else:
                            self._mog_takeoff_x_cm = to_x
                            self._mog_landing_x_cm = ld_x

                        # 重保存 takeoff/score 图像以绘制 YOLO 或 MOG2 修正线
                        if self._takeoff_frame_img is not None:
                            self._takeoff_saved = False
                            self._save_takeoff_image(
                                self._takeoff_frame_img,
                                self._takeoff_kpts,
                                self._takeoff_all_kpts_list,
                            )
                        # 落地图像由后面的 _save_landed_image 自然重新保存
                        # 重新保存 score 和 payload
                        img_base = self._landing_frame_img if self._landing_frame_img is not None else frame
                        self._score_saved = False
                        self._save_score_image(img_base)
                        self._save_payload()

                        self._save_diff_image()
                    else:
                        self._log("DIFF", "差分法计算失败（关键点或基准帧不足）")

                stop_after_output = False
                self._save_landed_image(frame, kpts)
                if self.state == "LANDED":
                    self._save_score_image(self._landing_frame_img if self._landing_frame_img is not None else frame)
                    if self.stream_mode:
                        self._record_stream_result()
                        self._reset_for_next_stream_jump()
                    elif not self.config.display:
                        # 如果开启了 --output-video/--record，先把当前最终画面写入视频，再退出。
                        stop_after_output = True

                if self.config.display or self.config.record_path:
                    composed = self._compose_display(frame, display_img, kpts)
                    if self.record_writer is not None:
                        self.record_writer.write(composed)
                    if self.config.display:
                        cv2.imshow("Auto Long Jump", composed)
                        if cv2.waitKey(1) == ord("q"):
                            break

                if stop_after_output:
                    break

                if self.pose_estimator.image_mode and not self.config.display:
                    break
        finally:
            self.pose_estimator.release()
            if self.record_writer is not None:
                self.record_writer.release()
            cv2.destroyAllWindows()
            if self._kpts_log_fh is not None:
                self._kpts_log_fh.close()
            self._log("EXIT", "系统退出")

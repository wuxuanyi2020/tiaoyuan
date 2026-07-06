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
from src.inference.pose_estimator import PoseEstimator
from src.visualization.rendering import Renderer, imwrite_safe


class StandingLongJumpSystem:
    def __init__(self, config: JumpConfig):
        self.config = config
        self.pose_estimator = PoseEstimator(config.video_source, backend=config.backend)
        self.calibrator = MatCalibrator(
            mat_length_cm=config.mat_length_cm,
            mat_width_cm=config.mat_width_cm,
            manual_mode=config.manual_calib,
        )
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

        self._log("INIT", f"系统初始化完成, video_source={config.video_source}")

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
        self.landing_frame = None

        # ── 轮廓法状态变量（替代骨架关键点） ──
        self._person_area_baseline = None      # READY 状态下的人体轮廓面积基线
        self._front_x_baseline = None          # READY 状态下的人体前缘 X 基线
        self._front_x_cm_hist = deque(maxlen=30)  # 前缘 X 历史，用于平滑取起跳点
        self._ready_stable_frames = 0          # READY 稳定帧数累计
        self._ready_missing_frames = 0         # 人体消失连续帧数（起跳判定）
        self._jump_frame_counter = 0           # JUMPING 状态帧计数器
        self._takeoff_recorded_x = None        # 记录的起跳 X(cm)
        self._last_kpts = None                 # 上一帧骨架数据（用于犯规检测）
        self._takeoff_frame_img = None
        self._landing_frame_img = None
        self._prev_frame_img = None
        self._landed_saved = False
        self._foul_saved = False
        self.record_writer = None

        self.takeoff_display_offset_cm = float(config.takeoff_offset_cm)
        self.landing_offset_cm = float(config.landing_offset_cm)

        if config.display:
            cv2.namedWindow("Auto Long Jump", cv2.WINDOW_NORMAL)
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
        if kind == "heel" and len(kpts) > 22:
            left = self._get_kpt(kpts, self.kpt_idx["l_heel"])
            right = self._get_kpt(kpts, self.kpt_idx["r_heel"])
            if left is not None or right is not None:
                return {"l": left, "r": right}
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
    def _front_toe(toe_l_xy, toe_r_xy, toe_l_cm, toe_r_cm):
        if toe_l_cm is not None and toe_r_cm is not None:
            return (toe_l_xy, toe_l_cm) if toe_l_cm[0] >= toe_r_cm[0] else (toe_r_xy, toe_r_cm)
        if toe_l_cm is not None:
            return toe_l_xy, toe_l_cm
        if toe_r_cm is not None:
            return toe_r_xy, toe_r_cm
        return None, None

    def _reset_round_state(self):
        self._person_area_baseline = None
        self._front_x_baseline = None
        self._front_x_cm_hist.clear()
        self._ready_stable_frames = 0
        self._ready_missing_frames = 0
        self._jump_frame_counter = 0
        self._takeoff_recorded_x = None
        self.takeoff_x_cm = None
        self.takeoff_pt_px = None
        self.landing_x_cm = None
        self.landing_pt_px = None
        self.final_distance_cm = None
        self._takeoff_frame_img = None
        self._landing_frame_img = None
        self.takeoff_pt_xy = None
        self.landing_pt_xy = None
        self._landed_saved = False
        self._foul_saved = False
        self.foul_detector.reset()
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

    def _save_payload(self):
        payload = {
            "score": float(self.final_distance_cm or 0.0),
            "valid": False if self.foul_detector.reason else True,
            "foul_reason": self.foul_detector.reason,
            "distance_cm": float(self.final_distance_cm or 0.0),
            "takeoff_x_cm": float(self.takeoff_x_cm or 0.0),
            "landing_x_cm": float(self.landing_x_cm or 0.0) if self.landing_x_cm is not None else None,
        }
        save_path = os.path.join(self.result_dir, "result.json") if self.result_dir else self.config.save_path
        with open(save_path, "w", encoding="utf-8") as f:
            import json
            json.dump(payload, f, ensure_ascii=False)
        self._log("SAVE", f"结果已保存到 {save_path}")

    def _save_foul_record(self, frame, kpts=None):
        if self._foul_saved or self.foul_detector.reason is None:
            return

        img = frame.copy()
        self.renderer.draw_mat_outline(img, self.calibrator)
        self.renderer.draw_x_line(img, self.calibrator.H_mat2img, self.calibrator.mat_width_cm,
                                  self.config.takeoff_line_cm, (255, 255, 255), thickness=1)

        actual_takeoff_x = self.takeoff_x_cm
        if actual_takeoff_x is None and self._front_x_baseline is not None:
            actual_takeoff_x = self._front_x_baseline + self.takeoff_display_offset_cm
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
        if self.takeoff_x_cm is not None:
            self.foul_detector.check_line_violation(self.takeoff_x_cm, self.config.takeoff_line_cm)
        if self.final_distance_cm is not None and self.landing_x_cm is not None:
            ld_cm = (self.landing_x_cm, 0.0)
            self.foul_detector.check_out_of_bounds(ld_cm)

    # ---------- 状态处理 ----------
    def _enter_ready_state(self, frame):
        self.state = "READY"
        self._log("STATE", "IDLE -> READY")
        self._reset_round_state()
        area = self.calibrator.get_person_area_px(frame)
        front_x = self.calibrator.get_person_front_x_cm(frame)
        self._person_area_baseline = area if area > 500 else 2000.0
        self._front_x_baseline = front_x if front_x is not None else 0.0
        self._log("READY", f"面积基线={self._person_area_baseline:.0f}px, 前缘X={self._front_x_baseline:.1f}cm")

    def _handle_idle(self, frame):
        self.foul_detector.reset()
        area = self.calibrator.get_person_area_px(frame)
        if area > 500:
            self._log("IDLE", f"检测到人体进入垫内, 轮廓面积={area:.0f}px")
            self._enter_ready_state(frame)

    def _handle_ready(self, frame_idx, frame):
        area = self.calibrator.get_person_area_px(frame)
        front_x = self.calibrator.get_person_front_x_cm(frame)

        if self._person_area_baseline is None:
            self._person_area_baseline = area if area > 500 else 2000.0
            self._ready_stable_frames = 0
            return

        # 当前面积相对于基线的比例
        area_ratio = area / max(self._person_area_baseline, 1.0)

        # ── 人还在垫上（轮廓面积 > 基线的 30%）─
        if area_ratio > 0.3:
            # 平滑更新基线
            self._person_area_baseline = 0.95 * self._person_area_baseline + 0.05 * area
            self._ready_stable_frames += 1
            self._ready_missing_frames = 0

            # 收集前缘 X（用于起跳点记录）
            if front_x is not None and 0 < front_x < 350:
                self._front_x_cm_hist.append(front_x)

            # 每10帧 Debug
            if frame_idx % 10 == 0:
                stable_front = np.mean(self._front_x_cm_hist) if self._front_x_cm_hist else 0
                print(f"[DEBUG] 帧{frame_idx}: 面积={area:.0f}/{self._person_area_baseline:.0f}, "
                      f"ratio={area_ratio:.2f}, 前缘X={front_x:.1f}/{stable_front:.1f}, "
                      f"stable={self._ready_stable_frames}")

        # ── 人体轮廓面积骤降（ratio < 0.3）→ 起跳判定 ──
        else:
            self._ready_missing_frames += 1

            if frame_idx % 5 == 0:
                print(f"[DEBUG] 帧{frame_idx}: 面积骤降 area={area:.0f}/{self._person_area_baseline:.0f} "
                      f"ratio={area_ratio:.2f}, missing={self._ready_missing_frames}")

            # 起跳判定：面积降至基线 30% 以下，且稳定足够帧数
            needs_stable_frames = 8
            needs_missing = max(2, self.config.trigger_frames)
            can_takeoff = (self._ready_stable_frames >= needs_stable_frames
                           and self._ready_missing_frames >= needs_missing
                           and area_ratio < 0.3)

            if not can_takeoff:
                return

            # ── 确定起跳点（取前缘 X 历史均值）──
            if not self._front_x_cm_hist:
                self._log("JUMP", "起跳失败：无前缘 X 历史数据")
                self._reset_round_state()
                self.state = "IDLE"
                return

            takeoff_x = float(np.mean(self._front_x_cm_hist))
            self._takeoff_recorded_x = takeoff_x

            self._log("JUMP", f"起跳成功！稳定期={self._ready_stable_frames}帧, "
                              f"起跳点={takeoff_x:.1f}cm (前缘均值, N={len(self._front_x_cm_hist)})")

            # 犯规检测（依赖骨架关键点）
            if self.foul_detector.enabled and self._last_kpts is not None:
                self.foul_detector.check_step_jump(None, takeoff_x)
                self.foul_detector.check_single_leg_takeoff(self._last_kpts)
                self.foul_detector.check_prop_assistance(self._last_kpts)
            self.foul_detector.check_line_violation(takeoff_x, self.config.takeoff_line_cm)
            if self.foul_detector.reason:
                self._log("FOUL", f"起跳时检测到犯规: {self.foul_detector.reason}")

            self.state = "JUMPING"
            self._log("STATE", "READY -> JUMPING")
            self._jump_frame_counter = 0
            self.takeoff_frame = frame_idx
            self.takeoff_x_cm = takeoff_x + self.takeoff_display_offset_cm
            self.takeoff_pt_px = None
            self.takeoff_pt_xy = None
            self._takeoff_frame_img = frame.copy()

    def _handle_jumping(self, frame_idx, frame):
        self._jump_frame_counter += 1
        area = self.calibrator.get_person_area_px(frame)
        back_x = self.calibrator.get_person_back_x_cm(frame)

        # ── 落地检测 ──
        detected_landing = False
        if self._jump_frame_counter >= self.config.min_flight_frames:
            # 轮廓面积 > 阈值 = 人体重新出现在垫上 = 落地
            if area > 2000:
                detected_landing = True

        # 超时强制落地
        if not detected_landing and self._jump_frame_counter >= self.config.max_jump_frames:
            detected_landing = True
            self._log("LAND", f"落地超时({self.config.max_jump_frames}帧)，强制落地")

        if not detected_landing:
            return

        # ── 确定落地位置 ──
        landing_x = None
        used_back = False
        if back_x is not None and 0 < back_x < 350:
            landing_x = back_x
            used_back = True
        else:
            # 后备：用轮廓底部 Y 处的 X
            bottom_x = self.calibrator.get_person_bottom_x_cm(frame)
            if bottom_x is not None and 0 < bottom_x < 350:
                landing_x = bottom_x
            else:
                # 用轮廓重心 X
                centroid_x = self.calibrator.get_person_centroid_x_cm(frame)
                if centroid_x is not None and 0 < centroid_x < 350:
                    landing_x = centroid_x

        if landing_x is None:
            self._log("LAND", "无法获取落地位置，忽略")
            self.state = "READY"
            return

        # ── 计算距离 ──
        temp_dist = landing_x - (self.takeoff_x_cm if self.takeoff_x_cm else 0.0) + self.landing_offset_cm

        if temp_dist < 50.0:
            self._log("LAND", f"忽略假动作 (距离 {temp_dist:.1f} cm 过短)，重置为 READY")
            self._reset_round_state()
            self.state = "READY"
            self._enter_ready_state(frame)
            return

        method = "back_x" if used_back else "bottom_x/centroid"
        self._log("LAND", f"落地触发({method}), landing_x={landing_x:.1f}, 距离={temp_dist:.1f}cm")
        self.state = "LANDED"
        self.landing_frame = frame_idx
        self.landing_pt_px = None
        self.landing_pt_xy = None
        self.landing_x_cm = landing_x
        self.final_distance_cm = max(0.0, temp_dist)
        if self.config.debug_dir:
            self._landing_frame_img = frame.copy()

    # ---------- 结果保存 ----------
    def _save_landed_image(self, frame, kpts):
        if self.state != "LANDED" or self._landed_saved:
            return

        img = (self._landing_frame_img if self._landing_frame_img is not None else frame).copy()
        self.renderer.draw_mat_outline(img, self.calibrator)
        self.renderer.draw_x_line(img, self.calibrator.H_mat2img, self.calibrator.mat_width_cm,
                                  self.config.takeoff_line_cm, (255, 255, 255), thickness=1)
        if kpts is not None:
            self.renderer.draw_pose(img, kpts, self.pose_estimator.mp_connections, color=(0, 255, 0))
        if self.takeoff_x_cm is not None:
            self.renderer.draw_x_line(img, self.calibrator.H_mat2img, self.calibrator.mat_width_cm,
                                      self.takeoff_x_cm, (0, 255, 255))
        if self.landing_x_cm is not None:
            self.renderer.draw_x_line(img, self.calibrator.H_mat2img, self.calibrator.mat_width_cm,
                                      self.landing_x_cm, (0, 0, 255))
        self.renderer.draw_measurement_line(img, self.calibrator.H_mat2img, self.calibrator.mat_width_cm,
                                            self.takeoff_x_cm, self.landing_x_cm)
        # 左上角标注成绩
        score_text = f"成绩: {self.final_distance_cm:.1f} cm" if self.final_distance_cm is not None else "成绩: 无"
        takeoff_text = f"起跳点: {self.takeoff_x_cm:.1f} cm" if self.takeoff_x_cm is not None else ""
        landing_text = f"落地点: {self.landing_x_cm:.1f} cm" if self.landing_x_cm is not None else ""
        img = self.renderer.put_text_chinese(img, score_text, (50, 80), (0, 255, 0), size=50)
        if takeoff_text:
            img = self.renderer.put_text_chinese(img, takeoff_text, (50, 140), (255, 255, 0), size=30)
        if landing_text:
            img = self.renderer.put_text_chinese(img, landing_text, (50, 180), (255, 255, 0), size=30)
        if self.foul_detector.reason:
            img = self.renderer.put_text_chinese(img, f"INVALID: {self.foul_detector.reason}", (50, 230), (0, 0, 255), size=50)

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self.images_dir, f"landed-{ts}.jpeg") if self.images_dir else f"landed-{ts}.jpeg"
        imwrite_safe(filename, img)
        self._log("SAVE", f"落地图片已保存: {filename}")
        self._landed_saved = True
        self._save_payload()

    # ---------- 显示合成 ----------
    def _compose_display(self, frame, display_img, kpts):
        if kpts is not None:
            self.renderer.draw_pose(display_img, kpts, self.pose_estimator.mp_connections, color=(0, 255, 0))
            self.renderer.draw_feet(display_img, self._get_feet, kpts)

        self.renderer.draw_mat_outline(display_img, self.calibrator)
        self.renderer.draw_x_line(display_img, self.calibrator.H_mat2img, self.calibrator.mat_width_cm,
                                  self.config.takeoff_line_cm, (255, 255, 255), thickness=1, label="Limit")
        self.renderer.draw_measurement_line(display_img, self.calibrator.H_mat2img, self.calibrator.mat_width_cm,
                                            self.takeoff_x_cm, self.landing_x_cm)

        if self.takeoff_x_cm is not None:
            ct = (0, 0, 255) if self.foul_detector.reason == "踩线 (Line Violation)" else (0, 255, 255)
            self.renderer.draw_x_line(display_img, self.calibrator.H_mat2img, self.calibrator.mat_width_cm,
                                      self.takeoff_x_cm, ct)
        elif self._front_x_baseline is not None:
            px = self._front_x_baseline + self.takeoff_display_offset_cm
            cp = (0, 0, 255) if px > (self.config.takeoff_line_cm + 1.0) else (255, 0, 0)
            self.renderer.draw_x_line(display_img, self.calibrator.H_mat2img, self.calibrator.mat_width_cm, px, cp)

        if self.landing_x_cm is not None:
            self.renderer.draw_x_line(display_img, self.calibrator.H_mat2img, self.calibrator.mat_width_cm,
                                      self.landing_x_cm, (0, 0, 255))
        if self.takeoff_pt_xy is not None:
            cv2.circle(display_img, (int(self.takeoff_pt_xy[0]), int(self.takeoff_pt_xy[1])), 8, (0, 255, 255), 2, lineType=cv2.LINE_AA)
        if self.landing_pt_xy is not None:
            cv2.circle(display_img, (int(self.landing_pt_xy[0]), int(self.landing_pt_xy[1])), 8, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        cv2.putText(display_img, f"State: {self.state}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if self.state == "READY" and self.calibrator.calibrated:
            cv2.putText(display_img, f"Stable: {self._ready_stable_frames}/{8}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if self.foul_detector.reason:
            display_img = self.renderer.put_text_chinese(display_img, f"犯规: {self.foul_detector.reason}", (20, 180), (0, 0, 255), size=40)
        elif self.final_distance_cm is not None:
            cv2.putText(display_img, f"RESULT: {self.final_distance_cm:.1f} cm", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        mv = self.renderer.render_mat_view(frame, self.calibrator, self._get_feet, kpts,
                                           self.config.takeoff_line_cm, self.takeoff_x_cm, self.landing_x_cm)
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
                if self.state == "IDLE" and self.calibrator.update(frame):
                    if not self.calibrator.mat_locked:
                        self.calibrator.mat_locked = True
                        self._log("CALIB", "垫子标定完成")
                        # 保存两张垫子识别图
                        if self.images_dir:
                            mask_quad = self.calibrator.render_mask(frame)
                            if mask_quad is not None:
                                imwrite_safe(os.path.join(self.images_dir, "mat_mask_quad.jpeg"), cv2.cvtColor(mask_quad, cv2.COLOR_GRAY2BGR))
                            mask_hsv = self.calibrator.render_hsv_mask(frame)
                            if mask_hsv is not None:
                                imwrite_safe(os.path.join(self.images_dir, "mat_mask_hsv.jpeg"), cv2.cvtColor(mask_hsv, cv2.COLOR_GRAY2BGR))
                            self._log("CALIB", "垫子识别图已保存: mat_mask_quad.jpeg (四边形拟合), mat_mask_hsv.jpeg (HSV原始)")

                # 保存上一帧骨架数据（供犯规检测使用）
                if kpts is not None:
                    self._last_kpts = kpts

                # ── 状态机：完全基于轮廓检测，不依赖骨架关键点 ──
                if self.calibrator.calibrated:
                    if self.state == "IDLE":
                        self._handle_idle(frame)
                    elif self.state == "READY":
                        self._handle_ready(frame_idx, frame)

                if self.state == "JUMPING":
                    self._handle_jumping(frame_idx, frame)

                if self.calibrator.calibrated:
                    self._recalc_results_with_current_mat()

                self._save_landed_image(frame, kpts)
                if self.state == "LANDED" and not self.config.display:
                    break

                if self.config.display or self.config.record_path:
                    composed = self._compose_display(frame, display_img, kpts)
                    if self.record_writer is not None:
                        self.record_writer.write(composed)
                    if self.config.display:
                        cv2.imshow("Auto Long Jump", composed)
                        if cv2.waitKey(1) == ord("q"):
                            break

                if self.pose_estimator.image_mode and not self.config.display and not self.config.record_path:
                    break
        finally:
            self.pose_estimator.release()
            if self.record_writer is not None:
                self.record_writer.release()
            cv2.destroyAllWindows()
            if self._kpts_log_fh is not None:
                self._kpts_log_fh.close()
            self._log("EXIT", "系统退出")

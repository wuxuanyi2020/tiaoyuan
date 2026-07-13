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
        self.shoe_detector = ShoeEdgeDetector(self.calibrator)
        self.diff_detector = DiffDetector(self.calibrator, enable_seg=config.enable_seg)
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

        self._log("INIT", f"系统初始化完成, video_source={config.video_source}, 检测方式=骨骼关键点(skeleton)")

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
        self.landing_frame = None

        # ── 骨骼关键点法状态变量 ──
        self._skeleton_baseline_x_cm = None
        self._skeleton_baseline_hip_x = None
        self._skeleton_baseline_ankle_y = None
        self._skeleton_jump_trigger_counter = 0
        self._skeleton_ready_stable = 0
        self._skeleton_front_toe_hist = deque(maxlen=30)
        self._skeleton_toe_y_px_hist = deque(maxlen=12)
        self._skeleton_toe_missing_counter = 0
        self._skeleton_jump_counter = 0
        self._skeleton_prev_takeoff_data = None  # 前一帧的起跳相关数据，用于倒推起跳点
        self._skeleton_foot_hist = {
            "l": {"y_px": deque(maxlen=3), "x_cm": deque(maxlen=3), "y_cm": deque(maxlen=3), "xy_px": deque(maxlen=3)},
            "r": {"y_px": deque(maxlen=3), "x_cm": deque(maxlen=3), "y_cm": deque(maxlen=3), "xy_px": deque(maxlen=3)},
        }

        self._last_kpts = None  # 上一帧骨架数据（用于犯规检测）
        self._takeoff_frame_img = None
        self._landing_frame_img = None
        self._prev_frame_img = None
        self._takeoff_saved = False
        self._landed_saved = False
        self._foul_saved = False
        self.record_writer = None

        # ── 差分法状态变量 ──
        self._diff_computed = False

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
    def _front_toe(toe_l_xy, toe_r_xy, toe_l_cm, toe_r_cm):
        if toe_l_cm is not None and toe_r_cm is not None:
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
        self._skeleton_jump_trigger_counter = 0
        self._skeleton_ready_stable = 0
        self._skeleton_front_toe_hist.clear()
        self._skeleton_toe_y_px_hist.clear()
        self._skeleton_toe_missing_counter = 0
        self._skeleton_jump_counter = 0
        self._skeleton_prev_takeoff_data = None
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
        self.takeoff_pt_xy = None
        self.landing_pt_xy = None
        self._takeoff_saved = False
        self._landed_saved = False
        self._foul_saved = False
        self.foul_detector.reset()
        self._diff_computed = False
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

    # ═══════════════════════════════════════════════
    # 骨骼关键点法处理函数
    # ═══════════════════════════════════════════════


    @staticmethod
    def _skeleton_detect_contact(hist):
        """脚后跟 Y 轴触地检测（骨架法）。"""
        if len(hist["y_px"]) < 3 or len(hist["x_cm"]) < 3:
            return None, None
        y0, y1, y2 = list(hist["y_px"])
        x1 = list(hist["x_cm"])[1]
        if (y1 >= y0) and (y1 >= y2) and x1 > 20.0:
            xy_mid = list(hist["xy_px"])[1] if len(hist["xy_px"]) == 3 else None
            return x1, xy_mid
        return None, None

    def _skeleton_enter_ready(self, ankle_cm, front_toe_cm):
        if front_toe_cm is None or not self.calibrator.in_mat(front_toe_cm):
            return
        self.state = "READY"
        self._log("STATE", "IDLE -> READY")
        self._reset_round_state()
        self._skeleton_baseline_x_cm = front_toe_cm[0]
        self._log("READY", f"骨架法基线X={self._skeleton_baseline_x_cm:.1f}cm")

    def _handle_idle_skeleton(self, ankle_cm, front_toe_cm):
        self.foul_detector.reset()
        if ankle_cm is not None and self.calibrator.in_mat(ankle_cm):
            self._log("IDLE", f"检测到人体在垫内(骨架), ankle=({ankle_cm[0]:.1f},{ankle_cm[1]:.1f})cm")
            self._skeleton_enter_ready(ankle_cm, front_toe_cm)

    def _handle_ready_skeleton(self, frame_idx, kpts, ankle_cm, ankle_xy, front_toe_cm, front_toe_xy, toe_xy, takeoff_signal):
        hip_l = self._get_kpt(kpts, self.kpt_idx["l_hip"])
        hip_r = self._get_kpt(kpts, self.kpt_idx["r_hip"])
        hip_xy = self._avg_points({"l": hip_l, "r": hip_r})
        hip_cm = self.calibrator.transform_to_mat_cm(hip_xy) if hip_xy else None
        curr_ankle_y_px = ankle_xy[1] if ankle_xy else 99999

        if ankle_cm is None:
            return

        if front_toe_cm is None or not self.calibrator.in_mat(front_toe_cm):
            return  # 无脚尖位置数据，跳过本帧（不用脚踝替代起跳点）

        cur_x = front_toe_cm[0]
        if self._skeleton_baseline_x_cm is None:
            self._skeleton_baseline_x_cm = cur_x
        if self._skeleton_baseline_hip_x is None and hip_cm:
            self._skeleton_baseline_hip_x = hip_cm[0]
        if self._skeleton_baseline_ankle_y is None:
            self._skeleton_baseline_ankle_y = curr_ankle_y_px

        toe_moved = cur_x - self._skeleton_baseline_x_cm
        raw_toe_moved = toe_moved  # 保存原始值（重置前）
        raw_hip_moved = (hip_cm[0] - self._skeleton_baseline_hip_x) if (hip_cm and self._skeleton_baseline_hip_x) else 0  # 保存重置前的 hip_moved
        stable_before_update = self._skeleton_ready_stable
        stable_reset_after_ready = False
        if abs(toe_moved) < 6.0:
            self._skeleton_baseline_x_cm = (0.9 * self._skeleton_baseline_x_cm) + (0.1 * cur_x)
            if hip_cm and self._skeleton_baseline_hip_x:
                self._skeleton_baseline_hip_x = (0.9 * self._skeleton_baseline_hip_x) + (0.1 * hip_cm[0])
            if self._skeleton_baseline_ankle_y:
                self._skeleton_baseline_ankle_y = (0.9 * self._skeleton_baseline_ankle_y) + (0.1 * curr_ankle_y_px)
            self._skeleton_ready_stable += 1
        else:
            if not (self._skeleton_ready_stable >= 10 and toe_moved > 0):
                stable_reset_after_ready = stable_before_update > 35
                self._skeleton_ready_stable = 0
                self._skeleton_baseline_x_cm = cur_x
                if hip_cm:
                    self._skeleton_baseline_hip_x = hip_cm[0]
                self._skeleton_baseline_ankle_y = curr_ankle_y_px
                toe_moved = 0

        hip_moved = (hip_cm[0] - self._skeleton_baseline_hip_x) if (hip_cm and self._skeleton_baseline_hip_x) else 0
        ankle_lifted_px = (self._skeleton_baseline_ankle_y - curr_ankle_y_px) if self._skeleton_baseline_ankle_y else 0
        ankle_lifted = ankle_lifted_px > 20.0

        is_taking_off = False
        takeoff_reason = ""
        if stable_reset_after_ready:
            is_taking_off = True
            takeoff_reason = f"stable_before_update={stable_before_update}>35 后突然置0"
        elif self._skeleton_toe_missing_counter >= 3 and toe_moved > -3.0:
            is_taking_off = True
            takeoff_reason = f"toe_missing={self._skeleton_toe_missing_counter}>=3, toe_moved={toe_moved:.1f}>-3.0"
        if toe_moved > max(self.config.trigger_move_cm, 30.0):
            is_taking_off = True
            takeoff_reason = f"toe_moved={toe_moved:.1f} > max(trigger_move_cm={self.config.trigger_move_cm}, 30.0)"
        elif hip_moved > 35.0 and toe_moved > 10.0:
            is_taking_off = True
            takeoff_reason = f"hip_moved={hip_moved:.1f}>35.0 and toe_moved={toe_moved:.1f}>4.0"
        elif ankle_lifted and toe_moved > 3.0 and self._skeleton_ready_stable == 0:
            is_taking_off = True
            takeoff_reason = f"ankle_lifted={ankle_lifted_px:.1f}px>20.0px and toe_moved={toe_moved:.1f}>3.0 and stable=0"

        trigger_count = self._skeleton_jump_trigger_counter
        stable_gate_passed = self._skeleton_ready_stable >= 10 or stable_reset_after_ready
        self._skeleton_jump_trigger_counter = (trigger_count + 1 if is_taking_off else 0) if stable_gate_passed else 0

        if self.debug:
            self._log("DEBUG_TAKEOFF_PARAMS", f"帧{frame_idx}: toe_moved={toe_moved:.1f}(raw={raw_toe_moved:.1f}) | hip_moved={hip_moved:.1f}(raw={raw_hip_moved:.1f}) | "
                      f"ankle_lifted_px={ankle_lifted_px:.1f}(lifted={ankle_lifted}) | toe_missing={self._skeleton_toe_missing_counter} | "
                      f"trigger={self._skeleton_jump_trigger_counter}/1 | stable={self._skeleton_ready_stable} | stable_before={stable_before_update} | "
                      f"stable_reset_after_ready={stable_reset_after_ready} | is_taking_off={is_taking_off} | reason={takeoff_reason or '无'}")

        if not takeoff_signal and front_toe_cm is not None and front_toe_xy is not None and self.calibrator.in_mat(front_toe_cm):
            self._skeleton_front_toe_hist.append((float(front_toe_cm[0]), float(front_toe_cm[1]), float(front_toe_xy[0]), float(front_toe_xy[1])))

        if self._skeleton_jump_trigger_counter < 1:
            # 未起跳：保存本帧数据作为下一帧起跳时的倒推候选
            self._skeleton_prev_takeoff_data = {
                "front_toe_cm": front_toe_cm,
                "front_toe_xy": front_toe_xy,
                "toe_xy": toe_xy,
                "ankle_xy": ankle_xy,
                "frame_idx": frame_idx,
                "kpts": kpts,
                "all_kpts_list": list(self.pose_estimator.all_kpts_list) if self.pose_estimator.all_kpts_list else [],
                "frame_img": self._prev_frame_img.copy() if self._prev_frame_img is not None else None,
            }
            return

        # 起跳确认：使用前一帧的数据精确倒推起跳点（避免触发帧脚已离地前移）
        prev = self._skeleton_prev_takeoff_data
        if prev is not None and prev["front_toe_cm"] is not None:
            takeoff_x = prev["front_toe_cm"][0]
            takeoff_frame = prev["frame_idx"]
            takeoff_pt_px = (prev["front_toe_xy"] if prev["front_toe_xy"] is not None
                             else (prev["toe_xy"] if prev["toe_xy"] is not None else prev["ankle_xy"]))
            takeoff_kpts = prev["kpts"]
            takeoff_img = prev["frame_img"]
            takeoff_all_kpts = prev.get("all_kpts_list", [])
        else:
            # 无前一帧数据时回退到当前帧
            takeoff_x = front_toe_cm[0]
            takeoff_frame = frame_idx
            takeoff_pt_px = front_toe_xy if front_toe_xy is not None else (toe_xy if toe_xy else ankle_xy)
            takeoff_kpts = kpts
            takeoff_img = self._prev_frame_img if self._prev_frame_img is not None else frame
            takeoff_all_kpts = list(self.pose_estimator.all_kpts_list) if self.pose_estimator.all_kpts_list else []

        if self.debug:
            self._log("DEBUG_TAKEOFF", f"帧{takeoff_frame}(<-{frame_idx}): 起跳触发 | reason={takeoff_reason} | takeoff_x={takeoff_x:.1f}cm | stable={self._skeleton_ready_stable}帧 | trigger={self._skeleton_jump_trigger_counter}/1帧")
        self._log("JUMP", f"起跳成功(骨架)！稳定期={self._skeleton_ready_stable}帧, 起跳点={takeoff_x:.1f}cm(倒推自帧{takeoff_frame})")

        self.foul_detector.check_step_jump(self._skeleton_front_toe_hist, self._skeleton_baseline_x_cm)
        self.foul_detector.check_single_leg_takeoff(takeoff_kpts)
        self.foul_detector.check_line_violation(takeoff_x, self.config.takeoff_line_cm)
        self.foul_detector.check_prop_assistance(takeoff_kpts)
        if self.foul_detector.reason:
            self._log("FOUL", f"起跳时检测到犯规: {self.foul_detector.reason}")

        self.state = "JUMPING"
        self._log("STATE", "READY -> JUMPING")
        self._skeleton_jump_counter = 0
        self.takeoff_frame = takeoff_frame
        self.takeoff_pt_px = takeoff_pt_px
        self.takeoff_pt_xy = self.takeoff_pt_px
        self.takeoff_x_cm = takeoff_x + self.takeoff_display_offset_cm
        self._save_takeoff_image(takeoff_img if takeoff_img is not None else frame, takeoff_kpts, takeoff_all_kpts)
        self._takeoff_frame_img = takeoff_img.copy() if takeoff_img is not None else None
        # 差分法：保存起跳帧
        if self.diff_detector.has_base_frame:
            takeoff_frame_save = takeoff_img if takeoff_img is not None else frame
            self.diff_detector.save_takeoff_frame(takeoff_frame_save, takeoff_kpts)

    def _handle_jumping_skeleton(self, frame_idx, heel_l_xy, heel_r_xy, heel_l_cm, heel_r_cm):
        self._skeleton_jump_counter += 1
        if heel_l_xy is not None or heel_r_xy is not None:
            for side, heel_xy, heel_cm in [("l", heel_l_xy, heel_l_cm), ("r", heel_r_xy, heel_r_cm)]:
                if heel_xy is None or heel_cm is None:
                    continue
                self._skeleton_foot_hist[side]["y_px"].append(heel_xy[1])
                self._skeleton_foot_hist[side]["x_cm"].append(heel_cm[0] - (self.takeoff_x_cm or 0.0))
                self._skeleton_foot_hist[side]["y_cm"].append(heel_cm[1])
                self._skeleton_foot_hist[side]["xy_px"].append((float(heel_xy[0]), float(heel_xy[1])))

        detected_landing = False
        landing_xy_candidate = None
        landing_reason = ""
        contact_l = contact_r = None  # 提前声明，供 debug 日志使用

        if self._skeleton_jump_counter >= self.config.min_flight_frames:
            contact_l, contact_l_xy = self._skeleton_detect_contact(self._skeleton_foot_hist["l"])
            contact_r, contact_r_xy = self._skeleton_detect_contact(self._skeleton_foot_hist["r"])
            if contact_l is not None or contact_r is not None:
                detected_landing = True
                if contact_l is not None and contact_r is not None:
                    landing_xy_candidate = contact_l_xy if contact_l <= contact_r else contact_r_xy
                    landing_reason = f"双脚触地: L_x={contact_l:.1f}cm, R_x={contact_r:.1f}cm, 取min={min(contact_l, contact_r):.1f}cm"
                else:
                    landing_xy_candidate = contact_l_xy if contact_l is not None else contact_r_xy
                    landing_reason = f"单脚触地: {'左脚' if contact_l is not None else '右脚'}"

        if not detected_landing and self._skeleton_jump_counter >= self.config.max_jump_frames:
            detected_landing = True
            landing_reason = f"超时强制落地: jump_counter={self._skeleton_jump_counter} >= max_jump_frames={self.config.max_jump_frames}"

        if self.debug and not detected_landing:
            heel_l_x = self._skeleton_foot_hist["l"]["x_cm"][-1] if self._skeleton_foot_hist["l"]["x_cm"] else None
            heel_r_x = self._skeleton_foot_hist["r"]["x_cm"][-1] if self._skeleton_foot_hist["r"]["x_cm"] else None
            contact_l_info = f"L_contact={contact_l:.1f}" if contact_l is not None else "L_contact=None"
            contact_r_info = f"R_contact={contact_r:.1f}" if contact_r is not None else "R_contact=None"
            self._log("DEBUG_LANDING_PARAMS", f"帧{frame_idx}: jump_cnt={self._skeleton_jump_counter} | "
                      f"heel_L_x={heel_l_x}cm | heel_R_x={heel_r_x}cm | "
                      f"{contact_l_info} | {contact_r_info} | "
                      f"min_flight={self.config.min_flight_frames} | detected_landing={detected_landing} | reason={landing_reason or '无'}")

        if not detected_landing:
            return

        landing_x_for_dist = None
        if landing_xy_candidate is not None:
            temp_cm_pt = self.calibrator.transform_to_mat_cm(landing_xy_candidate)
            if temp_cm_pt is not None and 0 < temp_cm_pt[0] < 340:
                landing_x_for_dist = temp_cm_pt[0]

        if landing_x_for_dist is None:
            self._log("LAND", "无法获取骨架落地位置，忽略")
            self.state = "READY"
            return

        # 修正落地点 X（鞋后跟厚度补偿），统一后续画线/量距/成绩
        landing_x_for_dist += self.landing_offset_cm

        temp_dist = landing_x_for_dist - (self.takeoff_x_cm if self.takeoff_x_cm else 0.0)

        if temp_dist < 50.0:
            self._log("LAND", f"忽略假动作(骨架, 距离{temp_dist:.1f}cm 过短)，重置为 READY")
            self.state = "READY"
            self._skeleton_jump_trigger_counter = 0
            self._skeleton_ready_stable = 0
            self._skeleton_jump_counter = 0
            return

        self.state = "LANDED"
        if self.debug:
            self._log("DEBUG_LANDING", f"帧{frame_idx}: 落地触发 | reason={landing_reason} | landing_x={landing_x_for_dist:.1f}cm | jump_counter={self._skeleton_jump_counter}")
        self._log("LAND", f"落地触发(骨架), landing_x={landing_x_for_dist:.1f}, 距离={temp_dist:.1f}cm")
        self.landing_frame = frame_idx
        self.landing_pt_px = landing_xy_candidate
        self.landing_pt_xy = self.landing_pt_px
        self.landing_x_cm = landing_x_for_dist
        self.final_distance_cm = max(0.0, temp_dist)
        if self.config.debug_dir:
            self._landing_frame_img = None

    # ---------- 结果保存 ----------
    def _save_takeoff_image(self, frame, kpts=None, all_kpts_list=None):
        """保存起跳帧图像（含骨架、垫子轮廓、起跳线标注）。"""
        if not self.images_dir or self._takeoff_saved:
            return
        all_kpts = all_kpts_list if all_kpts_list is not None else (self.pose_estimator.all_kpts_list or [])
        img = frame.copy()
        self.renderer.draw_mat_outline(img, self.calibrator)
        self.renderer.draw_x_line(img, self.calibrator.H_mat2img, self.calibrator.mat_width_cm,
                                  self.config.takeoff_line_cm, (255, 255, 255), thickness=1)
        if kpts is not None:
            self.renderer.draw_pose(img, kpts, self.pose_estimator.mp_connections, color=(0, 255, 0))
            self.renderer.draw_feet(img, self._get_feet, kpts)
        for pk in all_kpts:
            self.renderer.draw_pose(img, pk, self.pose_estimator.mp_connections, color=(0, 255, 0))
            self.renderer.draw_feet(img, self._get_feet, pk)
        if self.takeoff_x_cm is not None:
            self.renderer.draw_x_line(img, self.calibrator.H_mat2img, self.calibrator.mat_width_cm,
                                      self.takeoff_x_cm, (0, 255, 255), thickness=2, label="Takeoff")
        takeoff_text = f"起跳点: {self.takeoff_x_cm:.1f} cm" if self.takeoff_x_cm is not None else "起跳点: 无"
        img = self.renderer.put_text_chinese(img, f"起跳帧", (50, 80), (0, 255, 255), size=50)
        img = self.renderer.put_text_chinese(img, takeoff_text, (50, 140), (255, 255, 0), size=30)
        if self.foul_detector.reason:
            img = self.renderer.put_text_chinese(img, f"犯规: {self.foul_detector.reason}", (50, 200), (0, 0, 255), size=40)
        filename = os.path.join(self.images_dir, "takeoff.jpeg")
        imwrite_safe(filename, img)
        self._log("SAVE", f"起跳图片已保存: {filename}")
        self._takeoff_saved = True

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
            # landing_x_cm 已包含 offset，落地线与量距线据此绘制保持一致
            self.renderer.draw_x_line(img, self.calibrator.H_mat2img, self.calibrator.mat_width_cm,
                                      self.landing_x_cm, (0, 0, 255))
            landing_display_x = self.landing_x_cm
        else:
            landing_display_x = None
        self.renderer.draw_measurement_line(img, self.calibrator.H_mat2img, self.calibrator.mat_width_cm,
                                            self.takeoff_x_cm, self.landing_x_cm)
        # 左上角标注成绩
        score_text = f"成绩: {self.final_distance_cm:.1f} cm" if self.final_distance_cm is not None else "成绩: 无"
        takeoff_text = f"起跳点: {self.takeoff_x_cm:.1f} cm" if self.takeoff_x_cm is not None else ""
        landing_text = f"落地点: {landing_display_x:.1f} cm (offset={self.landing_offset_cm:.1f})" if landing_display_x is not None else ""
        img = self.renderer.put_text_chinese(img, score_text, (50, 80), (0, 255, 0), size=50)
        if takeoff_text:
            img = self.renderer.put_text_chinese(img, takeoff_text, (50, 140), (255, 255, 0), size=30)
        if landing_text:
            img = self.renderer.put_text_chinese(img, landing_text, (50, 180), (255, 255, 0), size=30)
        if self.foul_detector.reason:
            img = self.renderer.put_text_chinese(img, f"INVALID: {self.foul_detector.reason}", (50, 230), (0, 0, 255), size=50)

        filename = os.path.join(self.images_dir, "landed.jpeg")
        imwrite_safe(filename, img)
        self._log("SAVE", f"落地图片已保存: {filename}")
        self._landed_saved = True
        self._save_payload()

    def _save_diff_image(self):
        """分步保存差分法各阶段过程照片到 images/diff/ 目录。

           输出文件名格式: diff-Stage{阶段}-{描述}.jpeg
        """
        if not self.images_diff_dir or not self.diff_detector.has_base_frame:
            return
        os.makedirs(self.images_diff_dir, exist_ok=True)
        stages = [
            # (mode,                  suffix)
            ("base",                  "Stage1-baseframe"),
            ("roi_takeoff",           "Stage2-roi-takeoff"),
            ("roi_landing",           "Stage2-roi-landing"),
            ("diffmap_takeoff",       "Stage3-edge-takeoff"),
            ("diffmap_landing",       "Stage3-edge-landing"),
            ("takeoff",               "Stage4-takeoff"),
            ("landing",               "Stage4-landing"),
            ("combined",              "Stage4-combined"),
        ]
        for mode, suffix in stages:
            img = self.diff_detector.render_result_image(mode=mode)
            if img is None:
                self._log("DIFF", f"差分照片[{suffix}] 生成失败，跳过")
                continue
            filename = os.path.join(self.images_diff_dir, f"diff-{suffix}.jpeg")
            imwrite_safe(filename, img)
        self._log("SAVE", "差分各阶段照片已保存: Stage1~Stage4")

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
                    if not _was_calibrated:
                        _was_calibrated = True
                        frame_idx = 0
                        self._log("CALIB", f"标定完成，帧计数器重置为0")
                    self._log("CALIB", "垫子标定完成")
                    # 保存两张垫子识别图（默认不输出）
                    if self.config.enable_mat_output and self.images_dir:
                        mask_quad = self.calibrator.render_mask(frame)
                        if mask_quad is not None:
                            imwrite_safe(os.path.join(self.images_dir, "mat_mask_quad.jpeg"), cv2.cvtColor(mask_quad, cv2.COLOR_GRAY2BGR))
                        mask_hsv = self.calibrator.render_hsv_mask(frame)
                        if mask_hsv is not None:
                            imwrite_safe(os.path.join(self.images_dir, "mat_mask_hsv.jpeg"), cv2.cvtColor(mask_hsv, cv2.COLOR_GRAY2BGR))
                        self._log("CALIB", "垫子识别图已保存: mat_mask_quad.jpeg (四边形拟合), mat_mask_hsv.jpeg (HSV原始)")

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
                            self._handle_idle_skeleton(ankle_cm, front_toe_cm)
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
                    self._handle_jumping_skeleton(frame_idx, heel_l_xy, heel_r_xy, heel_l_cm, heel_r_cm)
                    # 差分法：落地后保存落地帧（从主循环中调用，frame/kpts 有效）
                    if self.state == "LANDED" and self.diff_detector.has_base_frame:
                        self.diff_detector.save_landing_frame(frame, kpts)

                if self.calibrator.calibrated:
                    self._recalc_results_with_current_mat()

                # ── 差分法：落地后计算差分距离 ──
                if self.state == "LANDED" and not self._diff_computed and self.diff_detector.has_base_frame:
                    self._diff_computed = True
                    to_x, ld_x, dist = self.diff_detector.compute_combined_distance()
                    if to_x is not None and ld_x is not None:
                        self._log("DIFF",
                                  f"差分法结果: 起跳={to_x:.1f}cm, 落地={ld_x:.1f}cm, "
                                  f"距离={dist:.1f}cm")
                        self._save_diff_image()
                    else:
                        self._log("DIFF", "差分法计算失败（关键点或基准帧不足）")

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

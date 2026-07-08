"""轮廓检测模块：差分图生成、人体轮廓提取、起跳/落地坐标计算。"""
import cv2
import numpy as np


class ContourDetector:
    """处理差分和轮廓：基线帧、人体 mask、提取起跳/落地 X 坐标。

    通过绿色垫子反色提取垫子上的人体轮廓，支持基于轮廓面积骤降的起跳判定。
    """

    def __init__(self, calibrator):
        """
        参数:
            calibrator: MatCalibrator 实例，提供 _smooth_box、calibrated、
                        transform_to_mat_cm 等
        """
        self._calib = calibrator
        self._baseline_frame = None  # 标定后的第一帧（干净垫子），用于差分

    # ──────── 基线帧 & 差分图 ────────

    def set_baseline_frame(self, frame):
        """将标定后的第一帧保存为基线帧（干净垫子参考）。"""
        self._baseline_frame = frame.copy()

    def render_diff_image(self, frame, colormap=cv2.COLORMAP_JET):
        """生成垫子区域差分热力图：frame 与基线帧的差异。"""
        if self._baseline_frame is None or self._calib.smooth_box is None:
            return None

        cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        base_gray = cv2.cvtColor(self._baseline_frame, cv2.COLOR_BGR2GRAY)

        box_mask = np.zeros_like(cur_gray)
        cv2.fillPoly(box_mask, [self._calib.smooth_box.astype(np.int32)], 255)

        diff = cv2.absdiff(cur_gray, base_gray)
        diff = cv2.bitwise_and(diff, box_mask)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)

        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(diff_norm, colormap)

        result = frame.copy()
        heat_region = cv2.bitwise_and(heatmap, heatmap, mask=box_mask)
        cv2.addWeighted(heat_region, 0.6, result, 0.4, 0, result)
        cv2.polylines(result, [self._calib.smooth_box.astype(np.int32)], True, (255, 255, 255), 2)
        return result

    # ──────── 人体轮廓检测 ────────

    def get_person_mask(self, frame, morphology_close=True):
        """通过垫子绿色反色提取垫子上的人体轮廓二值图（白色=人体）。"""
        if frame is None or not self._calib.calibrated:
            return None
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, np.array([20, 30, 30]), np.array([95, 255, 255]))
        person = cv2.bitwise_not(green_mask)
        if self._calib.smooth_box is not None:
            box_mask = np.zeros_like(person)
            cv2.fillPoly(box_mask, [self._calib.smooth_box.astype(np.int32)], 255)
            person = cv2.bitwise_and(person, box_mask)
        kernel = np.ones((5, 5), np.uint8)
        person = cv2.morphologyEx(person, cv2.MORPH_OPEN, kernel, iterations=1)
        if morphology_close:
            person = cv2.morphologyEx(person, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(person, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            person = np.zeros_like(person)
            cv2.drawContours(person, [largest], -1, 255, -1)
        return person

    def get_person_area_px(self, frame):
        """返回垫子内人体轮廓的面积（像素数）。0 = 无人/人在空中。"""
        mask = self.get_person_mask(frame, morphology_close=False)
        if mask is None:
            return 0.0
        return float(cv2.countNonZero(mask))

    def get_person_front_x_cm(self, frame):
        """获取人体最靠前（X 最大=脚尖）的位置(cm)，用于起跳点确认。"""
        mask = self.get_person_mask(frame)
        if mask is None:
            return None
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        max_x_idx = np.argmax(xs)
        cm = self._calib.transform_to_mat_cm((float(xs[max_x_idx]), float(ys[max_x_idx])))
        return cm[0] if cm is not None else None

    def get_person_back_x_cm(self, frame):
        """获取人体最靠后（X 最小=脚后跟）的位置(cm)，用于落地判定。"""
        mask = self.get_person_mask(frame)
        if mask is None:
            return None
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        min_x_idx = np.argmin(xs)
        cm = self._calib.transform_to_mat_cm((float(xs[min_x_idx]), float(ys[min_x_idx])))
        return cm[0] if cm is not None else None

    def get_person_centroid_x_cm(self, frame):
        """获取人体轮廓重心 X (cm)。"""
        mask = self.get_person_mask(frame)
        if mask is None:
            return None
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        cm = self._calib.transform_to_mat_cm((cx, cy))
        return cm[0] if cm is not None else None

    def get_person_bottom_y_px(self, frame):
        """获取人体轮廓最底部（Y 最大）的像素坐标，用于判断脚是否离垫。"""
        mask = self.get_person_mask(frame, morphology_close=True)
        if mask is None:
            return None
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return None
        max_y_idx = np.argmax(ys)
        return float(ys[max_y_idx])

    def get_person_bottom_x_cm(self, frame):
        """获取人体轮廓最底部（Y 最大）处的 X(cm)，用于落地点。"""
        mask = self.get_person_mask(frame, morphology_close=True)
        if mask is None:
            return None
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return None
        max_y_idx = np.argmax(ys)
        cm = self._calib.transform_to_mat_cm((float(xs[max_y_idx]), float(ys[max_y_idx])))
        return cm[0] if cm is not None else None

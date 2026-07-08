"""鞋子边缘检测模块：基于 MediaPipe 关键点 + ROI Canny 边缘的局部微调。

专门解决骨骼关键点不精确（鞋子厚度）导致落地点/起跳点判定误差的问题。
"""
import cv2
import numpy as np


class ShoeEdgeDetector:
    """专注局部微调：依赖关键点切 ROI，做 Canny 边缘提取，精确获取鞋底/鞋尖位置。"""

    def __init__(self, calibrator):
        """
        参数:
            calibrator: MatCalibrator 实例，提供 transform_to_mat_cm
        """
        self._calib = calibrator

    # ──────── 静态工具 ────────

    @staticmethod
    def get_kpt(kpts, idx):
        """从 (33,2) numpy 数组中安全提取单关键点，返回 (x, y) 像素坐标，或 None。"""
        if kpts is None or idx < 0 or idx >= len(kpts):
            return None
        x, y = float(kpts[idx][0]), float(kpts[idx][1])
        if x == 0.0 and y == 0.0:
            return None
        return (x, y)

    # ──────── ROI 裁剪 ────────

    def _foot_roi(self, frame, kpts, foot_indices, expand=1.5):
        """根据关键点列表计算脚部 ROI 矩形。

        返回:
            (roi_gray, roi_origin_xy, roi_w, roi_h)
            或 None（关键点缺失）
        """
        pts_px = []
        for idx in foot_indices:
            kpt = self.get_kpt(kpts, idx)
            if kpt is None:
                continue
            pts_px.append((int(kpt[0]), int(kpt[1])))
        if len(pts_px) < 3:
            return None

        h, w = frame.shape[:2]
        xs = [p[0] for p in pts_px]
        ys = [p[1] for p in pts_px]
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        half_w = max((max(xs) - min(xs)) * expand, 40.0)
        half_h = max((max(ys) - min(ys)) * expand, 40.0)

        x1 = max(0, int(cx - half_w))
        y1 = max(0, int(cy - half_h * 0.6))
        x2 = min(w, int(cx + half_w))
        y2 = min(h, int(cy + half_h * 1.4))

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return roi_gray, (x1, y1), x2 - x1, y2 - y1

    # ──────── Canny 边缘 + 轮廓提取 ────────

    def detect_shoe_contour_px(self, roi_gray, roi_h=None):
        """对 ROI 灰度图做 Canny 边缘检测 + 轮廓筛选，返回鞋子轮廓像素点列表。

        返回:
            (contour_pixels, edge_img) 或 (None, None)
        """
        blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 31, 5)
        edges = cv2.Canny(binary, 30, 100)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 100:
            return None, None

        if roi_h is not None:
            max_y = max(p[0][1] for p in largest)
            if max_y < roi_h - 10:
                return None, None

        contour_mask = np.zeros_like(edges)
        cv2.drawContours(contour_mask, [largest], -1, 255, -1)
        ys, xs = np.where(contour_mask > 0)
        if len(xs) == 0:
            return None, None
        pixels = np.column_stack((xs, ys))

        edge_vis = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(edge_vis, [largest], -1, (0, 255, 0), 2)
        min_x_idx = np.argmin(xs)
        cv2.circle(edge_vis, (int(xs[min_x_idx]), int(ys[min_x_idx])), 4, (0, 0, 255), -1)

        return pixels, edge_vis

    # ──────── 落地 & 起跳点检测 ────────

    @property
    def _foot_configs(self):
        """左右脚关键点索引：(踝, 脚跟, 脚趾)"""
        return [
            {"label": "left",  "indices": [27, 29, 31]},
            {"label": "right", "indices": [28, 30, 32]},
        ]

    def get_shoe_landing_x_cm(self, frame, kpts):
        """基于 ROI 鞋子边缘检测的落地位置（脚后跟 X 最小点）。

        返回:
            (landing_x_cm, edge_vis) 或 (None, None)
        """
        best_heel_x_cm = None
        best_edge_vis = None

        for cfg in self._foot_configs:
            roi_result = self._foot_roi(frame, kpts, cfg["indices"], expand=1.5)
            if roi_result is None:
                continue
            roi_gray, origin_xy, roi_w, roi_h = roi_result
            contour_px, edge_vis = self.detect_shoe_contour_px(roi_gray, roi_h)
            if contour_px is None or edge_vis is None:
                continue

            abs_xs = contour_px[:, 0] + origin_xy[0]
            abs_ys = contour_px[:, 1] + origin_xy[1]

            min_idx = np.argmin(contour_px[:, 0])
            heel_px = (float(abs_xs[min_idx]), float(abs_ys[min_idx]))

            heel_cm = self._calib.transform_to_mat_cm(heel_px)
            if heel_cm is None or heel_cm[0] < 0 or heel_cm[0] > 350:
                continue

            if best_heel_x_cm is None or heel_cm[0] < best_heel_x_cm:
                best_heel_x_cm = heel_cm[0]
                best_edge_vis = edge_vis

        return best_heel_x_cm, best_edge_vis

    def detect_shoe_front_x_cm(self, frame, kpts):
        """基于 ROI 鞋子边缘检测的起跳点位置（脚尖 X 最大点）。

        返回:
            (front_x_cm, edge_vis) 或 (None, None)
        """
        best_front_x_cm = None
        best_edge_vis = None

        for cfg in self._foot_configs:
            roi_result = self._foot_roi(frame, kpts, cfg["indices"], expand=1.5)
            if roi_result is None:
                continue
            roi_gray, origin_xy, roi_w, roi_h = roi_result
            contour_px, edge_vis = self.detect_shoe_contour_px(roi_gray, roi_h)
            if contour_px is None or edge_vis is None:
                continue

            abs_xs = contour_px[:, 0] + origin_xy[0]
            abs_ys = contour_px[:, 1] + origin_xy[1]

            max_idx = np.argmax(contour_px[:, 0])
            toe_px = (float(abs_xs[max_idx]), float(abs_ys[max_idx]))

            toe_cm = self._calib.transform_to_mat_cm(toe_px)
            if toe_cm is None or toe_cm[0] < 0 or toe_cm[0] > 350:
                continue

            if best_front_x_cm is None or toe_cm[0] > best_front_x_cm:
                best_front_x_cm = toe_cm[0]
                best_edge_vis = edge_vis

        return best_front_x_cm, best_edge_vis

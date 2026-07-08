"""垫子标定模块：垫子检测、透视变换、坐标转换。"""
import cv2
import numpy as np
import os as _os

# 设置 ultralytics 缓存目录到项目内（避免 Windows 权限问题）
_os.environ.setdefault("ULTRALYTICS_SETTINGS_DIR",
                       _os.path.join(_os.path.dirname(__file__), "..", "..", ".ultralytics_cache"))
_os.makedirs(_os.environ["ULTRALYTICS_SETTINGS_DIR"], exist_ok=True)


class MatCalibrator:
    def __init__(self, mat_length_cm, mat_width_cm, manual_mode=False):
        self.mat_length_cm = float(mat_length_cm)
        self.mat_width_cm = float(mat_width_cm)
        self.manual_mode = bool(manual_mode)
        self.manual_points = []
        self.mat_locked = False
        self.calibrated = False
        self._smooth_box = None
        self._last_box_points = None
        self.H_img2mat = None
        self.H_mat2img = None
        self.jump_line_px = None
        self.mat_view_scale = 4.0
        self.px_per_cm = 0.0
        self._baseline_frame = None  # 标定后的第一帧（干净垫子），用于差分

    def mouse_callback(self, event, x, y, flags, param):
        if not self.manual_mode or self.mat_locked:
            return
        if event == cv2.EVENT_LBUTTONDOWN and len(self.manual_points) < 4:
            self.manual_points.append([float(x), float(y)])
            print(f"[Manual] Point {len(self.manual_points)} added: ({x}, {y})")
            if len(self.manual_points) == 4:
                print(">>> 手动标定完成！锁定区域。")
                self.mat_locked = True
                self._smooth_box = self.order_points(np.array(self.manual_points, dtype=np.float32))

    @staticmethod
    def order_points(pts):
        pts = np.array(pts, dtype=np.float32).reshape(-1, 2)
        y_sorted = pts[np.argsort(pts[:, 1])]
        top = y_sorted[:2]
        bottom = y_sorted[2:]
        tl, tr = top[np.argsort(top[:, 0])]
        bl, br = bottom[np.argsort(bottom[:, 0])]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def detect_mat_box(self, frame):
        if self.mat_locked and self._smooth_box is not None:
            return self._smooth_box
        if self.manual_mode:
            return None

        h_img, w_img = frame.shape[:2]
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        lower_green = np.array([20, 30, 30])
        upper_green = np.array([95, 255, 255])
        mat_mask = cv2.inRange(hsv, lower_green, upper_green)

        kernel = np.ones((5, 5), np.uint8)
        mat_mask = cv2.morphologyEx(mat_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(mat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self._smooth_box

        ratio_target = self.mat_length_cm / self.mat_width_cm if self.mat_width_cm > 1e-6 else 1.0
        best_quad = None
        best_key = None

        for contour in contours:
            if cv2.contourArea(contour) < (h_img * w_img * 0.015):
                continue
            hull = cv2.convexHull(contour)
            peri = cv2.arcLength(hull, True)
            quad = None
            for factor in [0.01, 0.02, 0.03, 0.05]:
                approx = cv2.approxPolyDP(hull, factor * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    quad = approx.reshape(4, 2).astype(np.float32)
                    break
            if quad is None:
                rect = cv2.minAreaRect(contour)
                quad = cv2.boxPoints(rect).astype(np.float32)
            quad = self.order_points(quad)
            edges = [float(np.linalg.norm(quad[(idx + 1) % 4] - quad[idx])) for idx in range(4)]
            short_edge = min(edges)
            if short_edge < 10.0:
                continue
            ratio = max(edges) / min(edges)
            area = cv2.contourArea(quad.reshape(-1, 1, 2))
            key = (abs(ratio - ratio_target), -area)
            if best_key is None or key < best_key:
                best_key = key
                best_quad = quad

        if best_quad is None:
            return self._smooth_box

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        refined = cv2.cornerSubPix(gray, best_quad, (11, 11), (-1, -1), criteria)
        if self._smooth_box is None:
            self._smooth_box = refined
        else:
            diff = np.linalg.norm(refined - self._smooth_box)
            alpha = 0.1
            if diff > 50.0:
                alpha = 0.5
            elif diff < 2.0:
                alpha = 0.05
            self._smooth_box = (alpha * refined + (1.0 - alpha) * self._smooth_box).astype(np.float32)
        return self._smooth_box

    def build_homography(self, box_points):
        quad = self.order_points(box_points)
        tl, tr, br, bl = quad
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = max(int(width_a), int(width_b))
        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = max(int(height_a), int(height_b))
        img_ratio = max_width / float(max_height) if max_height > 0 else 1.0

        src = np.array([tl, tr, br, bl], dtype=np.float32)
        if img_ratio >= 1.0:
            dst = np.array(
                [[0, 0], [self.mat_length_cm, 0], [self.mat_length_cm, self.mat_width_cm], [0, self.mat_width_cm]],
                dtype=np.float32,
            )
            line_p1, line_p2 = tl, bl
        else:
            dst = np.array(
                [[0, 0], [self.mat_width_cm, 0], [self.mat_width_cm, self.mat_length_cm], [0, self.mat_length_cm]],
                dtype=np.float32,
            )
            line_p1, line_p2 = tl, tr

        H = cv2.getPerspectiveTransform(src, dst)
        H_inv = np.linalg.inv(H)
        vec = line_p2 - line_p1
        normal = np.array([-vec[1], vec[0]], dtype=np.float32)
        c = -np.dot(normal, line_p1)
        jump_line_px = (normal[0], normal[1], c)
        return H, H_inv, jump_line_px

    def update(self, frame):
        box_points = self.detect_mat_box(frame)
        if box_points is None:
            return False
        H, H_inv, jump_line_px = self.build_homography(box_points)
        self.H_img2mat = H
        self.H_mat2img = H_inv
        self.jump_line_px = jump_line_px
        verts = self.order_points(box_points)
        len0 = float(np.linalg.norm(verts[1] - verts[0]))
        len2 = float(np.linalg.norm(verts[3] - verts[2]))
        self.px_per_cm = ((len0 + len2) * 0.5) / self.mat_length_cm
        self.calibrated = True
        self._last_box_points = box_points
        return True

    def transform_to_mat_cm(self, pt_xy):
        if self.H_img2mat is None or pt_xy is None:
            return None
        src = np.array([[[float(pt_xy[0]), float(pt_xy[1])]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, self.H_img2mat)[0][0]
        return float(dst[0]), float(dst[1])

    def strict_in_mat(self, xy_cm):
        if xy_cm is None:
            return False
        x, y = xy_cm
        return (-5.0 <= x <= self.mat_length_cm + 5.0) and (-5.0 <= y <= self.mat_width_cm + 5.0)

    def in_mat(self, xy_cm):
        if xy_cm is None:
            return False
        x, y = xy_cm
        return (-50.0 <= x <= self.mat_length_cm) and (-50.0 <= y <= self.mat_width_cm + 50.0)

    def get_H_img2mat_px(self):
        if self.H_img2mat is None:
            return None
        scale = float(self.mat_view_scale)
        S = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        return S @ self.H_img2mat

    def render_mask(self, frame):
        """生成 2 色垫子识别图（四边形拟合后）：垫子区域=白色(255), 背景=黑色(0)。"""
        if frame is None:
            return None
        mask = self._compute_hsv_mask(frame)
        # 仅在已标定框内保留白色
        if self._smooth_box is not None:
            box_img = np.zeros_like(mask)
            cv2.fillPoly(box_img, [self._smooth_box.astype(np.int32)], 255)
            mask = cv2.bitwise_and(mask, box_img)
        return mask

    def render_hsv_mask(self, frame):
        """生成原始 HSV 颜色分割二值图（无四边形裁剪）。"""
        if frame is None:
            return None
        return self._compute_hsv_mask(frame)

    def _compute_hsv_mask(self, frame):
        """HSV 颜色分割 + 形态学去噪。"""
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        lower_green = np.array([20, 30, 30])
        upper_green = np.array([95, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    # ──────── 基线帧 & 差分图 ────────

    def set_baseline_frame(self, frame):
        """将标定后的第一帧保存为基线帧（干净垫子参考）。"""
        self._baseline_frame = frame.copy()

    def render_diff_image(self, frame, colormap=cv2.COLORMAP_JET):
        """生成垫子区域差分热力图：frame 与基线帧的差异。

        返回：
          - 彩色热力图（BGR），垫子外区域保持原图
          - 若基线帧未设置，返回 None
        """
        if self._baseline_frame is None or self._smooth_box is None:
            return None

        # 转灰度
        cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        base_gray = cv2.cvtColor(self._baseline_frame, cv2.COLOR_BGR2GRAY)

        # 垫子区域遮罩
        box_mask = np.zeros_like(cur_gray)
        cv2.fillPoly(box_mask, [self._smooth_box.astype(np.int32)], 255)

        # 差异绝对值（仅垫子内）
        diff = cv2.absdiff(cur_gray, base_gray)
        diff = cv2.bitwise_and(diff, box_mask)

        # 高斯模糊去噪
        diff = cv2.GaussianBlur(diff, (5, 5), 0)

        # 归一化 → 伪彩色热力图
        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(diff_norm, colormap)

        # 与原图叠加（仅垫子区域）
        result = frame.copy()
        heat_region = cv2.bitwise_and(heatmap, heatmap, mask=box_mask)
        cv2.addWeighted(heat_region, 0.6, result, 0.4, 0, result)

        # 绘制垫子轮廓
        cv2.polylines(result, [self._smooth_box.astype(np.int32)], True, (255, 255, 255), 2)
        return result

    # ──────── 人体轮廓检测（替代骨架关键点） ────────

    def get_person_mask(self, frame, morphology_close=True):
        """通过垫子绿色反色提取垫子上的人体轮廓二值图（白色=人体）。"""
        if frame is None or not self.calibrated:
            return None
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, np.array([20, 30, 30]), np.array([95, 255, 255]))
        person = cv2.bitwise_not(green_mask)  # 非绿色 = 人体/衣物/鞋子
        # 仅保留垫子区域内的非绿色像素
        if self._smooth_box is not None:
            box_mask = np.zeros_like(person)
            cv2.fillPoly(box_mask, [self._smooth_box.astype(np.int32)], 255)
            person = cv2.bitwise_and(person, box_mask)
        kernel = np.ones((5, 5), np.uint8)
        person = cv2.morphologyEx(person, cv2.MORPH_OPEN, kernel, iterations=1)
        if morphology_close:
            person = cv2.morphologyEx(person, cv2.MORPH_CLOSE, kernel, iterations=1)
        # 去掉小噪点：只保留最大轮廓
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
        # 最靠前的点 = X 最大的像素
        max_x_idx = np.argmax(xs)
        cm = self.transform_to_mat_cm((float(xs[max_x_idx]), float(ys[max_x_idx])))
        return cm[0] if cm is not None else None

    def get_person_back_x_cm(self, frame):
        """获取人体最靠后（X 最小=脚后跟）的位置(cm)，用于落地判定。"""
        mask = self.get_person_mask(frame)
        if mask is None:
            return None
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        # 最靠后的点 = X 最小的像素（靠近起跳线侧）
        min_x_idx = np.argmin(xs)
        cm = self.transform_to_mat_cm((float(xs[min_x_idx]), float(ys[min_x_idx])))
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
        cm = self.transform_to_mat_cm((cx, cy))
        return cm[0] if cm is not None else None

    def get_person_bottom_y_px(self, frame):
        """获取人体轮廓最底部（Y 最大）的像素坐标，用于判断脚是否离垫。"""
        mask = self.get_person_mask(frame, morphology_close=True)
        if mask is None:
            return None
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return None
        max_y_idx = np.argmax(ys)  # Y 最大 = 图像底部 = 脚的位置
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
        cm = self.transform_to_mat_cm((float(xs[max_y_idx]), float(ys[max_y_idx])))
        return cm[0] if cm is not None else None

    # ──────── ROI 鞋子边缘检测（基于 MediaPipe 关键点） ────────

    def _foot_roi(self, frame, kpts, foot_indices, expand=1.5, return_bgr=False):
        """根据关键点列表计算脚部 ROI 矩形。

        参数:
            frame: 原图 (BGR)
            kpts: MediaPipe 关键点 (NormalizedLandmarkList)
            foot_indices: 关键点索引列表，如 [27, 29, 31] (左踝+脚跟+脚趾)
            expand: ROI 扩大倍数
            return_bgr: 若为 True，返回 BGR 彩色图；否则返回灰度图

        返回:
            (roi_img, roi_origin_xy, roi_w, roi_h)
            - roi_img: 裁剪出的 ROI 灰度图（默认）或 BGR 图（return_bgr=True）
            - roi_origin_xy: ROI 在原图的左上角 (x, y)
            - roi_w, roi_h: ROI 宽高
            - None 如果关键点缺失
        """
        pts_px = []
        for idx in foot_indices:
            kpt = self._get_kpt(kpts, idx)
            if kpt is None:
                continue
            pts_px.append((int(kpt[0]), int(kpt[1])))
        if len(pts_px) < 2:
            return None

        h, w = frame.shape[:2]
        xs = [p[0] for p in pts_px]
        ys = [p[1] for p in pts_px]
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        half_w = max((max(xs) - min(xs)) * expand, 40.0)
        half_h = max((max(ys) - min(ys)) * expand, 40.0)

        x1 = max(0, int(cx - half_w))
        y1 = max(0, int(cy - half_h * 0.6))  # 向上多留（脚踝以上）
        x2 = min(w, int(cx + half_w))
        y2 = min(h, int(cy + half_h * 1.4))  # 向下多留（脚底）

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        if return_bgr:
            return roi, (x1, y1), x2 - x1, y2 - y1
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return roi_gray, (x1, y1), x2 - x1, y2 - y1

    @staticmethod
    def _get_kpt(kpts, idx):
        """从 (33,2) numpy 数组中安全提取单关键点。

        返回 (x, y) 像素坐标，或 None。
        """
        if kpts is None or idx < 0 or idx >= len(kpts):
            return None
        x, y = float(kpts[idx][0]), float(kpts[idx][1])
        if x == 0.0 and y == 0.0:
            return None
        return (x, y)

    def detect_shoe_contour_px(self, roi_gray, roi_h=None, return_steps=False):
        """对 ROI 灰度图做 Canny 边缘检测 + 轮廓筛选，返回鞋子轮廓像素点列表。

        步骤：
          1. 高斯模糊去噪
          2. 自适应二值化（分离鞋底/垫面）
          3. Canny 边缘检测
          4. 形态学闭运算补全边缘缺口
          5. 筛选面积最大闭合轮廓
          6. 若 roi_h 提供，只保留底部贴地（轮廓底部接近 ROI 底部）的轮廓

        返回:
            (contour_pixels, edge_img) — contour_pixels 是 (N,2) 的像素坐标数组（相对 ROI），
            edge_img 是可视化边缘图（调试用）。找不到返回 (None, None)。
        """
        steps = None
        if return_steps:
            steps = {"roi_gray": roi_gray}

        blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)
        if steps is not None:
            steps["blurred"] = blurred

        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            5,
        )
        if steps is not None:
            steps["binary"] = binary

        edges_raw = cv2.Canny(binary, 30, 100)
        if steps is not None:
            steps["edges_raw"] = edges_raw

        kernel = np.ones((5, 5), np.uint8)
        edges_closed = cv2.morphologyEx(edges_raw, cv2.MORPH_CLOSE, kernel, iterations=2)
        if steps is not None:
            steps["edges_closed"] = edges_closed

        contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            if return_steps:
                return None, None, steps
            return None, None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 100:
            if return_steps:
                return None, None, steps
            return None, None

        # 落地过滤：轮廓底部必须接近 ROI 底部（鞋子贴地）
        if roi_h is not None:
            max_y = max(p[0][1] for p in largest)  # 轮廓底部 Y（ROI 相对坐标）
            if max_y < roi_h - 10:  # 底部距离 ROI 底边超过10px → 在空中，非落地
                if return_steps:
                    return None, None, steps
                return None, None

        contour_mask = np.zeros_like(edges_closed)
        cv2.drawContours(contour_mask, [largest], -1, 255, -1)
        if steps is not None:
            steps["contour_mask"] = contour_mask
        ys, xs = np.where(contour_mask > 0)
        if len(xs) == 0:
            if return_steps:
                return None, None, steps
            return None, None
        pixels = np.column_stack((xs, ys))

        edge_vis = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(edge_vis, [largest], -1, (0, 255, 0), 2)
        # 标注后跟点（X 最小点）
        min_x_idx = np.argmin(xs)
        cv2.circle(edge_vis, (int(xs[min_x_idx]), int(ys[min_x_idx])), 4, (0, 0, 255), -1)
        if steps is not None:
            steps["edge_vis"] = edge_vis

        if return_steps:
            return pixels, edge_vis, steps
        return pixels, edge_vis

    def get_shoe_landing_x_cm(self, frame, kpts, return_steps=False):
        """基于 ROI 鞋子边缘检测的落地位置。

        取脚后跟轮廓中 X 最小的点（最靠近起跳线），换算到垫子坐标系。

        返回:
            (landing_x_cm, edge_vis) 或 (None, None)
            edge_vis 是 ROI 可视化边缘图（用于 debug 保存）
        """
        # 左右脚关键点索引
        foot_configs = [
            {"label": "left", "indices": [27, 29, 31]},   # 左踝+脚跟+脚趾
            {"label": "right", "indices": [28, 30, 32]},  # 右踝+脚跟+脚趾
        ]

        best_heel_x_cm = None
        best_edge_vis = None
        best_debug = None

        for cfg in foot_configs:
            roi_result = self._foot_roi(frame, kpts, cfg["indices"], expand=1.5)
            if roi_result is None:
                continue
            roi_gray, origin_xy, roi_w, roi_h = roi_result
            x1, y1 = int(origin_xy[0]), int(origin_xy[1])
            x2, y2 = x1 + int(roi_w), y1 + int(roi_h)

            if return_steps:
                contour_px, edge_vis, steps = self.detect_shoe_contour_px(roi_gray, roi_h, return_steps=True)
            else:
                contour_px, edge_vis = self.detect_shoe_contour_px(roi_gray, roi_h, return_steps=False)
                steps = None

            if return_steps and steps is not None and contour_px is None:
                frame_roi = frame.copy()
                cv2.rectangle(frame_roi, (x1, y1), (x2, y2), (0, 255, 255), 2)
                fallback_debug = {
                    "label": cfg["label"],
                    "roi_rect": (x1, y1, x2, y2),
                    "frame_roi": frame_roi,
                    "steps": steps,
                }

            if contour_px is None or edge_vis is None:
                if return_steps and steps is not None and fallback_debug is None:
                    frame_roi = frame.copy()
                    cv2.rectangle(frame_roi, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    fallback_debug = {
                        "label": f"{cfg['label']}-fallback",
                        "roi_rect": (x1, y1, x2, y2),
                        "frame_roi": frame_roi,
                        "steps": steps,
                    }
                continue

            # 把 ROI 相对坐标 → 原图绝对坐标
            abs_xs = contour_px[:, 0] + origin_xy[0]
            abs_ys = contour_px[:, 1] + origin_xy[1]

            # 取 X 最小的点（后跟最后沿 ≈ 最靠近起跳线）
            min_idx = np.argmin(contour_px[:, 0])  # 在 ROI 内 X 最小的点
            heel_px = (float(abs_xs[min_idx]), float(abs_ys[min_idx]))

            # 换算垫子坐标
            heel_cm = self.transform_to_mat_cm(heel_px)
            if heel_cm is None or heel_cm[0] < 0 or heel_cm[0] > 350:
                continue

            if best_heel_x_cm is None or heel_cm[0] < best_heel_x_cm:
                best_heel_x_cm = heel_cm[0]
                best_edge_vis = edge_vis
                if return_steps:
                    frame_roi = frame.copy()
                    cv2.rectangle(frame_roi, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    best_debug = {
                        "label": cfg["label"],
                        "roi_rect": (x1, y1, x2, y2),
                        "frame_roi": frame_roi,
                        "steps": steps,
                    }

        if return_steps:
            return best_heel_x_cm, best_edge_vis, best_debug
        return best_heel_x_cm, best_edge_vis

    def detect_shoe_front_x_cm(self, frame, kpts, return_steps=False):
        """基于 ROI 鞋子边缘检测的起跳点位置。

        取脚尖轮廓中 X 最大的点（离起跳线最远），换算到垫子坐标系。

        返回:
            (front_x_cm, edge_vis) 或 (None, None)
        """
        foot_configs = [
            {"label": "left", "indices": [27, 29, 31]},
            {"label": "right", "indices": [28, 30, 32]},
        ]

        best_front_x_cm = None
        best_edge_vis = None
        best_debug = None
        fallback_debug = None

        for cfg in foot_configs:
            roi_result = self._foot_roi(frame, kpts, cfg["indices"], expand=1.5)
            if roi_result is None:
                if return_steps and fallback_debug is None:
                    fallback_debug = {
                        "label": f"{cfg['label']}-no-roi",
                        "frame_roi": frame.copy(),
                        "error": "_foot_roi returned None",
                        "steps": {},
                    }
                continue
            roi_gray, origin_xy, roi_w, roi_h = roi_result
            x1, y1 = int(origin_xy[0]), int(origin_xy[1])
            x2, y2 = x1 + int(roi_w), y1 + int(roi_h)

            if return_steps:
                contour_px, edge_vis, steps = self.detect_shoe_contour_px(roi_gray, roi_h, return_steps=True)
            else:
                contour_px, edge_vis = self.detect_shoe_contour_px(roi_gray, roi_h, return_steps=False)
                steps = None

            if contour_px is None or edge_vis is None:
                if return_steps and steps is not None and fallback_debug is None:
                    frame_roi = frame.copy()
                    cv2.rectangle(frame_roi, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    fallback_debug = {
                        "label": f"{cfg['label']}-fallback",
                        "roi_rect": (x1, y1, x2, y2),
                        "frame_roi": frame_roi,
                        "steps": steps,
                    }
                continue

            abs_xs = contour_px[:, 0] + origin_xy[0]
            abs_ys = contour_px[:, 1] + origin_xy[1]

            # 取 X 最大的点（脚尖最前沿）
            max_idx = np.argmax(contour_px[:, 0])
            toe_px = (float(abs_xs[max_idx]), float(abs_ys[max_idx]))

            toe_cm = self.transform_to_mat_cm(toe_px)
            if toe_cm is None or toe_cm[0] < 0 or toe_cm[0] > 350:
                continue

            if best_front_x_cm is None or toe_cm[0] > best_front_x_cm:
                best_front_x_cm = toe_cm[0]
                best_edge_vis = edge_vis
                if return_steps:
                    frame_roi = frame.copy()
                    cv2.rectangle(frame_roi, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    best_debug = {
                        "label": cfg["label"],
                        "roi_rect": (x1, y1, x2, y2),
                        "frame_roi": frame_roi,
                        "steps": steps,
                    }

        if return_steps:
            return best_front_x_cm, best_edge_vis, best_debug or fallback_debug
        return best_front_x_cm, best_edge_vis

    # ──────── YOLO 实例分割（鞋底检测） ────────

    _yolo_model = None

    @classmethod
    def _get_yolo_model(cls):
        """懒加载 YOLOv8-seg 模型（YOLOv10 无官方 seg 权重，用 v8-seg 替代）。"""
        if cls._yolo_model is None:
            from ultralytics import YOLO
            cls._yolo_model = YOLO("yolov8n-seg.pt", verbose=False)
        return cls._yolo_model

    def detect_shoe_yolo_seg(self, roi_bgr, roi_origin_xy, return_steps=False):
        """用 YOLOv10-seg 在 ROI 内做实例分割，提取鞋子后跟 X。

        步骤：
          1. YOLO seg 推理 → 获取 masks
          2. 筛选 ROI 底部区域内的 mask（鞋子贴地）
          3. 取 X 最小的 mask 像素点 → 后跟坐标
          4. 换算到垫子坐标系

        返回: (landing_x_cm, edge_vis, [steps_dict]) 或 (None, None, None)
        """
        steps = {} if return_steps else None
        try:
            model = self._get_yolo_model()
            results = model(roi_bgr, verbose=False, conf=0.3, iou=0.5)
        except Exception as e:
            if return_steps:
                steps["error"] = str(e)
                fallback = self._make_fallback_vis(roi_bgr, roi_origin_xy)
                return None, fallback, steps
            return None, None

        if steps is not None:
            steps["frame_roi"] = roi_bgr.copy()

        h, w = roi_bgr.shape[:2]
        best_heel_x_cm = None
        best_vis = None
        total_masks = 0
        passed_bottom = 0

        for r in results:
            if r.masks is None:
                continue
            for seg_idx in range(len(r.masks)):
                total_masks += 1
                mask = r.masks.data[seg_idx].cpu().numpy()
                mask_h, mask_w = mask.shape
                if mask_h != h or mask_w != w:
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                mask_bin = (mask > 0.5).astype(np.uint8) * 255

                ys, xs = np.where(mask_bin > 0)
                if len(ys) == 0:
                    continue
                if float(np.max(ys)) < h - 10:
                    continue
                passed_bottom += 1

                min_x_idx = np.argmin(xs)
                heel_roi_x = float(xs[min_x_idx])
                heel_roi_y = float(ys[min_x_idx])

                heel_px = (heel_roi_x + roi_origin_xy[0], heel_roi_y + roi_origin_xy[1])
                heel_cm = self.transform_to_mat_cm(heel_px)
                if heel_cm is None or heel_cm[0] < 0 or heel_cm[0] > 350:
                    continue

                if best_heel_x_cm is None or heel_cm[0] < best_heel_x_cm:
                    best_heel_x_cm = heel_cm[0]
                    vis = roi_bgr.copy()
                    overlay = np.zeros_like(vis)
                    overlay[mask_bin > 0] = (0, 255, 0)
                    vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
                    cv2.circle(vis, (int(heel_roi_x), int(heel_roi_y)), 5, (0, 0, 255), -1)
                    cls_id = int(r.boxes.data[seg_idx][5]) if r.boxes is not None else -1
                    conf = float(r.boxes.data[seg_idx][4]) if r.boxes is not None else 0.0
                    cls_name = r.names.get(cls_id, f"cls{cls_id}") if r.names else f"cls{cls_id}"
                    cv2.putText(vis, f"{cls_name} {conf:.2f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    best_vis = vis

        if return_steps:
            if best_vis is not None:
                steps["edge_vis"] = best_vis
            steps["yolo_masks"] = total_masks
            steps["yolo_bottom"] = passed_bottom
            return best_heel_x_cm, best_vis, steps
        return best_heel_x_cm, best_vis

    def _make_fallback_vis(self, roi_bgr, roi_origin_xy):
        fallback = roi_bgr.copy()
        cv2.rectangle(fallback, (0, 0), (roi_bgr.shape[1] - 1, roi_bgr.shape[0] - 1), (0, 255, 255), 2)
        cv2.putText(fallback, "YOLO FAILED", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return fallback

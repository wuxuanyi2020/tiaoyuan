"""垫子标定模块：垫子检测、透视变换、坐标转换。"""
import cv2
import numpy as np


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

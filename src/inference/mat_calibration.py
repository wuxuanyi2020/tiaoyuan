"""垫子标定模块：垫子检测、透视变换、坐标转换。

改进点：
1) 颜色提案不再只依赖单一 HSV 绿色阈值，加入低饱和/反光/阴影下的绿色灰度判定；
2) 检测四边形时使用“水平闭运算”的检测掩码，把被反光/遮挡切断的垫子段合并；
3) render_mask 输出真正的实心二值垫子识别图：垫子=255，背景=0。
"""
import cv2
import numpy as np


class MatCalibrator:
    """专注静态标定：找垫子轮廓、生成透视变换矩阵、坐标转换工具函数。"""

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

    @property
    def smooth_box(self):
        return self._smooth_box

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

    @staticmethod
    def _odd(value):
        value = int(max(3, value))
        return value if value % 2 == 1 else value + 1

    def detect_mat_box(self, frame):
        if self.mat_locked and self._smooth_box is not None:
            return self._smooth_box
        if self.manual_mode:
            return None

        best_quad = self._detect_quad_candidate(frame, hue_low=25)
        if best_quad is None:
            return self._smooth_box

        # HDR 视频经 OpenCV 解码后通常比 SDR 截图更亮、饱和度更低。
        # 1-16 这类强光帧里，垫子左上方的黄绿/灰白区域会被 H=25 的边界误合并，
        # 表现为“远端上边”几乎和近端下边一样长。遇到这种透视异常时，
        # 用稍严格的 H>=26 掩码重检一次，只替换这类明显外扩的结果。
        if self._is_hdr_like_frame(frame) and self._top_bottom_width_ratio(best_quad) > 0.88:
            strict_quad = self._detect_quad_candidate(frame, hue_low=26)
            if strict_quad is not None:
                raw_ratio = self._top_bottom_width_ratio(best_quad)
                strict_ratio = self._top_bottom_width_ratio(strict_quad)
                raw_area = cv2.contourArea(best_quad.reshape(-1, 1, 2))
                strict_area = cv2.contourArea(strict_quad.reshape(-1, 1, 2))
                if (0.55 <= strict_ratio <= 0.88
                        and strict_ratio < raw_ratio - 0.04
                        and strict_area >= raw_area * 0.70):
                    best_quad = strict_quad

        if self._smooth_box is None:
            self._smooth_box = best_quad
        else:
            diff = np.linalg.norm(best_quad - self._smooth_box)
            alpha = 0.1
            if diff > 50.0:
                alpha = 0.5
            elif diff < 2.0:
                alpha = 0.05
            self._smooth_box = (alpha * best_quad + (1.0 - alpha) * self._smooth_box).astype(np.float32)
        return self._smooth_box

    @staticmethod
    def _top_bottom_width_ratio(quad):
        quad = MatCalibrator.order_points(quad)
        top_w = float(np.linalg.norm(quad[1] - quad[0]))
        bottom_w = float(np.linalg.norm(quad[2] - quad[3]))
        return top_w / max(bottom_w, 1.0)

    @staticmethod
    def _is_hdr_like_frame(frame):
        """判断是否是强光/HDR 解码特征：下半区整体偏亮且饱和度被洗低。"""
        if frame is None:
            return False
        h_img = frame.shape[0]
        lower = frame[int(h_img * 0.5):, :]
        hsv = cv2.cvtColor(cv2.GaussianBlur(lower, (5, 5), 0), cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1].astype(np.float32)
        v = hsv[:, :, 2].astype(np.float32)
        return float(np.mean(v)) > 160.0 and float(np.mean(s)) < 45.0

    def _detect_quad_candidate(self, frame, hue_low=25):
        h_img, w_img = frame.shape[:2]
        mat_mask = self._compute_detection_mask(frame, hue_low=hue_low)
        contours, _ = cv2.findContours(mat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best_quad = None
        best_score = -1e18
        min_area = h_img * w_img * 0.01

        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area < min_area:
                continue

            x, y, ww, hh = cv2.boundingRect(contour)
            if hh < h_img * 0.035 or ww < w_img * 0.20:
                continue
            # 垫子在测试画面中是下半部分的长条透视四边形。
            if (y + 0.5 * hh) < h_img * 0.50:
                continue
            aspect = ww / float(max(hh, 1))
            if aspect < 3.0:
                continue

            hull = cv2.convexHull(contour)
            peri = cv2.arcLength(hull, True)
            hull_area = max(cv2.contourArea(hull), 1.0)
            quad = None
            # 从小到大放松近似精度；优先保留贴合轮廓的 4 点凸多边形。
            for factor in [0.003, 0.005, 0.008, 0.010, 0.012, 0.015, 0.020, 0.030]:
                approx = cv2.approxPolyDP(hull, factor * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    candidate = approx.reshape(4, 2).astype(np.float32)
                    candidate_area = cv2.contourArea(candidate.reshape(-1, 1, 2))
                    if candidate_area >= hull_area * 0.72:
                        quad = candidate
                        break
            if quad is None:
                rect = cv2.minAreaRect(contour)
                quad = cv2.boxPoints(rect).astype(np.float32)

            quad = self.order_points(quad)
            q_area = cv2.contourArea(quad.reshape(-1, 1, 2))
            if q_area < min_area:
                continue

            # 打分：面积越大越好、长条透视形状越好、中心越接近下半区越好。
            qx, qy, qw, qh = cv2.boundingRect(quad.astype(np.int32))
            q_aspect = qw / float(max(qh, 1))
            center_y = qy + 0.5 * qh
            score = q_area + 2500.0 * min(q_aspect, 12.0) - 0.15 * abs(center_y - h_img * 0.66) ** 2
            if score > best_score:
                best_score = score
                best_quad = quad

        if best_quad is None:
            return None

        # 不强依赖 cornerSubPix；自然图像角点常被反光/阴影扰动，线约束的四边形更稳定。
        best_quad[:, 0] = np.clip(best_quad[:, 0], 0, w_img - 1)
        best_quad[:, 1] = np.clip(best_quad[:, 1], 0, h_img - 1)
        return self._refine_quad_with_edge_lines(frame, best_quad.astype(np.float32))

    @staticmethod
    def _norm_angle_deg(angle):
        angle = (float(angle) + 180.0) % 180.0
        if angle > 90.0:
            angle -= 180.0
        return angle

    @staticmethod
    def _angle_diff_deg(a, b):
        diff = abs(float(a) - float(b))
        return min(diff, 180.0 - diff)

    @staticmethod
    def _cross2(a, b):
        return float(a[0] * b[1] - a[1] * b[0])

    @staticmethod
    def _fit_line(points):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        if len(points) < 2:
            return None
        line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
        return line.astype(np.float32)  # vx, vy, x0, y0

    @staticmethod
    def _intersect_lines(line_a, line_b):
        vx1, vy1, x1, y1 = [float(v) for v in line_a]
        vx2, vy2, x2, y2 = [float(v) for v in line_b]
        A = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float64)
        b = np.array([x2 - x1, y2 - y1], dtype=np.float64)
        det = float(np.linalg.det(A))
        if abs(det) < 1e-10:
            return None
        t, _ = np.linalg.solve(A, b)
        return np.array([x1 + t * vx1, y1 + t * vy1], dtype=np.float32)

    def _refine_quad_with_edge_lines(self, frame, quad):
        """用原图边缘线轻量精修四个角点。

        这一层只做“微调”，不再让拟合线把角点拉飞。
        对每条边只挑最可信的一组线段，并且最终位移超过阈值就放弃。
        """
        quad = self.order_points(quad).astype(np.float32)
        h_img, w_img = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        poly = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.fillPoly(poly, [quad.astype(np.int32)], 255)
        band_size = self._odd(min(w_img, h_img) * 0.028)
        band_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (band_size, band_size))
        band = cv2.subtract(cv2.dilate(poly, band_kernel), cv2.erode(poly, band_kernel))

        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        edges = cv2.bitwise_and(edges, band)
        min_line_len = max(60, int(w_img * 0.030))
        raw_lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=40,
            minLineLength=min_line_len,
            maxLineGap=25,
        )
        if raw_lines is None:
            return quad
        if raw_lines.ndim == 3:
            raw_lines = raw_lines[:, 0, :]

        segments = []
        for x1, y1, x2, y2 in raw_lines.astype(np.float32):
            dx, dy = x2 - x1, y2 - y1
            length = float(np.hypot(dx, dy))
            if length < min_line_len:
                continue
            angle = self._norm_angle_deg(np.degrees(np.arctan2(dy, dx)))
            midpoint = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
            segments.append((angle, midpoint, length, (x1, y1, x2, y2)))
        if not segments:
            return quad

        fitted_lines = []
        supported_edges = 0
        dist_limit = max(18.0, min(w_img, h_img) * 0.024)
        angle_limit = 18.0

        for edge_idx in range(4):
            p1 = quad[edge_idx]
            p2 = quad[(edge_idx + 1) % 4]
            edge_vec = p2 - p1
            edge_len2 = float(np.dot(edge_vec, edge_vec))
            if edge_len2 < 1.0:
                fitted_lines.append(self._fit_line([p1, p2]))
                continue
            edge_len = float(np.sqrt(edge_len2))
            edge_angle = self._norm_angle_deg(np.degrees(np.arctan2(edge_vec[1], edge_vec[0])))

            best_key = None
            best_seg = None
            for seg_angle, midpoint, seg_len, seg in segments:
                angle_diff = self._angle_diff_deg(seg_angle, edge_angle)
                if angle_diff > angle_limit:
                    continue
                dist = abs(self._cross2(edge_vec, midpoint - p1)) / edge_len
                if dist > dist_limit:
                    continue
                t = float(np.dot(midpoint - p1, edge_vec) / edge_len2)
                if t < -0.18 or t > 1.18:
                    continue
                key = (seg_len, -dist, -angle_diff, -abs(t - 0.5))
                if best_key is None or key > best_key:
                    best_key = key
                    best_seg = seg

            if best_seg is None:
                fitted_lines.append(self._fit_line([p1, p2]))
                continue

            x1, y1, x2, y2 = best_seg
            line = self._fit_line([[x1, y1], [x2, y2]])
            if line is None:
                line = self._fit_line([p1, p2])
            else:
                supported_edges += 1
            fitted_lines.append(line)

        if any(line is None for line in fitted_lines) or supported_edges < 2:
            return quad

        refined = [
            self._intersect_lines(fitted_lines[3], fitted_lines[0]),
            self._intersect_lines(fitted_lines[0], fitted_lines[1]),
            self._intersect_lines(fitted_lines[1], fitted_lines[2]),
            self._intersect_lines(fitted_lines[2], fitted_lines[3]),
        ]
        if any(pt is None for pt in refined):
            return quad
        refined = np.array(refined, dtype=np.float32)
        if not np.all(np.isfinite(refined)):
            return quad

        old_area = max(cv2.contourArea(quad.reshape(-1, 1, 2)), 1.0)
        new_area = cv2.contourArea(refined.reshape(-1, 1, 2))
        if new_area <= 0 or new_area / old_area < 0.82 or new_area / old_area > 1.18:
            return quad

        # 逐角点接收，避免 1-14 这种“右上角需要大幅修正、其它角不该被拉动”的情况。
        # 0=左上, 1=右上, 2=右下, 3=左下。
        moves = np.linalg.norm(refined - quad, axis=1)
        small_move_limit = max(18.0, min(w_img, h_img) * 0.016)
        large_move_limit = max(60.0, min(w_img, h_img) * 0.060)
        mixed = quad.copy()
        for corner_idx in range(4):
            move = float(moves[corner_idx])
            if move <= small_move_limit:
                mixed[corner_idx] = refined[corner_idx]
            elif corner_idx == 1 and move <= large_move_limit and supported_edges >= 3:
                # 右上角经常被颜色凸包向右拖；如果 Hough 右侧边给出强支持，允许右上角单独大幅校正。
                mixed[corner_idx] = refined[corner_idx]

        mixed_area = cv2.contourArea(mixed.reshape(-1, 1, 2))
        if mixed_area <= 0 or mixed_area / old_area < 0.85 or mixed_area / old_area > 1.15:
            return quad
        if not cv2.isContourConvex(mixed.reshape(-1, 1, 2).astype(np.float32)):
            return quad

        mixed[:, 0] = np.clip(mixed[:, 0], 0, w_img - 1)
        mixed[:, 1] = np.clip(mixed[:, 1], 0, h_img - 1)
        return mixed.astype(np.float32)

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

    def in_mat(self, xy_cm):
        if xy_cm is None:
            return False
        x, y = xy_cm
        return (0.0 <= x <= self.mat_length_cm) and (0.0 <= y <= self.mat_width_cm)

    def get_H_img2mat_px(self):
        if self.H_img2mat is None:
            return None
        scale = float(self.mat_view_scale)
        S = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        return S @ self.H_img2mat

    def render_mask(self, frame):
        """生成实心 2 色垫子识别图：垫子区域=白色(255)，背景=黑色(0)。"""
        if frame is None:
            return None
        if self._smooth_box is None:
            self.detect_mat_box(frame)
        out = np.zeros(frame.shape[:2], dtype=np.uint8)
        if self._smooth_box is not None:
            cv2.fillPoly(out, [self._smooth_box.astype(np.int32)], 255)
        return out

    def render_visible_mask(self, frame):
        """生成可见垫子颜色区域：颜色提案 + 四边形裁剪，适合检查反光/阴影下的真实可见像素。"""
        if frame is None:
            return None
        color_mask = self._compute_hsv_mask(frame)
        if self._smooth_box is None:
            self.detect_mat_box(frame)
        if self._smooth_box is not None:
            box_img = np.zeros_like(color_mask)
            cv2.fillPoly(box_img, [self._smooth_box.astype(np.int32)], 255)
            color_mask = cv2.bitwise_and(color_mask, box_img)
        return color_mask

    def render_hsv_mask(self, frame):
        """生成颜色提案二值图（无四边形裁剪），用于观察原始颜色分割效果。"""
        if frame is None:
            return None
        return self._compute_hsv_mask(frame)

    def render_detection_mask(self, frame):
        """生成用于找四边形的内部检测掩码（会做强水平闭合，不建议当最终识别图）。"""
        if frame is None:
            return None
        return self._compute_detection_mask(frame)

    def _compute_hsv_mask(self, frame, hue_low=25):
        """HSV 颜色分割 + 形态学去噪 + 强光区域补全。"""
        h_img, w_img = frame.shape[:2]

        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # 主范围：正常绿色
        lower_green = np.array([int(hue_low), 15, 15])
        upper_green = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # 辅助范围：强光反射区域（绿色被洗白 → 低饱和度、高亮度）
        lower_glare = np.array([int(hue_low), 5, 180])
        upper_glare = np.array([90, 40, 255])
        mask_glare = cv2.inRange(hsv, lower_glare, upper_glare)

        # 合并两个范围
        mask = cv2.bitwise_or(mask, mask_glare)

        # 将图像上方 50% 区域涂黑（屏蔽天空和树木噪声）
        mask[0:int(h_img * 0.5), :] = 0

        # 开运算：清除零散噪点
        open_kernel = np.ones((9, 9), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

        # 闭运算（小核）：平滑边缘
        close_kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)

        # 闭运算（大核）：填充强光造成的内部孔洞
        fill_kernel = np.ones((21, 21), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, fill_kernel, iterations=1)

        return mask

    def _compute_detection_mask(self, frame, hue_low=25):
        """检测专用掩码：在颜色提案基础上做强水平闭合，解决 1-16 这类被反光/遮挡切断的问题。"""
        h_img, w_img = frame.shape[:2]
        mask = self._compute_hsv_mask(frame, hue_low=hue_low)

        # 水平长核把同一条垫子上被人/反光/阴影断开的片段连接起来。
        close_w = self._odd(w_img * 0.050)   # 1920 图约 97 px
        close_h = self._odd(h_img * 0.016)   # 1080 图约 17 px
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, close_h))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, horizontal_kernel, iterations=1)

        # 过滤小噪点，保留长条主体。
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self._odd(w_img * 0.008), self._odd(h_img * 0.006)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
        return mask

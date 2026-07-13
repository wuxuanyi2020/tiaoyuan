"""差分法检测模块：基于边缘差分（Canny/Sobel）替代灰度差分，精确计算脚部位置。

核心改进:
  - 灰度差分 → 边缘差分：分别对基准帧和当前帧求 Sobel 梯度幅值，再相减
  - 阴影无清晰边缘 → 梯度差很弱；鞋子轮廓产生强梯度差 → 精确分离
  - Otsu 自适应阈值，不依赖固定值
  - 高斯模糊前置（相减前对两帧分别做），抹平高频噪声
  - 移除 CLAHE，黑鞋 vs 灰垫子灰度差异已足够

原始流程不变:
  1. 确定基准帧（垫子标定范围无人体关键点时的画面）
  2. 骨骼法定位起跳/落地帧后，用脚部关键点圈出 ROI
  3. 基准帧 ROI 与当前帧 ROI → 各自 Sobel 梯度幅值 → absdiff → Otsu 二值化
  4. 叠加起跳/落地差分图，计算鞋子接触位置

与骨骼关键点法解耦，不影响起跳/落地的判断逻辑。
"""
import cv2
import numpy as np


class DiffDetector:
    """差分法检测器。

    通过"无人体基准帧"与起跳/落地帧在脚部 ROI 内的边缘差分，
    精确提取鞋子在垫子上的接触轮廓，减小不同鞋子尺寸带来的误差。
    """

    # ── 起跳 → 只取脚尖关键点 ──
    _FOOT_CONFIGS_TOE = [
        {"label": "left",  "indices": (31,)},
        {"label": "right", "indices": (32,)},
    ]
    # ── 落地 → 只取脚后跟关键点 ──
    _FOOT_CONFIGS_HEEL = [
        {"label": "left",  "indices": (29,)},
        {"label": "right", "indices": (30,)},
    ]

    @staticmethod
    def _get_foot_configs(edge_side):
        if edge_side == "toe":
            return DiffDetector._FOOT_CONFIGS_TOE
        return DiffDetector._FOOT_CONFIGS_HEEL

    def __init__(self, calibrator):
        self._calib = calibrator

        # ── 基准帧 ──
        self._base_frame_raw = None
        self._base_frame_gray = None
        self._base_frame_captured = False

        # ── 全帧差分二值 Mask（供可视化） ──
        self.takeoff_diff_mask = None
        self.landing_diff_mask = None

        # ── 检测到的边缘像素点 ──
        self._takeoff_edge_px = None
        self._landing_edge_px = None

        # ── 结果(cm) ──
        self.takeoff_shoe_x_cm = None
        self.landing_shoe_x_cm = None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 基准帧
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @property
    def has_base_frame(self):
        return self._base_frame_captured

    def capture_base_frame(self, frame):
        self._base_frame_raw = frame.copy()
        self._base_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._base_frame_captured = True

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 保存起跳/落地帧
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def save_takeoff_frame(self, frame, kpts):
        self._takeoff_frame = frame.copy()
        self._takeoff_kpts = kpts.copy() if kpts is not None else None

    def save_landing_frame(self, frame, kpts):
        self._landing_frame = frame.copy()
        self._landing_kpts = kpts.copy() if kpts is not None else None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 脚部 ROI 提取
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _get_kpt(kpts, idx):
        if kpts is None or idx >= len(kpts):
            return None
        x, y = float(kpts[idx][0]), float(kpts[idx][1])
        if x == 0.0 and y == 0.0:
            return None
        return (x, y)

    def _foot_roi_gray(self, gray_img, kpts, foot_indices,
                       expand=1.0, is_left=True, shrink_inward=0.0):
        pts_px = []
        for idx in foot_indices:
            kpt = self._get_kpt(kpts, idx)
            if kpt is None:
                continue
            pts_px.append((int(kpt[0]), int(kpt[1])))
        if len(pts_px) < 1:
            return None

        h_img, w_img = gray_img.shape[:2]
        xs = [p[0] for p in pts_px]
        ys = [p[1] for p in pts_px]
        cx, cy = float(np.mean(xs)), float(np.mean(ys))

        if len(pts_px) >= 2:
            half_w = max((max(xs) - min(xs)) * expand, 40.0)
            half_h = max((max(ys) - min(ys)) * expand, 40.0)
        else:
            half_w, half_h = 60.0, 60.0

        x1 = max(0, int(cx - half_w))
        y1 = max(0, int(cy - half_h * 0.0))  # 暴力下压，避开脚踝
        x2 = min(w_img, int(cx + half_w))
        y2 = min(h_img, int(cy + half_h * 0.6))  # 上提底部，避开下方阴影

        margin = 30
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w_img, x2 + margin)
        y2 = min(h_img, y2 + margin)

        if shrink_inward > 0.0:
            box_w = x2 - x1
            shrink_px = int(box_w * shrink_inward)
            if is_left:
                x2 = max(x1 + 1, x2 - shrink_px)
            else:
                x1 = min(x2 - 1, x1 + shrink_px)

        roi = gray_img[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        return roi, (x1, y1)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 单脚边缘差分 Mask 计算
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _edge_magnitude(gray_roi):
        """计算边缘梯度幅值（强化垂直边缘，抑制水平阴影）。"""
        grad_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
        # 放大 X 梯度（脚尖/脚跟的垂直边缘），缩小 Y 梯度（底部阴影的水平边缘）
        mag = np.sqrt((grad_x * 1.5) ** 2 + (grad_y * 0.4) ** 2)
        return np.clip(mag, 0, 255).astype(np.uint8)

    def _foot_diff_mask(self, frame_gray, kpts, foot_indices,
                        is_left=True, shrink_inward=0.0):
        """MOG2 去阴影 + 轮廓实心填充：比边缘差分更干净的前景提取。

        步骤:
          1. 切出 ROI（当前帧 + 基准帧相同区域）
          2. MOG2 背景建模：base_roi 训练背景 → current_roi 提取前景
          3. 阈值过滤阴影（fg_mask 中 127=阴影，255=前景物体）
          4. 形态学去噪 + 外轮廓查找 + 实心填充

        返回:
            (binary_mask, origin_xy) 或 None
        """
        roi_result = self._foot_roi_gray(
            frame_gray, kpts, foot_indices,
            is_left=is_left, shrink_inward=shrink_inward,
        )
        if roi_result is None:
            return None

        roi_current, (x1, y1) = roi_result
        h_roi, w_roi = roi_current.shape

        base_roi = self._base_frame_gray[y1:y1 + h_roi, x1:x1 + w_roi]
        if base_roi.shape != roi_current.shape:
            return None

        # ── MOG2 背景建模 ──
        # 注意: detectShadows=True 会把黑鞋（变暗区域）标为 127 → 视为阴影，
        # 所以我们取所有非零值(>0)，阴影和前景都保留，靠轮廓填充来连成完整鞋区
        bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=2, varThreshold=20, detectShadows=True)
        bg_sub.apply(base_roi)          # 训练背景
        fg_mask = bg_sub.apply(roi_current)  # 提取前景

        # ── 保留所有非零值（含阴影 127 和前景 255） ──
        _, binary = cv2.threshold(fg_mask, 1, 255, cv2.THRESH_BINARY)

        # ── 形态学去噪 ──
        kernel = np.ones((3, 3), np.uint8)
        clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # ── 外轮廓查找 + 实心填充 ──
        solid_mask = np.zeros_like(clean)
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 150:  # 过滤小噪点
                cv2.drawContours(solid_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        return solid_mask, (x1, y1)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 左右脚差分融合 & 鞋边缘提取
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _compute_shoe_extreme(self, frame_gray, kpts, edge_side):
        """对左右脚分别做边缘差分，融合后找鞋子最前/最后边缘。

        边缘差分 Mask 本身就是鞋子轮廓的边缘响应图，
        Mask 内最极端 X 坐标即为目标边缘（脚尖/脚后跟）。

        Args:
            frame_gray: 当前帧灰度图
            kpts:       当前帧关键点
            edge_side:  "toe"（脚尖, X 最大）或 "heel"（脚后跟, X 最小）

        返回:
            x_cm: 鞋子边缘在垫子坐标系下的 X 值(cm)
        """
        foot_configs = self._get_foot_configs(edge_side)
        shrink_ratio = 0.3

        per_foot = []
        all_px_cm = []

        for cfg in foot_configs:
            is_left = (cfg["label"] == "left")
            result = self._foot_diff_mask(
                frame_gray, kpts, cfg["indices"],
                is_left=is_left, shrink_inward=shrink_ratio,
            )
            if result is None:
                continue

            binary, (x1, y1) = result
            ys, xs = np.where(binary > 0)
            if len(xs) < 15:
                continue

            # 边缘差分 Mask 已经是鞋子轮廓，直接在 Mask 内取极值
            if edge_side == "toe":
                idx = np.argmax(xs)
            else:
                idx = np.argmin(xs)

            edge_px = (float(xs[idx] + x1), float(ys[idx] + y1))
            edge_cm = self._calib.transform_to_mat_cm(edge_px)
            if edge_cm is None or edge_cm[0] < -10 or edge_cm[0] > 360:
                continue

            per_foot.append({
                "label": cfg["label"],
                "binary": binary,
                "origin": (x1, y1),
                "edge_px": edge_px,
                "edge_cm": edge_cm[0],
            })
            all_px_cm.append((edge_cm[0], edge_px))

        if not all_px_cm:
            return None

        if edge_side == "toe":
            best_cm, best_px = max(all_px_cm, key=lambda x: x[0])
        else:
            best_cm, best_px = min(all_px_cm, key=lambda x: x[0])

        # 构建全帧差分 Mask
        full_mask = np.zeros(frame_gray.shape, dtype=np.uint8)
        for f in per_foot:
            x1f, y1f = f["origin"]
            hf, wf = f["binary"].shape
            full_mask[y1f:y1f + hf, x1f:x1f + wf] = cv2.bitwise_or(
                full_mask[y1f:y1f + hf, x1f:x1f + wf], f["binary"]
            )

        if edge_side == "toe":
            self.takeoff_diff_mask = full_mask
            self._takeoff_edge_px = best_px
        else:
            self.landing_diff_mask = full_mask
            self._landing_edge_px = best_px

        return best_cm

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 公共 API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def compute_takeoff(self):
        if not self._base_frame_captured:
            return None
        try:
            frame = getattr(self, "_takeoff_frame", None)
            kpts = getattr(self, "_takeoff_kpts", None)
            if frame is None or kpts is None:
                return None
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x_cm = self._compute_shoe_extreme(gray, kpts, "toe")
            self.takeoff_shoe_x_cm = x_cm
            return x_cm
        except Exception:
            return None

    def compute_landing(self):
        if not self._base_frame_captured:
            return None
        try:
            frame = getattr(self, "_landing_frame", None)
            kpts = getattr(self, "_landing_kpts", None)
            if frame is None or kpts is None:
                return None
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x_cm = self._compute_shoe_extreme(gray, kpts, "heel")
            self.landing_shoe_x_cm = x_cm
            return x_cm
        except Exception:
            return None

    def compute_combined_distance(self):
        to_x = self.compute_takeoff()
        ld_x = self.compute_landing()
        if to_x is None or ld_x is None:
            return None, None, None
        dist = max(0.0, ld_x - to_x)
        return to_x, ld_x, dist

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 差分照片渲染
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _get_frame_kpts(self, which):
        if which == "takeoff":
            return (getattr(self, "_takeoff_frame", None),
                    getattr(self, "_takeoff_kpts", None), "Takeoff")
        return (getattr(self, "_landing_frame", None),
                getattr(self, "_landing_kpts", None), "Landing")

    def _render_roi_view(self, which):
        frame, kpts, label = self._get_frame_kpts(which)
        if frame is None or kpts is None:
            return None

        edge_side = "toe" if which == "takeoff" else "heel"
        foot_configs = self._get_foot_configs(edge_side)
        vis = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        colors = [("left", (0, 255, 0)), ("right", (0, 255, 255))]
        for cfg, (_, color) in zip(foot_configs, colors):
            is_left = (cfg["label"] == "left")
            result = self._foot_roi_gray(gray, kpts, cfg["indices"],
                                         is_left=is_left, shrink_inward=0.3)
            if result is None:
                continue
            _, (x1, y1) = result
            roi = result[0]
            h, w = roi.shape
            x2, y2 = x1 + w, y1 + h
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
            cv2.putText(vis, f"{cfg['label']} ROI", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(vis, f"Stage2 - ROI Crop ({label})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return vis

    def _render_edge_diff_view(self, which):
        """Stage3-MOG2: RawBase | RawCurrent | MOG2_FG | SolidMask 四列。"""
        frame, kpts, label = self._get_frame_kpts(which)
        if frame is None or kpts is None:
            return None

        edge_side = "toe" if which == "takeoff" else "heel"
        foot_configs = self._get_foot_configs(edge_side)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        strips = []
        for cfg in foot_configs:
            is_left = (cfg["label"] == "left")
            roi_result = self._foot_roi_gray(gray, kpts, cfg["indices"],
                                             is_left=is_left, shrink_inward=0.3)
            if roi_result is None:
                continue
            roi_current, (x1, y1) = roi_result
            h, w = roi_current.shape
            base_roi = self._base_frame_gray[y1:y1 + h, x1:x1 + w]
            if base_roi.shape != roi_current.shape:
                continue

            # MOG2 提取（用于可视化中间结果）
            bg_sub = cv2.createBackgroundSubtractorMOG2(
                history=2, varThreshold=20, detectShadows=True)
            bg_sub.apply(base_roi)
            fg_mask = bg_sub.apply(roi_current)
            # 增强对比度便于观察
            fg_vis = cv2.normalize(fg_mask, None, 0, 255, cv2.NORM_MINMAX)

            # 最终的实心填充 Mask
            mask_result = self._foot_diff_mask(
                gray, kpts, cfg["indices"],
                is_left=is_left, shrink_inward=0.3,
            )
            binary = mask_result[0] if mask_result is not None else np.zeros_like(roi_current)

            target_h = 160

            def _r(img):
                s = target_h / img.shape[0]
                return cv2.resize(img, (max(1, int(img.shape[1] * s)), target_h),
                                  interpolation=cv2.INTER_AREA)

            col_raw_base = cv2.cvtColor(_r(base_roi), cv2.COLOR_GRAY2BGR)
            col_raw_curr = cv2.cvtColor(_r(roi_current), cv2.COLOR_GRAY2BGR)
            col_fg = cv2.cvtColor(_r(fg_vis), cv2.COLOR_GRAY2BGR)
            col_binary = cv2.cvtColor(_r(binary), cv2.COLOR_GRAY2BGR)

            tag_h = 28
            for cimg, t in [(col_raw_base, "RawBase"), (col_raw_curr, "RawCurrent"),
                            (col_fg, "MOG2_FG"), (col_binary, "SolidMask")]:
                tb = np.zeros((tag_h, cimg.shape[1], 3), dtype=np.uint8)
                cv2.putText(tb, t, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cimg[:] = np.vstack([tb, cimg])[:cimg.shape[0]]

            strip = cv2.hconcat([col_raw_base, col_raw_curr, col_fg, col_binary])
            cv2.putText(strip, cfg["label"], (8, tag_h + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if is_left else (0, 255, 255), 2)
            strips.append(strip)

        if not strips:
            return None

        collage = cv2.vconcat(strips)
        title_h = 50
        canvas = np.zeros((title_h + collage.shape[0], collage.shape[1], 3), dtype=np.uint8)
        canvas[title_h:, :collage.shape[1]] = collage
        cv2.putText(canvas,
                    f"Stage3 - MOG2 ({label}): 原始图(左 vs 右) → MOG2 前景 → 实心填充Mask",
                    (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return canvas

    def render_result_image(self, mode="combined"):
        if self._base_frame_raw is None:
            return None

        # ── 阶段①：基准帧 ──
        if mode == "base":
            vis = self._base_frame_raw.copy()
            cv2.putText(vis, "Stage1 - Base Frame (垫子标定范围无人)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            return vis

        # ── 阶段②：ROI 切割 ──
        if mode in ("roi_takeoff", "roi_landing"):
            which = mode.split("_")[1]
            return self._render_roi_view(which)

        # ── 阶段③：边缘差分（统一四列视图） ──
        if mode in ("diffmap_takeoff", "diffmap_landing",
                     "mask_takeoff", "mask_landing"):
            which = mode.split("_")[1]
            return self._render_edge_diff_view(which)

        # ── 阶段④：叠加结果 ──
        vis = self._base_frame_raw.copy()
        overlay = np.zeros_like(vis, dtype=np.uint8)

        if mode in ("takeoff", "combined") and self.takeoff_diff_mask is not None:
            overlay[self.takeoff_diff_mask > 0] = (0, 0, 255)

        if mode in ("landing", "combined") and self.landing_diff_mask is not None:
            red_area = (overlay[:, :, 2] > 0)
            overlay[self.landing_diff_mask > 0] = (255, 0, 0)
            if mode == "combined":
                overlay[red_area & (self.landing_diff_mask > 0)] = (255, 0, 255)

        cv2.addWeighted(overlay, 0.4, vis, 1.0, 0, vis)

        if mode in ("takeoff", "combined") and self._takeoff_edge_px is not None:
            pt = (int(self._takeoff_edge_px[0]), int(self._takeoff_edge_px[1]))
            cv2.circle(vis, pt, 6, (0, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(vis, pt, 10, (0, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.putText(vis, "Toe(EdgeDiff)", (pt[0] + 12, pt[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if mode in ("landing", "combined") and self._landing_edge_px is not None:
            pt = (int(self._landing_edge_px[0]), int(self._landing_edge_px[1]))
            cv2.circle(vis, pt, 6, (0, 255, 0), -1, lineType=cv2.LINE_AA)
            cv2.circle(vis, pt, 10, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            cv2.putText(vis, "Heel(EdgeDiff)", (pt[0] + 12, pt[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        y_offset = 40
        title_map = {
            "takeoff":  "Stage4 - 起跳边缘差分(红)",
            "landing":  "Stage4 - 落地边缘差分(蓝)",
            "combined": "Stage4 - 边缘差分叠加(红+蓝=紫)",
        }
        cv2.putText(vis, title_map.get(mode, ""), (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        y_offset = 80
        if mode in ("takeoff", "combined") and self.takeoff_shoe_x_cm is not None:
            cv2.putText(vis, f"Diff Takeoff: {self.takeoff_shoe_x_cm:.1f} cm",
                        (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset += 30
        if mode in ("landing", "combined") and self.landing_shoe_x_cm is not None:
            cv2.putText(vis, f"Diff Landing: {self.landing_shoe_x_cm:.1f} cm",
                        (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
        if mode == "combined" and self.takeoff_shoe_x_cm is not None and self.landing_shoe_x_cm is not None:
            dist = max(0.0, self.landing_shoe_x_cm - self.takeoff_shoe_x_cm)
            cv2.putText(vis, f"Diff Distance: {dist:.1f} cm",
                        (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return vis

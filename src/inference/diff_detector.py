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
from PIL import Image, ImageDraw, ImageFont


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

    @staticmethod
    def _put_text_cn(img, text, pos, color, size=28):
        """使用 PIL 在 OpenCV 图像上渲染中文文本，避免 cv2.putText 中文乱码。"""
        if img is None:
            return img
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = None
        for path in ["msyh.ttc", "simhei.ttf", "simsun.ttc",
                      "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
                      "C:/Windows/Fonts/simsun.ttc"]:
            try:
                font = ImageFont.truetype(path, size)
                break
            except Exception:
                continue
        if font is None:
            try:
                font = ImageFont.load_default()
            except Exception:
                return img
        draw.text(pos, text, font=font, fill=color[::-1])
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def __init__(self, calibrator, enable_seg=False):
        self._calib = calibrator
        self.enable_seg = enable_seg

        # ── 基准帧 ──
        self._base_frame_raw = None
        self._base_frame_gray = None
        self._base_frame_captured = False

        # ── YOLO 实例分割模型 ──
        self.seg_model = None
        if self.enable_seg:
            from ultralytics import YOLO
            self.seg_model = YOLO("yolo11x-seg.pt")
            print("[DIFF] YOLOv11-seg 实例分割已启用 (模型: yolo11x-seg.pt)")

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

    def _foot_diff_mask(self, frame_bgr, kpts, foot_indices,
                        is_left=True, shrink_inward=0.0):
        """提取脚部 ROI 内的鞋子二值 Mask。

        根据 self.enable_seg 选择两种实现之一：
          - True:  使用 YOLOv11-seg 在全图上做实例分割，再按关键点 ROI 裁剪
          - False: 使用 MOG2 背景差分 + 轮廓实心填充（原有逻辑）

        返回:
            (binary_mask, origin_xy) 或 None
        """
        # ── YOLOv11-seg 路径：先在全图上推理，再按 ROI 裁剪 ──
        if self.enable_seg:
            img_h, img_w = frame_bgr.shape[:2]
            # 全图推理：只筛选 person 类
            results = self.seg_model(frame_bgr, classes=[0], conf=0.3, verbose=False)

            # 在全图尺寸上构建二值人体 Mask
            full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            if len(results) > 0 and results[0].masks is not None:
                for mask_xy in results[0].masks.xy:
                    pts = np.array([mask_xy], dtype=np.int32)
                    cv2.fillPoly(full_mask, [pts], 255)

            # 通过关键点获取 ROI 坐标（在灰度图上计算，不依赖分割结果）
            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            roi_result = self._foot_roi_gray(
                frame_gray, kpts, foot_indices,
                is_left=is_left, shrink_inward=shrink_inward,
            )
            if roi_result is None:
                return None

            _, (x1, y1) = roi_result
            h_roi, w_roi = roi_result[0].shape

            # 从全图 Mask 中裁剪出 ROI 区域
            solid_mask = full_mask[y1:y1 + h_roi, x1:x1 + w_roi].copy()

            # 工程 Trick：上半部分（前 40% 高度）置零，只保留贴近地面的鞋子部分
            solid_mask[0:int(h_roi * 0.4), :] = 0

            return solid_mask, (x1, y1)

        # ── MOG2 路径：先获取 ROI，再在 ROI 内做差分 ──
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        roi_result = self._foot_roi_gray(
            frame_gray, kpts, foot_indices,
            is_left=is_left, shrink_inward=shrink_inward,
        )
        if roi_result is None:
            return None

        roi_gray, (x1, y1) = roi_result
        h_roi, w_roi = roi_gray.shape

        base_roi = self._base_frame_gray[y1:y1 + h_roi, x1:x1 + w_roi]
        if base_roi.shape != roi_gray.shape:
            return None

        bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=2, varThreshold=20, detectShadows=True)
        bg_sub.apply(base_roi)          # 训练背景
        fg_mask = bg_sub.apply(roi_gray)  # 提取前景

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

    def _compute_shoe_extreme(self, frame_bgr, kpts, edge_side):
        """对左右脚分别做边缘差分，融合后找鞋子最前/最后边缘。

        边缘差分 Mask 本身就是鞋子轮廓的边缘响应图，
        Mask 内最极端 X 坐标即为目标边缘（脚尖/脚后跟）。

        Args:
            frame_bgr: 当前帧彩色 BGR 图像
            kpts:      当前帧关键点
            edge_side: "toe"（脚尖, X 最大）或 "heel"（脚后跟, X 最小）

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
                frame_bgr, kpts, cfg["indices"],
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
        full_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
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
        if not self.enable_seg and not self._base_frame_captured:
            return None
        try:
            frame = getattr(self, "_takeoff_frame", None)
            kpts = getattr(self, "_takeoff_kpts", None)
            if frame is None or kpts is None:
                return None
            x_cm = self._compute_shoe_extreme(frame, kpts, "toe")
            self.takeoff_shoe_x_cm = x_cm
            return x_cm
        except Exception:
            return None

    def compute_landing(self):
        if not self.enable_seg and not self._base_frame_captured:
            return None
        try:
            frame = getattr(self, "_landing_frame", None)
            kpts = getattr(self, "_landing_kpts", None)
            if frame is None or kpts is None:
                return None
            x_cm = self._compute_shoe_extreme(frame, kpts, "heel")
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
        """Stage3 可视化：根据 self.enable_seg 显示不同内容。

        YOLO 模式: RawCurrent | YOLO_ROI_Mask | FinalMask（三列）
        MOG2 模式: RawBase | RawCurrent | MOG2_FG | SolidMask（四列，原有逻辑）
        """
        frame, kpts, label = self._get_frame_kpts(which)
        if frame is None or kpts is None:
            return None

        edge_side = "toe" if which == "takeoff" else "heel"
        foot_configs = self._get_foot_configs(edge_side)
        h_img, w_img = frame.shape[:2]

        # ── YOLO 模式：无 MOG2 ──
        if self.enable_seg:
            # 全图 YOLO 推理一次
            results = self.seg_model(frame, classes=[0], conf=0.3, verbose=False)
            full_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            if len(results) > 0 and results[0].masks is not None:
                for mask_xy in results[0].masks.xy:
                    pts = np.array([mask_xy], dtype=np.int32)
                    cv2.fillPoly(full_mask, [pts], 255)

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            strips = []
            for cfg in foot_configs:
                is_left = (cfg["label"] == "left")
                roi_result = self._foot_roi_gray(
                    frame_gray, kpts, cfg["indices"],
                    is_left=is_left, shrink_inward=0.3)
                if roi_result is None:
                    continue
                _, (x1, y1) = roi_result
                h_roi, w_roi = roi_result[0].shape

                # 列1: 原始BGR ROI
                col_raw = frame[y1:y1 + h_roi, x1:x1 + w_roi]
                # 列2: 未经高度过滤的 YOLO Mask 交集
                raw_mask = full_mask[y1:y1 + h_roi, x1:x1 + w_roi].copy()
                # 列3: 经过 40% 高度过滤的最终 Mask
                final_mask = raw_mask.copy()
                final_mask[0:int(h_roi * 0.4), :] = 0

                target_h = 160
                def _r(img):
                    s = target_h / img.shape[0]
                    nw = max(1, int(img.shape[1] * s))
                    if len(img.shape) == 2:
                        return cv2.resize(img, (nw, target_h), interpolation=cv2.INTER_AREA)
                    return cv2.resize(img, (nw, target_h), interpolation=cv2.INTER_AREA)

                col_raw_r = _r(col_raw)
                col_mask_r = cv2.cvtColor(_r(raw_mask), cv2.COLOR_GRAY2BGR)
                col_final_r = cv2.cvtColor(_r(final_mask), cv2.COLOR_GRAY2BGR)

                tag_h = 28
                for cimg, t in [(col_raw_r, "RawCurrent"),
                                (col_mask_r, "YOLO_ROI_Mask"),
                                (col_final_r, "FinalMask")]:
                    tb = np.zeros((tag_h, cimg.shape[1], 3), dtype=np.uint8)
                    cv2.putText(tb, t, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cimg[:] = np.vstack([tb, cimg])[:cimg.shape[0]]

                strip = cv2.hconcat([col_raw_r, col_mask_r, col_final_r])
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
            canvas = self._put_text_cn(
                canvas,
                f"Stage3 - YOLO ({label}): 原图ROI → YOLO人体Mask交集 → 高度过滤后Mask",
                (20, 6), (255, 255, 255), size=22)
            return canvas

        # ── MOG2 模式：原有逻辑 ──
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

            # MOG2 提取
            bg_sub = cv2.createBackgroundSubtractorMOG2(
                history=2, varThreshold=20, detectShadows=True)
            bg_sub.apply(base_roi)
            fg_mask = bg_sub.apply(roi_current)
            fg_vis = cv2.normalize(fg_mask, None, 0, 255, cv2.NORM_MINMAX)

            mask_result = self._foot_diff_mask(
                frame, kpts, cfg["indices"],
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
        canvas = self._put_text_cn(
            canvas,
            f"Stage3 - MOG2 ({label}): 原始图(左 vs 右) → MOG2 前景 → 实心填充Mask",
            (20, 6), (255, 255, 255), size=22)
        return canvas

    def _render_seg_overlay(self, which):
        """绘制 YOLO 人体实例分割结果覆盖图（Stage1）。"""
        if which == "takeoff":
            frame = getattr(self, "_takeoff_frame", None)
        else:
            frame = getattr(self, "_landing_frame", None)
        if frame is None:
            return None
        h, w = frame.shape[:2]

        results = self.seg_model(frame, classes=[0], conf=0.3, verbose=False)
        full_mask = np.zeros((h, w), dtype=np.uint8)
        if len(results) > 0 and results[0].masks is not None:
            for mask_xy in results[0].masks.xy:
                pts = np.array([mask_xy], dtype=np.int32)
                cv2.fillPoly(full_mask, [pts], 255)

        # 原图 + 半透明 Mask 叠加
        overlay = frame.copy()
        overlay[full_mask > 0] = overlay[full_mask > 0] * 0.5 + np.array([0, 200, 0], dtype=np.uint8) * 0.5
        overlay = overlay.astype(np.uint8)

        label = "起跳帧" if which == "takeoff" else "落地帧"
        overlay = self._put_text_cn(overlay, f"Stage1 - YOLO 人体实例分割 ({label})",
                                    (20, 8), (255, 255, 255), size=26)
        return overlay

    def render_result_image(self, mode="combined"):
        """渲染各阶段结果图。MOG2 模式依赖基准帧，YOLO 模式使用起跳/落地帧。"""

        # ── 阶段①：YOLO 人体分割或 MOG2 基准帧 ──
        if mode in ("base_takeoff", "base_landing"):
            if self.enable_seg:
                which = mode.split("_")[1]
                return self._render_seg_overlay(which)
            return None
        if mode == "base":
            if self._base_frame_raw is None:
                return None
            vis = self._base_frame_raw.copy()
            vis = self._put_text_cn(vis, "Stage1 - Base Frame (垫子标定范围无人)",
                                    (20, 8), (255, 255, 255), size=26)
            return vis

        # ── 阶段②：ROI 切割 ──
        if mode in ("roi_takeoff", "roi_landing"):
            which = mode.split("_")[1]
            return self._render_roi_view(which)

        # ── 阶段③：边缘差分 / YOLO Mask 切片 ──
        if mode in ("diffmap_takeoff", "diffmap_landing",
                     "mask_takeoff", "mask_landing"):
            which = mode.split("_")[1]
            return self._render_edge_diff_view(which)

        # ── 阶段④：叠加结果 ──
        if self.enable_seg:
            frame = None
            if mode in ("takeoff", "combined"):
                frame = getattr(self, "_takeoff_frame", None)
            elif mode == "landing":
                frame = getattr(self, "_landing_frame", None)
            # combined 模式优先使用起跳帧
            if frame is None:
                frame = getattr(self, "_takeoff_frame",
                                getattr(self, "_landing_frame", None))
            if frame is None:
                return None
            vis = frame.copy()
        else:
            if self._base_frame_raw is None:
                return None
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

        y_offset = 8
        prefix = "Stage4 - YOLO" if self.enable_seg else "Stage4 - MOG2"
        title_map = {
            "takeoff":  f"{prefix}起跳(红)",
            "landing":  f"{prefix}落地(蓝)",
            "combined": f"{prefix}叠加(红+蓝=紫)",
        }
        vis = self._put_text_cn(vis, title_map.get(mode, ""), (20, y_offset),
                                (255, 255, 255), size=26)

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

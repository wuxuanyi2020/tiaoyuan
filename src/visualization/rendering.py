"""可视化渲染模块：绘制骨架、脚点、测量线、垫子俯视图等。"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def imwrite_safe(path, img_bgr):
    """安全保存图片（替代 cv2.imwrite），支持包含中文的路径。"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(path, "JPEG")
        return True
    except Exception as e:
        print(f"[imwrite_safe] 保存失败: {path}, 错误: {e}")
        return False


class Renderer:
    def __init__(self, kpt_idx):
        self.kpt_idx = kpt_idx

    @staticmethod
    def put_text_chinese(img, text, pos, color, size=30):
        if not isinstance(img, np.ndarray):
            return img
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = None
        font_paths = [
            "msyh.ttc", "simhei.ttf", "simsun.ttc",
            "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simsun.ttc",
        ]
        for path in font_paths:
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

    @staticmethod
    def draw_pose(img, kpts, connections, color=(0, 255, 0)):
        if img is None or kpts is None:
            return
        if connections is not None:
            for start_idx, end_idx in connections:
                if start_idx >= len(kpts) or end_idx >= len(kpts):
                    continue
                p1, p2 = kpts[start_idx], kpts[end_idx]
                if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
                    cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 2, lineType=cv2.LINE_AA)
        for point in kpts:
            if point[0] > 0 and point[1] > 0:
                cv2.circle(img, (int(point[0]), int(point[1])), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    def draw_feet(self, img, feet_getter, kpts):
        if img is None or kpts is None:
            return
        toes = feet_getter(kpts, "toe")
        heels = feet_getter(kpts, "heel")
        for point in [toes.get("l"), toes.get("r")]:
            if point is not None:
                cv2.circle(img, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        for point in [heels.get("l"), heels.get("r")]:
            if point is not None:
                cv2.circle(img, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1, lineType=cv2.LINE_AA)

    @staticmethod
    def draw_x_line(img, H_mat2img, mat_width_cm, x_cm, color, thickness=2, label=None):
        if img is None or H_mat2img is None or x_cm is None:
            return
        ys = np.linspace(0.0, float(mat_width_cm), 25, dtype=np.float32)
        pts_mat = np.stack([np.full_like(ys, float(x_cm), dtype=np.float32), ys], axis=1).reshape(1, -1, 2)
        pts_img = cv2.perspectiveTransform(pts_mat, H_mat2img).reshape(-1, 2)
        poly = pts_img.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [poly], False, color, thickness, lineType=cv2.LINE_AA)
        if label:
            pt = pts_img[0].astype(int)
            cv2.putText(img, label, (pt[0], pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    @staticmethod
    def draw_x_point(img, H_mat2img, mat_width_cm, x_cm, color, radius=6, label=None, label_offset_y=-8):
        """在垫子转换到图像的指定X位置画一个圆点。"""
        if img is None or H_mat2img is None or x_cm is None:
            return
        mid_y = mat_width_cm / 2.0
        pt_mat = np.array([[[x_cm, mid_y]]], dtype=np.float32)
        pt_img = cv2.perspectiveTransform(pt_mat, H_mat2img).reshape(-1, 2)[0]
        center = (int(pt_img[0]), int(pt_img[1]))
        cv2.circle(img, center, radius, color, -1, lineType=cv2.LINE_AA)
        if label:
            cv2.putText(img, label, (center[0] + 8, center[1] + label_offset_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    @staticmethod
    def draw_measurement_line(img, H_mat2img, mat_width_cm, takeoff_x_cm, landing_x_cm):
        if img is None or H_mat2img is None or takeoff_x_cm is None or landing_x_cm is None:
            return
        mid_y = mat_width_cm / 2.0
        pt1_mat = np.array([[[takeoff_x_cm, mid_y]]], dtype=np.float32)
        pt2_mat = np.array([[[landing_x_cm, mid_y]]], dtype=np.float32)
        pt1_img = cv2.perspectiveTransform(pt1_mat, H_mat2img).reshape(-1, 2)[0]
        pt2_img = cv2.perspectiveTransform(pt2_mat, H_mat2img).reshape(-1, 2)[0]
        p1, p2 = (int(pt1_img[0]), int(pt1_img[1])), (int(pt2_img[0]), int(pt2_img[1]))
        cv2.line(img, p1, p2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(img, p1, 4, (0, 255, 255), -1)
        cv2.circle(img, p2, 4, (0, 0, 255), -1)

    @staticmethod
    def draw_mat_outline(img, calibrator, color=(0, 200, 255)):
        if calibrator.manual_mode and not calibrator.mat_locked:
            for idx, pt in enumerate(calibrator.manual_points):
                cv2.circle(img, (int(pt[0]), int(pt[1])), 6, (0, 0, 255), -1)
                cv2.putText(img, str(idx + 1), (int(pt[0]) + 10, int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if len(calibrator.manual_points) < 4:
                return
        if img is None or calibrator._smooth_box is None:
            return
        poly = calibrator._smooth_box.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [poly], True, color, 1, lineType=cv2.LINE_AA)
        for pt in calibrator._smooth_box:
            cv2.circle(img, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    @staticmethod
    def render_mat_view(frame, calibrator, feet_getter, kpts, takeoff_line_cm, takeoff_x_cm, landing_x_cm):
        if frame is None:
            return None
        H_px = calibrator.get_H_img2mat_px()
        if H_px is None:
            return None
        width = int(round(calibrator.mat_length_cm * calibrator.mat_view_scale))
        height = int(round(calibrator.mat_width_cm * calibrator.mat_view_scale))
        if width <= 0 or height <= 0:
            return None

        warped = cv2.warpPerspective(frame, H_px, (width, height))
        x_line = int(round(takeoff_line_cm * calibrator.mat_view_scale))
        cv2.line(warped, (x_line, 0), (x_line, height - 1), (255, 255, 255), 2, cv2.LINE_AA)

        if takeoff_x_cm is not None:
            x = int(round(takeoff_x_cm * calibrator.mat_view_scale))
            cv2.line(warped, (x, 0), (x, height - 1), (0, 0, 255), 2, cv2.LINE_AA)
        if landing_x_cm is not None:
            x = int(round(landing_x_cm * calibrator.mat_view_scale))
            cv2.line(warped, (x, 0), (x, height - 1), (0, 0, 255), 2, cv2.LINE_AA)
        if takeoff_x_cm is not None and landing_x_cm is not None:
            x1 = int(round(takeoff_x_cm * calibrator.mat_view_scale))
            x2 = int(round(landing_x_cm * calibrator.mat_view_scale))
            y_mid = height // 2
            cv2.line(warped, (x1, y_mid), (x2, y_mid), (0, 255, 0), 2)

        if kpts is not None:
            toes = feet_getter(kpts, "toe")
            heels = feet_getter(kpts, "heel")
            for point in [toes.get("l"), toes.get("r")]:
                cm = calibrator.transform_to_mat_cm(point) if point is not None else None
                if cm is not None and calibrator.in_mat(cm):
                    cv2.circle(warped, (int(round(cm[0] * calibrator.mat_view_scale)), int(round(cm[1] * calibrator.mat_view_scale))), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            for point in [heels.get("l"), heels.get("r")]:
                cm = calibrator.transform_to_mat_cm(point) if point is not None else None
                if cm is not None and calibrator.in_mat(cm):
                    cv2.circle(warped, (int(round(cm[0] * calibrator.mat_view_scale)), int(round(cm[1] * calibrator.mat_view_scale))), 4, (255, 0, 0), -1, lineType=cv2.LINE_AA)
        return warped

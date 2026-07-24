"""左上角模拟时间 OCR（固定 OpenCV 绿色时间码）。

用于 simulated_stream.mp4 这类左上角叠加 HH:MM:SS 的模拟流：
- 先按绿色/高饱和文字做二值化；
- 再用 cv2.putText 渲染同字体模板，在可能秒数范围内做形状匹配；
- 最后把 OCR 到的秒级时间和当前解码帧号合成 HH:MM:SS:FF，达到帧精度。
"""
from __future__ import annotations

import cv2
import numpy as np


class TimeOCR:
    def __init__(self, roi=(15, 18, 155, 30), fps=30.0, max_seconds=120, enabled=True):
        self.roi = tuple(int(v) for v in roi)
        self.fps = float(fps) if fps and fps > 1e-3 else 30.0
        self.max_seconds = int(max(0, max_seconds or 0))
        self.enabled = bool(enabled)
        self._templates = {}
        self._last = None
        # 记录 OCR 秒数字样真正切换的解码帧。simulated_stream 的叠字不一定
        # 严格在 idx % fps == 0 处换秒，因此帧内编号以 OCR 秒跳变点为准。
        self._second_start_frame = {}
        self._last_second = None

    @staticmethod
    def format_seconds(seconds: int) -> str:
        seconds = int(max(0, seconds))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def format_frame_time(self, frame_idx0: int, seconds: int | None = None, ok: bool = False, confidence: float = 0.0, use_ocr_anchor: bool = True):
        frame_idx0 = int(max(0, frame_idx0))
        fps_i = max(1, int(round(self.fps)))
        if seconds is None:
            seconds = int(frame_idx0 // fps_i)
        seconds = int(max(0, seconds))
        if use_ocr_anchor and seconds in self._second_start_frame:
            start_frame = self._second_start_frame[seconds]
            frame_in_second = int(frame_idx0 - start_frame)
            frame_in_second = max(0, min(fps_i - 1, frame_in_second))
        elif use_ocr_anchor and self._second_start_frame:
            # 用最近的 OCR 秒锚点外推，避免 idx % fps 与叠字换秒点相差 1 帧。
            anchor_sec = max(self._second_start_frame, key=lambda sec: self._second_start_frame[sec])
            start_frame = self._second_start_frame[anchor_sec] + (seconds - anchor_sec) * fps_i
            frame_in_second = int(frame_idx0 - start_frame)
            frame_in_second = max(0, min(fps_i - 1, frame_in_second))
        else:
            frame_in_second = int(frame_idx0 % fps_i)
        text = self.format_seconds(seconds)
        return {
            "ok": bool(ok),
            "text": text,
            "second": int(seconds),
            "frame_in_second": frame_in_second,
            "timecode": f"{text}:{frame_in_second:02d}",
            "frame_idx0": frame_idx0,
            "confidence": round(float(confidence), 4),
        }

    def _text_template(self, text: str, shape):
        key = (text, int(shape[0]), int(shape[1]))
        cached = self._templates.get(key)
        if cached is not None:
            return cached
        font = cv2.FONT_HERSHEY_SIMPLEX
        best_img = None
        best_score = -1.0
        # simulated_stream 的时间码与 OpenCV putText 非常接近；多试几个厚度/缩放适配压缩损失。
        for scale in (0.8, 0.9, 1.0, 1.1, 1.2):
            for thick in (2, 3):
                (w, h), base = cv2.getTextSize(text, font, scale, thick)
                tmp = np.zeros((h + base + 10, w + 10), dtype=np.uint8)
                cv2.putText(tmp, text, (5, h + 5), font, scale, 255, thick, cv2.LINE_AA)
                pts = cv2.findNonZero(tmp)
                if pts is None:
                    continue
                x, y, ww, hh = cv2.boundingRect(pts)
                tmp = tmp[y:y + hh, x:x + ww]
                resized = cv2.resize(tmp, (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_NEAREST)
                fill_ratio = float((resized > 0).mean())
                # 只用于模板内部排序：填充率过高/过低都不稳定。
                sc = -abs(fill_ratio - 0.38)
                if sc > best_score:
                    best_score = sc
                    best_img = resized
        if best_img is None:
            best_img = np.zeros(shape, dtype=np.uint8)
        self._templates[key] = best_img
        return best_img

    @staticmethod
    def _iou(mask, templ):
        a = mask > 0
        b = templ > 0
        union = np.logical_or(a, b).sum()
        if union <= 0:
            return 0.0
        return float(np.logical_and(a, b).sum()) / float(union)

    def _extract_mask(self, frame):
        if frame is None:
            return None, None
        x, y, w, h = self.roi
        H, W = frame.shape[:2]
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))
        roi = frame[y:y + h, x:x + w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # 绿色时间字；兼容压缩后偏浅/偏白的抗锯齿边缘。
        mask_green = ((hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 90) &
                      (hsv[:, :, 1] > 45) & (hsv[:, :, 2] > 100))
        mask = mask_green.astype(np.uint8) * 255
        # 去掉小噪点，不做横向闭合，避免数字粘成块后形状失真。
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
        pts = cv2.findNonZero(mask)
        if pts is None:
            return None, None
        bx, by, bw, bh = cv2.boundingRect(pts)
        if bw < 40 or bh < 10:
            return None, None
        return mask[by:by + bh, bx:bx + bw], (x + bx, y + by, bw, bh)

    def recognize(self, frame, frame_idx0: int, fps: float | None = None):
        if fps and fps > 1e-3 and abs(float(fps) - self.fps) > 1e-3:
            self.fps = float(fps)
        fallback = self.format_frame_time(frame_idx0, seconds=None, ok=False, confidence=0.0)
        if not self.enabled:
            self._last = fallback
            return fallback
        mask, box = self._extract_mask(frame)
        if mask is None:
            self._last = fallback
            return fallback

        fps_i = max(1, int(round(self.fps)))
        expected_sec = int(max(0, frame_idx0) // fps_i)
        predicted_sec = expected_sec
        if self._second_start_frame:
            # 按最近一次可靠 OCR 秒跳变点预测当前秒，解决 simulated_stream 秒切换点
            # 比 CAP_PROP_POS_FRAMES//fps 晚/早 1 帧的问题，也抑制相似数字误识别。
            anchor_sec = max(self._second_start_frame, key=lambda sec: self._second_start_frame[sec])
            anchor_frame = self._second_start_frame[anchor_sec]
            if frame_idx0 >= anchor_frame:
                predicted_sec = int(anchor_sec + (frame_idx0 - anchor_frame) // fps_i)
        predicted_sec = max(0, min(self.max_seconds, predicted_sec))
        if self._second_start_frame:
            anchor_sec_for_start = max(self._second_start_frame, key=lambda sec: self._second_start_frame[sec])
            predicted_start_frame = self._second_start_frame[anchor_sec_for_start] + (predicted_sec - anchor_sec_for_start) * fps_i
        else:
            predicted_start_frame = predicted_sec * fps_i
        predicted_frame_in_second = int(frame_idx0 - predicted_start_frame)
        search_radius = 1 if self._second_start_frame else 2
        lo = max(0, predicted_sec - search_radius)
        hi = min(self.max_seconds, predicted_sec + search_radius)
        seconds_iter = range(lo, hi + 1)

        best = (-1.0, predicted_sec)
        scores = {}
        for sec in seconds_iter:
            text = self.format_seconds(sec)
            templ = self._text_template(text, mask.shape)
            sc = self._iou(mask, templ)
            scores[int(sec)] = sc
            if sc > best[0]:
                best = (sc, sec)
        best_conf, best_sec = best
        pred_conf = scores.get(int(predicted_sec), -1.0)
        # 时间码是连续叠字，单帧模板相似时优先相信“上一秒锚点 + 帧计数”的预测，
        # 避免 35/36、43/44 这类相似数字来回跳。只有分数明显更好时才偏离预测。
        if int(best_sec) != int(predicted_sec):
            # 秒边界处，真实叠字可能比帧号预测晚 1 帧；只在预测新秒刚开始的
            # 头 1 帧允许“上一秒”覆盖预测，避免中间帧被相似数字拉回上一秒。
            if int(best_sec) < int(predicted_sec):
                margin = 0.02 if (int(predicted_sec) not in self._second_start_frame and predicted_frame_in_second <= 1) else 999.0
            else:
                margin = 0.12
            if best_conf >= pred_conf + margin:
                conf, sec = best_conf, int(best_sec)
            else:
                conf, sec = pred_conf, int(predicted_sec)
        else:
            conf, sec = pred_conf, int(predicted_sec)
        ok = conf >= 0.35
        if ok:
            sec = int(sec)
            if self._last_second != sec and sec not in self._second_start_frame:
                self._second_start_frame[sec] = int(max(0, frame_idx0))
            elif sec not in self._second_start_frame:
                # 如果不是从视频开头开始识别，也至少给当前秒一个锚点。
                self._second_start_frame[sec] = int(max(0, frame_idx0))
            self._last_second = sec
        info = self.format_frame_time(frame_idx0, seconds=sec if ok else predicted_sec, ok=ok, confidence=conf)
        info["bbox"] = box
        self._last = info
        return info

"""姿态估计模块：MediaPipe 模型初始化、视频/图片读取、关键点推理。"""
import os
import urllib.request

import cv2
import numpy as np


class PoseEstimator:
    def __init__(self, video_source, backend="mediapipe"):
        self.backend = backend or "mediapipe"
        self.model = None
        self.mp_pose = None
        self.mp_landmarker = None
        self.mp_connections = None
        self._mp_ts_ms = 0
        self._mp_step_ms = 33
        self.image_mode = False
        self.static_frame = None
        self.cap = None
        self.person_count = 0
        self.all_kpts_list = []
        # 保守提高关键点置信度阈值：之前 foot=0.30 容易在强光/逆光
        # 或右向左视频中接受单帧脚尖漂移；这里回收一点阈值，宁可
        # 短暂丢几帧脚点，也不要把明显乱飞的脚尖当作起跳触发。
        self._foot_landmark_ids = {27, 28, 29, 30, 31, 32}
        self._body_visibility_threshold = 0.50
        self._foot_visibility_threshold = 0.30
        self._last_pose_center = None


        try:
            import mediapipe as mp
        except ImportError as exc:
            raise SystemExit("未找到 mediapipe 库，请安装：pip install mediapipe") from exc

        try:
            model_path = self._ensure_pose_landmarker_model()
            base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_poses=5,
                min_pose_detection_confidence=0.45,
                min_pose_presence_confidence=0.45,
                min_tracking_confidence=0.50,
                output_segmentation_masks=False,
            )
            self.mp_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
            self.mp_connections = [
                (c.start, c.end) for c in mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
            ]
            print(">>> MediaPipe PoseLandmarker initialized (Multi-pose enabled)")
        except Exception as exc:
            print(f"PoseLandmarker 初始化失败: {exc}，尝试回退到 Legacy Pose...")
            if hasattr(mp, "solutions"):
                self.mp_pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=2,
                    enable_segmentation=False,
                    min_detection_confidence=0.45,
                    min_tracking_confidence=0.50,
                )
                self.mp_connections = mp.solutions.pose.POSE_CONNECTIONS
                print(">>> MediaPipe Legacy Pose initialized (Single-pose only)")
            else:
                raise SystemExit(f"mediapipe 模型加载失败：{exc}") from exc

        self._open_source(video_source)

    def _open_source(self, video_source):
        if isinstance(video_source, str) and os.path.isfile(video_source):
            ext = os.path.splitext(video_source)[1].lower()
            if ext in {".jpg", ".jpeg", ".png", ".bmp"}:
                frame = cv2.imread(video_source)
                if frame is None:
                    print(f"无法读取图片: {video_source}")
                    raise SystemExit(1)
                self.image_mode = True
                self.static_frame = frame
                return
            self.cap = cv2.VideoCapture(video_source)
        else:
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened() and isinstance(video_source, int):
                for idx in range(4):
                    if idx == video_source:
                        continue
                    candidate = cv2.VideoCapture(idx)
                    if candidate.isOpened():
                        cap.release()
                        cap = candidate
                        video_source = idx
                        print(f"已切换到摄像头 {idx}")
                        break
                    candidate.release()
            self.cap = cap

        if not self.image_mode and (self.cap is None or not self.cap.isOpened()):
            print(f"无法打开视频/摄像头: {video_source}")
            raise SystemExit(1)

        if self.backend == "mediapipe" and not self.image_mode:
            fps = float(self.cap.get(cv2.CAP_PROP_FPS)) if self.cap is not None else 0.0
            if not (fps > 1e-3):
                fps = 30.0
            self._mp_step_ms = max(1, int(round(1000.0 / fps)))

    def _ensure_pose_landmarker_model(self):
        urls = [
            "https://cdn.jsdelivr.net/gh/google-ai-edge/mediapipe@master/mediapipe/tasks/testdata/vision/pose_landmarker_heavy.task",
            "https://mirror.ghproxy.com/https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/tasks/testdata/vision/pose_landmarker_heavy.task",
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
        ]
        base_dir = os.path.join(os.path.expanduser("~"), ".mediapipe")
        os.makedirs(base_dir, exist_ok=True)
        dst = os.path.join(base_dir, "pose_landmarker_heavy.task")
        if os.path.exists(dst) and os.path.getsize(dst) > 1024:
            print(f">>> 使用已有模型: {dst} ({os.path.getsize(dst)//1024}KB)")
            return dst

        print("正在下载 MediaPipe Heavy 模型 (pose_landmarker_heavy.task)...")
        tmp = dst + ".tmp"
        for url in urls:
            try:
                self._download_file(url, tmp)
                if os.path.exists(tmp) and os.path.getsize(tmp) > 1024:
                    os.replace(tmp, dst)
                    print(f">>> 下载完成: {dst} ({os.path.getsize(dst)//1024}KB)")
                    return dst
            except Exception:
                continue
        raise RuntimeError("无法下载模型文件，请检查网络。")

    @staticmethod
    def _download_file(url, dst_path):
        import socket
        socket.setdefaulttimeout(30.0)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response, open(dst_path, "wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)

    def read_frame(self):
        if self.image_mode:
            return True, self.static_frame.copy()
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    @staticmethod
    def _pose_center(kpts):
        """用髋/肩中心作为跨帧选人的稳定锚点。"""
        if kpts is None:
            return None
        pts = []
        for idx in (23, 24, 11, 12):
            if idx < len(kpts) and not np.allclose(kpts[idx], 0):
                pts.append(kpts[idx])
        if not pts:
            return None
        return np.mean(np.asarray(pts, dtype=np.float32), axis=0)

    def _choose_stable_pose(self, poses):
        """多人体时优先选择与上一帧躯干中心最接近的人，避免 result[0] 跳人。"""
        if not poses:
            return None
        if len(poses) == 1 or self._last_pose_center is None:
            chosen = poses[0]
        else:
            scored = []
            for pose in poses:
                center = self._pose_center(pose)
                if center is None:
                    dist = 1e9
                else:
                    dist = float(np.linalg.norm(center - self._last_pose_center))
                scored.append((dist, pose))
            scored.sort(key=lambda item: item[0])
            chosen = scored[0][1]
        center = self._pose_center(chosen)
        if center is not None:
            self._last_pose_center = center
        return chosen

    def infer_keypoints(self, frame):
        self.person_count = 0
        self.all_kpts_list = []

        if self.mp_pose is not None:
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out = self.mp_pose.process(rgb)
            if out.pose_landmarks is None:
                return None
            self.person_count = 1
            kpts = np.zeros((33, 2), dtype=np.float32)
            for idx, lm in enumerate(out.pose_landmarks.landmark):
                visibility_threshold = (self._foot_visibility_threshold
                                        if idx in self._foot_landmark_ids
                                        else self._body_visibility_threshold)
                if float(getattr(lm, "visibility", 1.0)) < visibility_threshold:
                    continue
                x = float(lm.x) * w
                y = float(lm.y) * h
                if 0 <= x < w and 0 <= y < h:
                    kpts[idx] = (x, y)
            self.all_kpts_list.append(kpts)
            chosen = self._choose_stable_pose(self.all_kpts_list)
            return chosen

        if self.mp_landmarker is not None:
            h, w = frame.shape[:2]
            import mediapipe as mp
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.mp_landmarker.detect_for_video(image, int(self._mp_ts_ms))
            self._mp_ts_ms += self._mp_step_ms
            if result is None or not getattr(result, "pose_landmarks", None):
                return None
            self.person_count = len(result.pose_landmarks)
            if self.person_count == 0:
                return None
            for landmarks in result.pose_landmarks:
                person_kpts = np.zeros((33, 2), dtype=np.float32)
                for idx, lm in enumerate(landmarks):
                    visibility_threshold = (self._foot_visibility_threshold
                                            if idx in self._foot_landmark_ids
                                            else self._body_visibility_threshold)
                    if float(getattr(lm, "visibility", 1.0)) < visibility_threshold:
                        continue
                    x = float(lm.x) * w
                    y = float(lm.y) * h
                    if 0 <= x < w and 0 <= y < h:
                        person_kpts[idx] = (x, y)
                self.all_kpts_list.append(person_kpts)
            chosen = self._choose_stable_pose(self.all_kpts_list)
            if chosen is None:
                return None
            # 保留全部人体用于调试绘制，但把本帧选中的运动员放在第一个，
            # 后续逻辑始终使用返回的 chosen。
            self.all_kpts_list = [chosen] + [p for p in self.all_kpts_list if p is not chosen]
            return chosen
        return None

    def release(self):
        if self.cap is not None:
            self.cap.release()

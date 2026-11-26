import math
import os
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np


class OneEuroFilter:
    """Filtro passa-baixas adaptativo (1 Euro) para suavizar landmarks sem lag."""

    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.045, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: Optional[np.ndarray] = None
        self.dx_prev: Optional[np.ndarray] = None
        self.t_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: np.ndarray, dt: float) -> np.ndarray:
        tau = 1.0 / (2.0 * math.pi * np.maximum(cutoff, 1e-4))
        return 1.0 / (1.0 + tau / max(dt, 1e-5))

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def filter(self, x: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(x, dtype=np.float32)
        if self.x_prev is None or self.t_prev is None:
            self.x_prev = arr.copy()
            self.dx_prev = np.zeros_like(arr)
            self.t_prev = t
            return self.x_prev.copy(), self.dx_prev.copy()

        dt = max(t - self.t_prev, 1e-4)
        self.t_prev = t

        dx = (arr - self.x_prev) / dt
        a_d = float(self._alpha(np.asarray(self.d_cutoff, dtype=np.float32), dt))
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * arr + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat.copy(), dx_hat.copy()


class ThreadedCamera:
    """Captura assincrona da webcam em thread dedicada com BUFFERSIZE=1 (zero fila acumulada)."""

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480, fps: int = 60):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._latest_frame: Optional[np.ndarray] = None
        self._seq_id = 0
        self._error: Optional[str] = None

    def start(self) -> bool:
        if self._running:
            return self.cap is not None and self.cap.isOpened()

        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            cap.release()
            self._error = "Webcam nao encontrada (indice 0)."
            return False

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.cap = cap
        self._error = None
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def _capture_loop(self):
        while self._running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._latest_frame = frame
                    self._seq_id += 1
            else:
                time.sleep(0.005)

    def is_ready(self) -> bool:
        return self._running and self.cap is not None and self.cap.isOpened()

    def error_message(self) -> Optional[str]:
        return self._error

    def read_latest(self) -> Tuple[bool, Optional[np.ndarray], int]:
        with self._lock:
            if self._latest_frame is None:
                return False, None, self._seq_id
            return True, self._latest_frame.copy(), self._seq_id

    def stop(self):
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        with self._lock:
            self._latest_frame = None


def hand_openness(landmarks: List[Tuple[float, float, float]]) -> float:
    """Abertura relativa a palma, independente da distancia ate a camera."""
    wrist = landmarks[0]
    palm = landmarks[9]
    palm_size = math.hypot(palm[0] - wrist[0], palm[1] - wrist[1])
    if palm_size < 1e-5:
        return 0.0
    extension = sum(
        math.hypot(landmarks[i][0] - wrist[0], landmarks[i][1] - wrist[1])
        for i in (8, 12, 16, 20)
    ) / (4.0 * palm_size)
    return max(0.0, min(1.0, extension - 1.0))


@dataclass
class HandData:
    label: str  # "Left" ou "Right" (MediaPipe raw label)
    landmarks_norm: List[Tuple[float, float, float]]  # (x, y, z) 0.0 a 1.0
    landmarks_px: List[Tuple[int, int]]  # (x, y) pixels
    fingers_up: List[bool]  # [Polegar, Indicador, Medio, Anelar, Minimo]
    openness: float  # 0.0 a 1.0
    center: Tuple[int, int]  # Centro aproximado da palma (px)
    pinch_distance: float  # Distancia normalizada entre polegar e indicador
    tilt_angle: float  # Inclinacao em graus (-180 a 180)
    tip_velocity: float = 0.0  # Velocidade da ponta do indicador (norm/s)
    palm_vy: float = 0.0  # Velocidade vertical da palma (norm/s, + para baixo)


class HandTracker:
    """Rastreamento de maos com MediaPipe Tasks API + Filtro 1 Euro."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_hands: int = 2,
        min_detection_conf: float = 0.65,
        min_track_conf: float = 0.55,
        min_presence_conf: float = 0.55,
    ):
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "hand_landmarker.task")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_conf,
            min_hand_presence_confidence=min_presence_conf,
            min_tracking_confidence=min_track_conf,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.last_timestamp_ms = 0
        self._filters = {
            "Left": OneEuroFilter(min_cutoff=1.5, beta=0.045, d_cutoff=1.0),
            "Right": OneEuroFilter(min_cutoff=1.5, beta=0.045, d_cutoff=1.0),
        }
        self._last_seen = {"Left": 0.0, "Right": 0.0}

    def process_frame(
        self, frame_bgr: np.ndarray, target_w: Optional[int] = None, target_h: Optional[int] = None
    ) -> Tuple[List[HandData], np.ndarray]:
        h, w, _ = frame_bgr.shape
        render_w = target_w if target_w is not None else w
        render_h = target_h if target_h is not None else h

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        mono_now = time.monotonic()
        now_ms = max(int(mono_now * 1000), self.last_timestamp_ms + 1)
        self.last_timestamp_ms = now_ms
        result = self.detector.detect_for_video(mp_image, now_ms)

        hands_data: List[HandData] = []
        seen_labels = set()

        if result.hand_landmarks:
            for idx, hand_lms in enumerate(result.hand_landmarks):
                label = "Right"
                if result.handedness and idx < len(result.handedness) and result.handedness[idx]:
                    cat = result.handedness[idx][0]
                    label = getattr(cat, "category_name", getattr(cat, "display_name", "Right"))

                filter_key = label if label not in seen_labels else ("Right" if label == "Left" else "Left")
                seen_labels.add(filter_key)

                if mono_now - self._last_seen.get(filter_key, 0.0) > 0.25:
                    self._filters[filter_key].reset()
                self._last_seen[filter_key] = mono_now

                raw_arr = np.array([(lm.x, lm.y, lm.z) for lm in hand_lms], dtype=np.float32)
                filtered_arr, vel_arr = self._filters[filter_key].filter(raw_arr, mono_now)

                lms_norm = [(float(p[0]), float(p[1]), float(p[2])) for p in filtered_arr]
                lms_px = [(int(p[0] * render_w), int(p[1] * render_h)) for p in filtered_arr]

                fingers_up = self._calculate_fingers_up(lms_norm, label)
                openness = hand_openness(lms_norm)

                palm_indices = [0, 5, 9, 13, 17]
                cx = int(sum(lms_px[i][0] for i in palm_indices) / len(palm_indices))
                cy = int(sum(lms_px[i][1] for i in palm_indices) / len(palm_indices))

                p_thumb = lms_norm[4]
                p_index = lms_norm[8]
                pinch_dist = math.hypot(p_thumb[0] - p_index[0], p_thumb[1] - p_index[1])

                dx = lms_norm[9][0] - lms_norm[0][0]
                dy = lms_norm[9][1] - lms_norm[0][1]
                tilt = math.degrees(math.atan2(dx, -dy))

                tip_vel = float(math.hypot(vel_arr[8, 0], vel_arr[8, 1]))
                palm_vy = float(np.mean(vel_arr[palm_indices, 1]))

                hands_data.append(
                    HandData(
                        label=label,
                        landmarks_norm=lms_norm,
                        landmarks_px=lms_px,
                        fingers_up=fingers_up,
                        openness=float(openness),
                        center=(cx, cy),
                        pinch_distance=float(pinch_dist),
                        tilt_angle=float(tilt),
                        tip_velocity=tip_vel,
                        palm_vy=palm_vy,
                    )
                )

        return hands_data, frame_bgr

    def _calculate_fingers_up(self, lms: List[Tuple[float, float, float]], hand_label: str) -> List[bool]:
        fingers = []
        if hand_label == "Right":
            thumb_up = lms[4][0] < lms[3][0]
        else:
            thumb_up = lms[4][0] > lms[3][0]
        fingers.append(thumb_up)

        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        for tip, pip in zip(tips, pips):
            fingers.append(lms[tip][1] < lms[pip][1])

        return fingers

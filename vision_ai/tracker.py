import math
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np


def hand_openness(landmarks: List[Tuple[float, float, float]]) -> float:
    """Abertura relativa à palma, independente da distância até a câmera."""
    wrist = landmarks[0]
    palm = landmarks[9]
    palm_size = math.hypot(palm[0] - wrist[0], palm[1] - wrist[1])
    if palm_size < 1e-5:
        return 0.0
    # Quatro dedos evitam que a posição lateral do polegar domine a medida.
    extension = sum(
        math.hypot(landmarks[i][0] - wrist[0], landmarks[i][1] - wrist[1])
        for i in (8, 12, 16, 20)
    ) / (4.0 * palm_size)
    return max(0.0, min(1.0, extension - 1.0))


@dataclass
class HandData:
    label: str  # "Left" ou "Right"
    landmarks_norm: List[Tuple[float, float, float]]  # (x, y, z) 0.0 a 1.0
    landmarks_px: List[Tuple[int, int]]  # (x, y) pixels
    fingers_up: List[bool]  # [Polegar, Indicador, Médio, Anelar, Mínimo]
    openness: float  # 0.0 (punho fechado) a 1.0 (mão totalmente aberta)
    center: Tuple[int, int]  # Centro aproximado da palma
    pinch_distance: float  # Distância entre polegar e indicador (normalizada)
    tilt_angle: float  # Inclinação em graus (-90 a 90)


class HandTracker:
    """Rastreamento de mãos e extração de cinemática com MediaPipe Tasks API."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_hands: int = 2,
        min_detection_conf: float = 0.6,
        min_track_conf: float = 0.5,
        min_presence_conf: float = 0.5,
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

    def process_frame(
        self, frame_bgr: np.ndarray, target_w: Optional[int] = None, target_h: Optional[int] = None
    ) -> Tuple[List[HandData], np.ndarray]:
        h, w, _ = frame_bgr.shape
        render_w = target_w if target_w is not None else w
        render_h = target_h if target_h is not None else h

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        now_ms = max(int(time.time() * 1000), self.last_timestamp_ms + 1)
        self.last_timestamp_ms = now_ms
        result = self.detector.detect_for_video(mp_image, now_ms)

        hands_data: List[HandData] = []

        if result.hand_landmarks:
            for idx, hand_lms in enumerate(result.hand_landmarks):
                # Determina handedness ("Left" ou "Right")
                label = "Right"
                if result.handedness and idx < len(result.handedness) and result.handedness[idx]:
                    cat = result.handedness[idx][0]
                    label = getattr(cat, "category_name", getattr(cat, "display_name", "Right"))

                lms_norm = [(lm.x, lm.y, lm.z) for lm in hand_lms]
                # Projeta os pontos diretamente na resolução HD de renderização
                lms_px = [(int(lm.x * render_w), int(lm.y * render_h)) for lm in hand_lms]

                # Finger states (up/down)
                fingers_up = self._calculate_fingers_up(lms_norm, label)

                openness = hand_openness(lms_norm)

                # Centro da palma (média dos pontos 0, 5, 9, 13, 17)
                palm_pts = [lms_px[i] for i in [0, 5, 9, 13, 17]]
                cx = int(sum(p[0] for p in palm_pts) / len(palm_pts))
                cy = int(sum(p[1] for p in palm_pts) / len(palm_pts))

                # Pinch distance (Polegar tip 4 e Indicador tip 8)
                p_thumb = lms_norm[4]
                p_index = lms_norm[8]
                pinch_dist = math.sqrt((p_thumb[0] - p_index[0]) ** 2 + (p_thumb[1] - p_index[1]) ** 2)

                # Tilt angle (inclinação do pulso ao dedo médio)
                dx = lms_norm[9][0] - lms_norm[0][0]
                dy = lms_norm[9][1] - lms_norm[0][1]
                tilt = math.degrees(math.atan2(dx, -dy))

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
                    )
                )

        return hands_data, frame_bgr

    def _calculate_fingers_up(self, lms: List[Tuple[float, float, float]], hand_label: str) -> List[bool]:
        fingers = []

        # Polegar: verifica coordenada X relativo à base do polegar
        if hand_label == "Right":
            thumb_up = lms[4][0] < lms[3][0]
        else:
            thumb_up = lms[4][0] > lms[3][0]
        fingers.append(thumb_up)

        # 4 Dedos: ponta (tip) acima da articulação (pip)
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        for tip, pip in zip(tips, pips):
            fingers.append(lms[tip][1] < lms[pip][1])

        return fingers

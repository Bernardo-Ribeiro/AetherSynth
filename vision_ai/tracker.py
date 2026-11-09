import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np


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

    def __init__(self, model_path: Optional[str] = None, max_hands: int = 2):
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "hand_landmarker.task")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[List[HandData], np.ndarray]:
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect(mp_image)

        hands_data: List[HandData] = []

        if result.hand_landmarks:
            for idx, hand_lms in enumerate(result.hand_landmarks):
                # Determina handedness ("Left" ou "Right")
                label = "Right"
                if result.handedness and idx < len(result.handedness) and result.handedness[idx]:
                    cat = result.handedness[idx][0]
                    label = getattr(cat, "category_name", getattr(cat, "display_name", "Right"))

                lms_norm = [(lm.x, lm.y, lm.z) for lm in hand_lms]
                lms_px = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]

                # Finger states (up/down)
                fingers_up = self._calculate_fingers_up(lms_norm, label)

                # Openness (distância média das pontas ao pulso)
                wrist = lms_norm[0]
                tips = [4, 8, 12, 16, 20]
                dists = [
                    math.sqrt((lms_norm[tip][0] - wrist[0]) ** 2 + (lms_norm[tip][1] - wrist[1]) ** 2)
                    for tip in tips
                ]
                avg_dist = sum(dists) / len(dists)
                openness = np.clip((avg_dist - 0.15) / (0.42 - 0.15), 0.0, 1.0)

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

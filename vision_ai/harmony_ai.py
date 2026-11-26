import math
from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np

from gestures import PITCH_CLASS_CIFRA, PITCH_CLASS_PT

# Perfis tonais cognitivos de Krumhansl-Kessler
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float32)
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float32)

MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11, 12]
MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10, 12]


def chord_tones_for(root_pc: Optional[int], is_minor: bool) -> Set[int]:
    if root_pc is None or root_pc < 0 or root_pc > 11:
        return set()
    third = 3 if is_minor else 4
    return {root_pc % 12, (root_pc + third) % 12, (root_pc + 7) % 12}


def build_diatonic_scale(root_pc: int, is_minor: bool) -> List[int]:
    """Gera 8 notas MIDI da escala diatonica na regiao C4..E5 para os pads."""
    base = 60 + (root_pc % 12)
    if base > 67:
        base -= 12
    intervals = MINOR_INTERVALS if is_minor else MAJOR_INTERVALS
    return [base + iv for iv in intervals]


@dataclass
class DetectedKey:
    root_pc: int
    is_minor: bool
    key_name: str
    short_key: str
    confidence: float
    scale_notes: List[int]
    locked: bool


class HarmonyKeyDetector:
    """
    IA Probabilistica de Deteccao de Tonalidade (Krumhansl-Kessler + Verossimilhanca Diatonica).
    Analisa a progressao de acordes do Loop Session e trava a escala segura para melodia.
    """

    def __init__(self):
        self.last_detected: Optional[DetectedKey] = None

    @staticmethod
    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        a_c = a - np.mean(a)
        b_c = b - np.mean(b)
        denom = float(np.linalg.norm(a_c) * np.linalg.norm(b_c))
        if denom < 1e-8:
            return 0.0
        return float(np.dot(a_c, b_c) / denom)

    def detect_from_events(self, chord_events: List[Dict], locked: bool = True) -> Optional[DetectedKey]:
        if not chord_events:
            self.last_detected = None
            return None

        pitch_hist = np.zeros(12, dtype=np.float32)
        event_pcs: List[tuple] = []

        for idx, ev in enumerate(chord_events):
            root = int(ev["root_pc"]) % 12
            is_min = bool(ev["is_minor"])
            dur = max(0.25, float(ev.get("duration_beats", 1.0)))
            start = float(ev.get("start_beat", 0.0))

            # Peso metrico 1.25x para acordes iniciados no tempo 1 de um compasso
            beat_in_bar = start % 4.0
            metric_weight = 1.25 if (beat_in_bar < 0.35 or beat_in_bar > 3.65) else 1.0
            w = dur * metric_weight

            third = (root + (3 if is_min else 4)) % 12
            fifth = (root + 7) % 12
            pitch_hist[root] += w * 1.00
            pitch_hist[third] += w * 0.72
            pitch_hist[fifth] += w * 0.80
            event_pcs.append((root, is_min, {root, third, fifth}, w, idx))

        total_w = sum(item[3] for item in event_pcs) + 1e-6
        scores = []
        candidates = []

        for tonic in range(12):
            for is_min in (False, True):
                profile = np.roll(KK_MINOR if is_min else KK_MAJOR, tonic)
                kk_corr = self._pearson(pitch_hist, profile)

                scale_pcs = {(tonic + iv) % 12 for iv in (MINOR_INTERVALS if is_min else MAJOR_INTERVALS)}
                diatonic_hits = 0.0
                for root, ev_min, triad, w, _ in event_pcs:
                    in_scale = sum(1 for pc in triad if pc in scale_pcs) / 3.0
                    # Penalidade leve se o acorde inteiro pertence mas qualidade diatonica difere
                    diatonic_hits += w * in_scale

                diatonic_ratio = diatonic_hits / total_w

                # Bonus de cadencia / tônica no primeiro ou ultimo acorde do loop
                cadence_bonus = 0.0
                first_root, first_min = event_pcs[0][0], event_pcs[0][1]
                last_root, last_min = event_pcs[-1][0], event_pcs[-1][1]
                if first_root == tonic and first_min == is_min:
                    cadence_bonus += 0.18
                if last_root == tonic and last_min == is_min:
                    cadence_bonus += 0.14
                # Dominante maior (V) resolvendo ou fechando ciclo
                dom = (tonic + 7) % 12
                if any(r == dom and not m for r, m, _, _, _ in event_pcs):
                    cadence_bonus += 0.08

                score = 1.6 * kk_corr + 2.4 * diatonic_ratio + cadence_bonus
                scores.append(score)
                candidates.append((tonic, is_min))

        scores_arr = np.asarray(scores, dtype=np.float32)
        shifted = (scores_arr - np.max(scores_arr)) * 3.0
        probs = np.exp(shifted)
        probs /= np.sum(probs)

        best_idx = int(np.argmax(scores_arr))
        tonic, is_min = candidates[best_idx]
        conf = float(probs[best_idx])

        qual_pt = "Menor" if is_min else "Maior"
        key_name = f"{PITCH_CLASS_PT[tonic]} {qual_pt}"
        short_key = PITCH_CLASS_CIFRA[tonic] + ("m" if is_min else "")

        result = DetectedKey(
            root_pc=tonic,
            is_minor=is_min,
            key_name=key_name,
            short_key=short_key,
            confidence=conf,
            scale_notes=build_diatonic_scale(tonic, is_min),
            locked=locked,
        )
        self.last_detected = result
        return result

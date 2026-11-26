import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from tracker import HandData


# Escalas musicais manuais de fallback (notas MIDI)
SCALES = {
    "Do Maior (C Major)": [60, 62, 64, 65, 67, 69, 71, 72],
    "Pentatonica (A Menor)": [57, 60, 62, 64, 67, 69, 72],
    "Blues (C Blues)": [60, 63, 65, 66, 67, 70, 72],
    "Akebono (Japonesa)": [60, 62, 63, 67, 68, 72],
}

# Classes da Rede Neural MLP (Estagio A - Forma Intrinseca)
SHAPE_CLASSES = ["Punho", "Do", "Re", "Mi", "Fa", "Sol", "La", "Si"]

# Pitch classes (0..11) correspondentes as formas 1..7 (Do..Si)
SHAPE_TO_PITCH_CLASS = {
    1: 0,   # Do (C)
    2: 2,   # Re (D)
    3: 4,   # Mi (E)
    4: 5,   # Fa (F)
    5: 7,   # Sol (G)
    6: 9,   # La (A)
    7: 11,  # Si (B)
}

PITCH_CLASS_PT = ["Do", "Do#", "Re", "Mib", "Mi", "Fa", "Fa#", "Sol", "Lab", "La", "Sib", "Si"]
PITCH_CLASS_CIFRA = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def get_note_name(midi_pitch: int) -> str:
    """Retorna o nome ASCII limpo da nota (compativel com fontes OpenCV)."""
    octave = (midi_pitch // 12) - 1
    note = PITCH_CLASS_PT[midi_pitch % 12]
    return f"{note}{octave}"


def build_chord_midi(root_pc: int, is_minor: bool) -> List[int]:
    """Constroi voicing grave + triade na regiao media (C3/C4) para o motor Rust."""
    bass = 48 + (root_pc % 12)
    if bass > 55:
        bass -= 12
    root_mid = 60 + (root_pc % 12)
    third = root_mid + (3 if is_minor else 4)
    fifth = root_mid + 7
    return [bass, root_mid, third, fifth]


def _unit_vec(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-6:
        return np.zeros_like(v)
    return v / norm


def extract_geometric_features(landmarks_norm: List[Tuple[float, float, float]]) -> np.ndarray:
    """
    Extrai 30 atributos 3D invariantes a translacao, escala e rotacao 3D:
      - 15 cossenos de angulos articulares (3 por dedo, 5 dedos)
      - 5 distancias ponta->pulso normalizadas pela palma (L = ||P9 - P0||)
      - 5 distancias ponta->base(MCP) normalizadas pela palma
      - 5 relacoes interdigitais normalizadas pela palma (4-8, 8-12, 12-16, 16-20, 4-9)
    """
    pts = np.asarray(landmarks_norm, dtype=np.float32)
    wrist = pts[0]
    palm_size = float(np.linalg.norm(pts[9] - wrist))
    if palm_size < 1e-5:
        palm_size = 1e-5

    joint_triples = [
        (0, 1, 2), (1, 2, 3), (2, 3, 4),
        (0, 5, 6), (5, 6, 7), (6, 7, 8),
        (0, 9, 10), (9, 10, 11), (10, 11, 12),
        (0, 13, 14), (13, 14, 15), (14, 15, 16),
        (0, 17, 18), (17, 18, 19), (18, 19, 20),
    ]
    angles = []
    for a, b, c in joint_triples:
        v1 = _unit_vec(pts[b] - pts[a])
        v2 = _unit_vec(pts[c] - pts[b])
        angles.append(float(np.clip(np.dot(v1, v2), -1.0, 1.0)))

    tips = [4, 8, 12, 16, 20]
    tip_wrist = [float(np.linalg.norm(pts[t] - wrist) / palm_size) for t in tips]

    tip_mcp_pairs = [(4, 5), (8, 5), (12, 9), (16, 13), (20, 17)]
    tip_mcp = [float(np.linalg.norm(pts[t] - pts[m]) / palm_size) for t, m in tip_mcp_pairs]

    inter_pairs = [(4, 8), (8, 12), (12, 16), (16, 20), (4, 9)]
    inter = [float(np.linalg.norm(pts[a] - pts[b]) / palm_size) for a, b in inter_pairs]

    return np.asarray(angles + tip_wrist + tip_mcp + inter, dtype=np.float32)


def _random_rotation_3d(rng: np.random.Generator) -> np.ndarray:
    """Gera matriz de rotacao aleatoria em SO(3) via decomposicao QR."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    d = np.diag(np.sign(np.diag(r)))
    q = q @ d
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q.astype(np.float32)


def _synthesize_hand_landmarks(class_idx: int, rng: np.random.Generator) -> np.ndarray:
    """
    Gera um esqueleto 3D anatomico de mao para cada classe (0..7)
    com variacao natural de flexao, proporcoes osseas e rotacao SO(3) aleatoria.
    """
    pts = np.zeros((21, 3), dtype=np.float32)

    # Estados alvo dos dedos: [Polegar, Indicador, Medio, Anelar, Minimo]
    # Para polegar: 'open' (abduzido), 'up' (esticado p/ Si), 'tuck' (cruzado palma), 'fist' (fechado)
    if class_idx == 0:  # Punho
        thumb_mode, fingers_ext = "fist", [False, False, False, False]
    elif class_idx == 1:  # Do (1)
        thumb_mode, fingers_ext = "tuck", [True, False, False, False]
    elif class_idx == 2:  # Re (2)
        thumb_mode, fingers_ext = "tuck", [True, True, False, False]
    elif class_idx == 3:  # Mi (3)
        thumb_mode, fingers_ext = "tuck", [True, True, True, False]
    elif class_idx == 4:  # Fa (4)
        thumb_mode, fingers_ext = "tuck", [True, True, True, True]
    elif class_idx == 5:  # Sol (5)
        thumb_mode, fingers_ext = "open", [True, True, True, True]
    elif class_idx == 6:  # La (Hang-loose)
        thumb_mode, fingers_ext = "open", [False, False, False, True]
    else:  # 7: Si (Apenas polegar estendido)
        thumb_mode, fingers_ext = "up", [False, False, False, False]

    scale_jitter = rng.uniform(0.92, 1.08)

    # Bases MCP dos 4 dedos
    mcp_bases = [
        np.array([0.24, -0.95, 0.0], dtype=np.float32),
        np.array([0.02, -1.00, 0.0], dtype=np.float32),
        np.array([-0.20, -0.95, 0.0], dtype=np.float32),
        np.array([-0.40, -0.85, 0.0], dtype=np.float32),
    ]
    finger_roots = [5, 9, 13, 17]
    bone_lens = [
        (0.42, 0.27, 0.20),
        (0.45, 0.30, 0.22),
        (0.42, 0.28, 0.21),
        (0.35, 0.22, 0.17),
    ]
    splays = [0.14, 0.02, -0.09, -0.20]

    for f_idx, (root_idx, base, (l1, l2, l3), splay) in enumerate(
        zip(finger_roots, mcp_bases, bone_lens, splays)
    ):
        pts[root_idx] = base * rng.uniform(0.95, 1.05, size=3).astype(np.float32)
        if fingers_ext[f_idx]:
            a1 = rng.uniform(-0.10, 0.18)
            a2 = rng.uniform(0.02, 0.22)
            a3 = rng.uniform(0.02, 0.20)
            s = splay + rng.uniform(-0.08, 0.08)
        else:
            a1 = rng.uniform(1.35, 1.85)
            a2 = rng.uniform(1.45, 2.00)
            a3 = rng.uniform(1.05, 1.65)
            s = splay * 0.4 + rng.uniform(-0.05, 0.05)

        cur = pts[root_idx].copy()
        ang = a1
        for k, seg_len in enumerate((l1, l2, l3)):
            if k == 1:
                ang += a2
            elif k == 2:
                ang += a3
            dx = math.sin(s) * math.cos(ang * 0.5)
            dy = -math.cos(ang) * math.cos(s)
            dz = math.sin(ang)
            cur = cur + np.array([dx, dy, dz], dtype=np.float32) * seg_len * rng.uniform(0.94, 1.06)
            pts[root_idx + k + 1] = cur

    # Polegar (pontos 1, 2, 3, 4)
    pts[1] = np.array([0.20, -0.22, 0.04], dtype=np.float32)
    pts[2] = np.array([0.38, -0.44, 0.06], dtype=np.float32)

    if thumb_mode == "open":
        v23 = np.array([0.22, -0.18, 0.03], dtype=np.float32)
        v34 = np.array([0.20, -0.16, 0.02], dtype=np.float32)
    elif thumb_mode == "up":
        v23 = np.array([0.08, -0.32, 0.02], dtype=np.float32)
        v34 = np.array([0.04, -0.28, 0.02], dtype=np.float32)
    elif thumb_mode == "tuck":
        v23 = np.array([-0.24, -0.10, 0.16], dtype=np.float32)
        v34 = np.array([-0.22, -0.02, 0.10], dtype=np.float32)
    else:  # fist
        v23 = np.array([-0.14, -0.16, 0.24], dtype=np.float32)
        v34 = np.array([-0.16, 0.06, 0.18], dtype=np.float32)

    pts[3] = pts[2] + v23 + rng.normal(0, 0.025, size=3).astype(np.float32)
    pts[4] = pts[3] + v34 + rng.normal(0, 0.025, size=3).astype(np.float32)

    # Aplicacao de rotacao aleatoria 3D + escala + pequeno ruido de landmark
    rot = _random_rotation_3d(rng)
    pts = (pts @ rot.T) * scale_jitter
    pts += rng.normal(0, 0.012, size=pts.shape).astype(np.float32)
    return pts


class MLPChordClassifier:
    """
    Rede Neural MLP (30 -> 64 -> 32 -> 8) para classificacao dos sinais de acordes.
    Treinada com Scikit-Learn MLPClassifier + StandardScaler e executada em NumPy puro (~0.01ms).
    """

    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        self.weights_path = os.path.join(self.models_dir, "chord_mlp_weights.npz")
        self.user_samples_path = os.path.join(self.models_dir, "user_samples.npz")

        self.mean = np.zeros(30, dtype=np.float32)
        self.scale = np.ones(30, dtype=np.float32)
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self.val_accuracy = 0.0

        self.user_X: Optional[np.ndarray] = None
        self.user_y: Optional[np.ndarray] = None
        self._load_user_samples()

        if os.path.exists(self.weights_path):
            try:
                self._load_weights()
                return
            except Exception:
                pass
        self.train_and_save()

    def _generate_base_dataset(self, samples_per_class: int = 300, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        X_list, y_list = [], []
        for cls_idx in range(len(SHAPE_CLASSES)):
            for _ in range(samples_per_class):
                pts = _synthesize_hand_landmarks(cls_idx, rng)
                feat = extract_geometric_features(pts)
                X_list.append(feat)
                y_list.append(cls_idx)
        return np.asarray(X_list, dtype=np.float32), np.asarray(y_list, dtype=np.int64)

    def train_and_save(self, include_user_samples: bool = True) -> float:
        X_base, y_base = self._generate_base_dataset(samples_per_class=300, seed=42)
        X_train_full, X_val, y_train_full, y_val = train_test_split(
            X_base, y_base, test_size=0.2, stratify=y_base, random_state=42
        )

        if include_user_samples and self.user_X is not None and len(self.user_X) > 0:
            # Oversample 3x das amostras reais da mao do usuario para adaptar rapido
            X_u = np.repeat(self.user_X, 3, axis=0)
            y_u = np.repeat(self.user_y, 3, axis=0)
            X_train_full = np.vstack([X_train_full, X_u])
            y_train_full = np.concatenate([y_train_full, y_u])

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_train_full)
        X_val_scaled = scaler.transform(X_val)

        mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            max_iter=400,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=42,
        )
        mlp.fit(X_tr_scaled, y_train_full)
        self.val_accuracy = float(mlp.score(X_val_scaled, y_val))

        self.mean = scaler.mean_.astype(np.float32)
        self.scale = np.where(scaler.scale_ < 1e-6, 1.0, scaler.scale_).astype(np.float32)
        self.weights = [w.astype(np.float32) for w in mlp.coefs_]
        self.biases = [b.astype(np.float32) for b in mlp.intercepts_]
        self._save_weights()
        return self.val_accuracy

    def _save_weights(self):
        np.savez(
            self.weights_path,
            mean=self.mean,
            scale=self.scale,
            W0=self.weights[0],
            b0=self.biases[0],
            W1=self.weights[1],
            b1=self.biases[1],
            W2=self.weights[2],
            b2=self.biases[2],
            val_accuracy=np.array([self.val_accuracy], dtype=np.float32),
        )

    def _load_weights(self):
        data = np.load(self.weights_path)
        self.mean = data["mean"].astype(np.float32)
        self.scale = data["scale"].astype(np.float32)
        self.weights = [data["W0"].astype(np.float32), data["W1"].astype(np.float32), data["W2"].astype(np.float32)]
        self.biases = [data["b0"].astype(np.float32), data["b1"].astype(np.float32), data["b2"].astype(np.float32)]
        self.val_accuracy = float(data["val_accuracy"][0])

    def _load_user_samples(self):
        if os.path.exists(self.user_samples_path):
            try:
                data = np.load(self.user_samples_path)
                self.user_X = data["X"].astype(np.float32)
                self.user_y = data["y"].astype(np.int64)
            except Exception:
                self.user_X = None
                self.user_y = None

    def record_user_sample(self, class_idx: int, landmarks_norm: List[Tuple[float, float, float]]):
        feat = extract_geometric_features(landmarks_norm).reshape(1, -1)
        lbl = np.array([class_idx], dtype=np.int64)
        if self.user_X is None or len(self.user_X) == 0:
            self.user_X = feat
            self.user_y = lbl
        else:
            self.user_X = np.vstack([self.user_X, feat])
            self.user_y = np.concatenate([self.user_y, lbl])

    def clear_class_samples(self, class_idx: int):
        if self.user_X is None or self.user_y is None:
            return
        mask = self.user_y != class_idx
        self.user_X = self.user_X[mask]
        self.user_y = self.user_y[mask]

    def user_sample_counts(self) -> List[int]:
        counts = [0] * len(SHAPE_CLASSES)
        if self.user_y is not None:
            for c in range(len(SHAPE_CLASSES)):
                counts[c] = int(np.sum(self.user_y == c))
        return counts

    def retrain_from_user_samples(self) -> float:
        if self.user_X is not None and len(self.user_X) > 0:
            np.savez(self.user_samples_path, X=self.user_X, y=self.user_y)
        return self.train_and_save(include_user_samples=True)

    def reset_to_default_model(self) -> float:
        self.user_X = None
        self.user_y = None
        if os.path.exists(self.user_samples_path):
            os.remove(self.user_samples_path)
        return self.train_and_save(include_user_samples=False)

    def predict_shape(self, landmarks_norm: List[Tuple[float, float, float]]) -> Tuple[int, str, float]:
        """Inferencia puramente matricial (MLP forward pass em NumPy, ~0.01ms)."""
        feat = extract_geometric_features(landmarks_norm)
        x = (feat - self.mean) / self.scale

        h1 = np.maximum(0.0, x @ self.weights[0] + self.biases[0])
        h2 = np.maximum(0.0, h1 @ self.weights[1] + self.biases[1])
        logits = h2 @ self.weights[2] + self.biases[2]

        shifted = logits - np.max(logits)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores)

        best_idx = int(np.argmax(probs))
        conf = float(probs[best_idx])
        return best_idx, SHAPE_CLASSES[best_idx], conf

    def predict(self, hand: HandData) -> Tuple[str, float]:
        _, name, conf = self.predict_shape(hand.landmarks_norm)
        return name, conf


# Alias para compatibilidade
MLGestureClassifier = MLPChordClassifier


@dataclass
class ChordDetection:
    shape_idx: int
    shape_name: str
    confidence: float
    orientation: str  # "up", "down", "right", "left"
    root_pc: Optional[int]
    is_minor: bool
    chord_label: str
    short_label: str
    midi_notes: List[int]
    arrow_dir: str  # "up", "down", "right", "left", "none"


class ChordInterpreter:
    """
    Estagio B: Combina a forma da Rede Neural (MLP) com orientacao espacial
    + histerese angular (±10 deg) + estabilizacao temporal para sinais de palco.
    """

    def __init__(self, classifier: MLPChordClassifier):
        self.classifier = classifier
        self._last_orientation: str = "up"
        self._candidate_key: Optional[Tuple[int, str]] = None
        self._candidate_frames: int = 0
        self._stable_detection: Optional[ChordDetection] = None
        self._low_conf_streak: int = 0

    def reset(self):
        self._candidate_key = None
        self._candidate_frames = 0
        self._stable_detection = None
        self._low_conf_streak = 0

    def _classify_orientation(self, landmarks_norm: List[Tuple[float, float, float]], shape_idx: int) -> str:
        # Si (classe 7) usa vetor do polegar (P4 - P2); Do..La (1..6) usam eixo da palma (P9 - P0).
        if shape_idx == 7:
            dx = landmarks_norm[4][0] - landmarks_norm[2][0]
            dy = landmarks_norm[4][1] - landmarks_norm[2][1]
        else:
            dx = landmarks_norm[9][0] - landmarks_norm[0][0]
            dy = landmarks_norm[9][1] - landmarks_norm[0][1]

        angle = math.degrees(math.atan2(-dy, dx))  # +90 = cima, -90 = baixo, 0 = direita, ±180 = esquerda
        hyst = 10.0

        prev = self._last_orientation
        if prev == "up" and (45.0 - hyst) <= angle <= (135.0 + hyst):
            return "up"
        if prev == "down" and (-135.0 - hyst) <= angle <= (-45.0 + hyst):
            return "down"
        if prev == "right" and (-45.0 - hyst) <= angle <= (45.0 + hyst):
            return "right"
        if prev == "left" and (angle >= (135.0 - hyst) or angle <= (-135.0 + hyst)):
            return "left"

        if 45.0 <= angle <= 135.0:
            chosen = "up"
        elif -135.0 <= angle <= -45.0:
            chosen = "down"
        elif -45.0 < angle < 45.0:
            chosen = "right"
        else:
            chosen = "left"

        self._last_orientation = chosen
        return chosen

    def _build_detection(
        self, shape_idx: int, shape_name: str, conf: float, orientation: str
    ) -> ChordDetection:
        if shape_idx == 0:
            return ChordDetection(
                shape_idx=0,
                shape_name=shape_name,
                confidence=conf,
                orientation=orientation,
                root_pc=None,
                is_minor=False,
                chord_label="Silencio",
                short_label="-",
                midi_notes=[],
                arrow_dir="none",
            )

        base_pc = SHAPE_TO_PITCH_CLASS[shape_idx]
        if orientation == "up":
            root_pc = base_pc
            is_minor = False
        elif orientation == "down":
            root_pc = base_pc
            is_minor = True
        elif orientation == "right":
            root_pc = (base_pc + 1) % 12
            is_minor = False
        else:  # left
            root_pc = (base_pc - 1) % 12
            is_minor = False

        pt_name = PITCH_CLASS_PT[root_pc]
        cifra = PITCH_CLASS_CIFRA[root_pc] + ("m" if is_minor else "")
        qual = "Menor" if is_minor else "Maior"
        return ChordDetection(
            shape_idx=shape_idx,
            shape_name=shape_name,
            confidence=conf,
            orientation=orientation,
            root_pc=root_pc,
            is_minor=is_minor,
            chord_label=f"{pt_name} {qual}",
            short_label=cifra,
            midi_notes=build_chord_midi(root_pc, is_minor),
            arrow_dir=orientation,
        )

    def update(self, hand: Optional[HandData]) -> Optional[ChordDetection]:
        if hand is None:
            self.reset()
            return None

        shape_idx, shape_name, conf = self.classifier.predict_shape(hand.landmarks_norm)

        if conf < 0.58:
            self._low_conf_streak += 1
            if self._low_conf_streak <= 2 and self._stable_detection is not None:
                return self._stable_detection
            self._candidate_key = None
            self._candidate_frames = 0
            return self._stable_detection

        self._low_conf_streak = 0
        orientation = "none" if shape_idx == 0 else self._classify_orientation(hand.landmarks_norm, shape_idx)
        key = (shape_idx, orientation)

        if key == self._candidate_key:
            self._candidate_frames += 1
        else:
            self._candidate_key = key
            self._candidate_frames = 1

        required_frames = 2 if (shape_idx == 0 or conf >= 0.78) else 3
        if self._candidate_frames >= required_frames or self._stable_detection is None:
            self._stable_detection = self._build_detection(shape_idx, shape_name, conf, orientation)
        else:
            # Atualiza confianca visual sem trocar de acorde antes de estabilizar
            self._stable_detection.confidence = conf

        return self._stable_detection


class HolographicVirtualKeyboard:
    """Teclas virtuais flutuantes adaptaveis ao tom detectado pela IA."""

    def __init__(self, scale_name: str = "Do Maior (C Major)"):
        self.scale_name = scale_name
        self.notes = SCALES.get(scale_name, SCALES["Do Maior (C Major)"])
        self.chord_tone_pcs: set = set()
        self.is_auto_locked: bool = False
        self._cached_key = None
        self._cached_zones: List[dict] = []

    def set_scale(self, scale_name: str):
        if scale_name in SCALES:
            self.scale_name = scale_name
            self.notes = SCALES[scale_name]
            self.is_auto_locked = False
            self._cached_key = None

    def set_custom_scale(
        self,
        scale_name: str,
        notes: List[int],
        chord_tone_pcs: Optional[set] = None,
        locked: bool = True,
    ):
        self.scale_name = scale_name
        self.notes = list(notes)
        self.chord_tone_pcs = set(chord_tone_pcs) if chord_tone_pcs else set()
        self.is_auto_locked = locked
        self._cached_key = None

    def set_active_chord_tones(self, chord_tone_pcs: set):
        self.chord_tone_pcs = set(chord_tone_pcs)

    def get_zones(self, frame_w: int, frame_h: int) -> List[dict]:
        cache_key = (frame_w, frame_h, tuple(self.notes))
        if cache_key == self._cached_key:
            return self._cached_zones

        n = max(1, len(self.notes))
        margin_x = int(frame_w * 0.08)
        width_avail = frame_w - (2 * margin_x)
        key_w = width_avail // n
        key_h = int(frame_h * 0.19)
        key_y = int(frame_h * 0.16)

        zones = []
        for i, note in enumerate(self.notes):
            kx = margin_x + i * key_w
            zones.append({
                "note": note,
                "pc": note % 12,
                "name": get_note_name(note),
                "rect": (kx, key_y, key_w - 8, key_h),
                "center": (kx + (key_w - 8) // 2, key_y + key_h // 2),
                "radius": min(key_w - 8, key_h) // 2,
            })
        self._cached_key = cache_key
        self._cached_zones = zones
        return zones

    def check_hit(self, hand: HandData, frame_w: int, frame_h: int) -> Optional[int]:
        tip_x, tip_y = hand.landmarks_px[8]
        for zone in self.get_zones(frame_w, frame_h):
            x, y, w, h = zone["rect"]
            if x <= tip_x <= x + w and y <= tip_y <= y + h:
                if hand.fingers_up[1] or hand.pinch_distance < 0.08:
                    return zone["note"]
        return None

    @staticmethod
    def compute_strike_velocity(hand: HandData) -> float:
        """Mapeia velocidade do indicador (1-Euro) para expressividade dinamica [0.55..1.0]."""
        return float(np.clip(0.55 + hand.tip_velocity * 0.35, 0.55, 1.0))

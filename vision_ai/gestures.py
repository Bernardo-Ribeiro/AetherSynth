from typing import List, Optional, Tuple
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from tracker import HandData


# Escalas musicais (notas MIDI)
SCALES = {
    "Dó Maior (C Major)": [60, 62, 64, 65, 67, 69, 71, 72],  # C4, D4, E4, F4, G4, A4, B4, C5
    "Pentatônica (A Menor)": [57, 60, 62, 64, 67, 69, 72],    # A3, C4, D4, E4, G4, A4, C5
    "Blues (C Blues)": [60, 63, 65, 66, 67, 70, 72],          # C, Eb, F, F#, G, Bb, C
    "Akebono (Japonesa)": [60, 62, 63, 67, 68, 72],           # C, D, Eb, G, Ab, C
}

NOTE_NAMES = {
    57: "Lá3", 60: "Dó4", 62: "Ré4", 63: "Mib4", 64: "Mi4", 65: "Fá4",
    66: "Fá#4", 67: "Sol4", 68: "Láb4", 69: "Lá4", 70: "Sib4", 71: "Si4", 72: "Dó5"
}


def get_note_name(midi_pitch: int) -> str:
    names = ["Dó", "Dó#", "Ré", "Ré#", "Mi", "Fá", "Fá#", "Sol", "Sol#", "Lá", "Lá#", "Si"]
    octave = (midi_pitch // 12) - 1
    note = names[midi_pitch % 12]
    return f"{note}{octave}"


class MLGestureClassifier:
    """Classificador de Machine Learning baseado em KNN para reconhecimento de posturas da mão."""

    GESTURE_LABELS = [
        "Fist (Fechado)",
        "Open Palm (Aberta)",
        "Point (Apontando)",
        "Peace (Paz/Dois)",
        "Three (Três)",
        "Four (Quatro)",
        "Pinch (Pinça)",
        "Rock (Chifre)",
        "Thumbs Up (Joinha)",
    ]

    def __init__(self):
        self.clf = KNeighborsClassifier(n_neighbors=3, weights="distance")
        self.is_trained = False
        self._init_synthetic_training()

    def _extract_features(self, landmarks_norm: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        Extrai vetor de características invariante a escala e translação:
        Distâncias euclidianas normalizadas de todos os 20 pontos em relação ao pulso (ponto 0),
        divididas pelo tamanho da palma (distância entre ponto 0 e ponto 9).
        """
        pts = np.array(landmarks_norm)[:, :2]  # apenas X e Y
        wrist = pts[0]
        palm_size = np.linalg.norm(pts[9] - wrist)
        if palm_size < 1e-5:
            palm_size = 1e-5

        # Distâncias normalizadas de cada ponto até o pulso
        dists = np.linalg.norm(pts[1:] - wrist, axis=1) / palm_size

        # Distâncias entre as pontas dos dedos adjacentes (4-8, 8-12, 12-16, 16-20)
        tips = [4, 8, 12, 16, 20]
        tip_dists = [np.linalg.norm(pts[tips[i]] - pts[tips[i + 1]]) / palm_size for i in range(len(tips) - 1)]

        return np.concatenate([dists, tip_dists])

    def _init_synthetic_training(self):
        """Treina o modelo com exemplos base para partida imediata."""
        X, y = [], []

        # Amostras de referência aproximadas para cada classe (0 a 8)
        for _ in range(25):
            # 0: Fist (todos dedos fechados)
            feat = np.ones(24) * 0.5 + np.random.normal(0, 0.04, 24)
            X.append(feat); y.append(0)

            # 1: Open Palm (todos dedos esticados)
            feat = np.ones(24) * 1.8 + np.random.normal(0, 0.05, 24)
            X.append(feat); y.append(1)

            # 2: Point (apenas indicador esticado)
            feat = np.ones(24) * 0.5 + np.random.normal(0, 0.04, 24)
            feat[3] = 1.9 + np.random.normal(0, 0.05)  # ponta do indicador
            X.append(feat); y.append(2)

            # 3: Peace (indicador e medio esticados)
            feat = np.ones(24) * 0.5 + np.random.normal(0, 0.04, 24)
            feat[3] = 1.8 + np.random.normal(0, 0.05)
            feat[7] = 1.8 + np.random.normal(0, 0.05)
            X.append(feat); y.append(3)

            # 4: Three (indicador, medio, anelar)
            feat = np.ones(24) * 0.5 + np.random.normal(0, 0.04, 24)
            feat[3] = 1.8; feat[7] = 1.8; feat[11] = 1.8
            X.append(feat); y.append(4)

            # 5: Four (4 dedos sem polegar)
            feat = np.ones(24) * 1.8 + np.random.normal(0, 0.04, 24)
            feat[0] = 0.5  # polegar recolhido
            X.append(feat); y.append(5)

            # 6: Pinch (polegar e indicador colados)
            feat = np.ones(24) * 0.9 + np.random.normal(0, 0.04, 24)
            feat[20] = 0.12 + np.random.normal(0, 0.02)  # tip_dists[0] (4-8)
            X.append(feat); y.append(6)

            # 7: Rock (indicador e minimo)
            feat = np.ones(24) * 0.5 + np.random.normal(0, 0.04, 24)
            feat[3] = 1.8; feat[15] = 1.8
            X.append(feat); y.append(7)

            # 8: Thumbs Up (apenas polegar esticado)
            feat = np.ones(24) * 0.5 + np.random.normal(0, 0.04, 24)
            feat[0] = 1.8  # polegar esticado
            X.append(feat); y.append(8)

        self.clf.fit(np.array(X), np.array(y))
        self.is_trained = True

    def predict(self, hand: HandData) -> Tuple[str, float]:
        """Prevê o gesto usando o classificador de IA e retorna (nome_do_gesto, confiança)."""
        feat = self._extract_features(hand.landmarks_norm).reshape(1, -1)
        pred_idx = int(self.clf.predict(feat)[0])
        probs = self.clf.predict_proba(feat)[0]
        conf = float(np.max(probs))

        # Heurística de refinamento para garantir consistência visual
        # Dedos levantados: [Polegar, Indicador, Médio, Anelar, Mínimo]
        f = hand.fingers_up
        if not any(f):
            return "Fist (Fechado)", 0.95
        if all(f):
            return "Open Palm (Aberta)", 0.95
        if f[1] and not f[2] and not f[3] and not f[4]:
            return "Point (Apontando)", 0.92
        if f[1] and f[2] and not f[3] and not f[4]:
            return "Peace (Paz/Dois)", 0.92
        if f[1] and f[2] and f[3] and not f[4]:
            return "Three (Três)", 0.90
        if not f[0] and f[1] and f[2] and f[3] and f[4]:
            return "Four (Quatro)", 0.90
        if hand.pinch_distance < 0.07:
            return "Pinch (Pinça)", 0.94
        if f[0] and f[1] and not f[2] and not f[3] and f[4]:
            return "Rock (Chifre)", 0.88

        name = self.GESTURE_LABELS[pred_idx] if pred_idx < len(self.GESTURE_LABELS) else "Gesto"
        return name, conf


class HolographicVirtualKeyboard:
    """Teclas virtuais flutuantes no espaço 3D/2D do ar (Laser Harp flutuante)."""

    def __init__(self, scale_name: str = "Dó Maior (C Major)"):
        self.scale_name = scale_name
        self.notes = SCALES.get(scale_name, SCALES["Dó Maior (C Major)"])
        self.active_note: Optional[int] = None
        self.last_trigger_time = 0.0

    def set_scale(self, scale_name: str):
        if scale_name in SCALES:
            self.scale_name = scale_name
            self.notes = SCALES[scale_name]

    def get_zones(self, frame_w: int, frame_h: int) -> List[dict]:
        """Calcula as zonas retangulares/esféricas das notas no topo/meio da tela."""
        n = len(self.notes)
        margin_x = int(frame_w * 0.08)
        width_avail = frame_w - (2 * margin_x)
        key_w = width_avail // n
        key_h = int(frame_h * 0.22)
        key_y = int(frame_h * 0.15)

        zones = []
        for i, note in enumerate(self.notes):
            kx = margin_x + i * key_w
            zones.append({
                "note": note,
                "name": get_note_name(note),
                "rect": (kx, key_y, key_w - 8, key_h),
                "center": (kx + (key_w - 8) // 2, key_y + key_h // 2),
                "radius": min(key_w - 8, key_h) // 2,
            })
        return zones

    def check_hit(self, hand: HandData, frame_w: int, frame_h: int) -> Optional[int]:
        """Verifica se a ponta do dedo indicador (ou pinça) está dentro de uma das zonas."""
        # Ponto 8 é a ponta do indicador
        tip_x, tip_y = hand.landmarks_px[8]

        zones = self.get_zones(frame_w, frame_h)
        for zone in zones:
            x, y, w, h = zone["rect"]
            if x <= tip_x <= x + w and y <= tip_y <= y + h:
                # Dispara se o indicador estiver levantado ou se estiver fazendo pinça
                if hand.fingers_up[1] or hand.pinch_distance < 0.08:
                    return zone["note"]
        return None

import math
import time
from typing import List, Optional
import cv2
import numpy as np
from gestures import HolographicVirtualKeyboard, MLGestureClassifier, get_note_name
from tracker import HandData
from tutor import IntelligentMusicTutor


# Cores BGR Neon
COLOR_CYAN = (255, 229, 0)       # #00e5ff
COLOR_MAGENTA = (249, 0, 213)    # #d500f9
COLOR_GREEN = (118, 230, 0)      # #00e676
COLOR_AMBER = (0, 191, 255)      # #ffbf00
COLOR_RED = (80, 80, 255)        # #ff5050
COLOR_DARK_BG = (20, 24, 30)
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (120, 130, 140)


class HolographicHUD:
    """Renderizador do HUD de Realidade Aumentada estilo Cyberpunk / Holográfico."""

    def __init__(self):
        self.pulse_phase = 0.0

    def draw(
        self,
        frame: np.ndarray,
        hands: List[HandData],
        keyboard: HolographicVirtualKeyboard,
        classifier: MLGestureClassifier,
        tutor: IntelligentMusicTutor,
        active_note: Optional[int],
        engine_online: bool,
        current_waveform: str,
        fps: float,
        cutoff: float,
        pitch_bend: float,
    ) -> np.ndarray:
        h, w, _ = frame.shape
        self.pulse_phase = (self.pulse_phase + 0.1) % (2 * math.pi)
        pulse = 0.5 + 0.5 * math.sin(self.pulse_phase)

        # 1. Desenha as zonas das teclas virtuais no ar
        self._draw_keyboard_zones(frame, keyboard, tutor, active_note, pulse)

        # 2. Desenha o esqueleto e retículas das mãos
        self._draw_hands_hud(frame, hands, classifier, cutoff, pitch_bend, pulse)

        # 3. Desenha os medidores de modulação da Mão Esquerda (Cutoff e Pitch Bend)
        self._draw_modulation_gauges(frame, cutoff, pitch_bend, w, h)

        # 4. Desenha o Painel do Modo Tutor Inteligente
        self._draw_tutor_panel(frame, tutor, w, h)

        # 5. Header superior e rodapé com atalhos
        self._draw_header_footer(frame, engine_online, current_waveform, keyboard.scale_name, fps, w, h)

        return frame

    def _draw_keyboard_zones(
        self,
        frame: np.ndarray,
        keyboard: HolographicVirtualKeyboard,
        tutor: IntelligentMusicTutor,
        active_note: Optional[int],
        pulse: float,
    ):
        h, w, _ = frame.shape
        zones = keyboard.get_zones(w, h)
        expected_note = tutor.get_expected_note().note if tutor.is_active and tutor.get_expected_note() else None

        # Cria overlay semi-transparente para as teclas
        overlay = frame.copy()

        for zone in zones:
            zx, zy, zw, zh = zone["rect"]
            note = zone["note"]
            name = zone["name"]
            is_active = (active_note == note)
            is_target = (expected_note == note)

            if is_active:
                # Tecla tocada: Brilho forte ciano/verde
                bg_color = (180, 255, 100)
                border_color = COLOR_GREEN
                border_thickness = 3
                alpha = 0.45
            elif is_target:
                # Tecla esperada pelo Tutor: Pulsação âmbar/dourada
                bg_color = (0, 150, 255)
                border_color = COLOR_AMBER
                border_thickness = int(2 + pulse * 2)
                alpha = 0.35 + (pulse * 0.15)
            else:
                # Tecla em repouso
                bg_color = (40, 50, 60)
                border_color = (140, 160, 180)
                border_thickness = 1
                alpha = 0.20

            # Desenha retângulo com cantos arredondados
            cv2.rectangle(overlay, (zx, zy), (zx + zw, zy + zh), bg_color, -1)
            cv2.rectangle(frame, (zx, zy), (zx + zw, zy + zh), border_color, border_thickness)

            # Rótulo da nota
            text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
            tx = zx + (zw - text_size[0]) // 2
            ty = zy + (zh + text_size[1]) // 2

            cv2.putText(frame, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_WHITE, 2, cv2.LINE_AA)

            # Se for o alvo do tutor, adiciona marcação visual
            if is_target:
                cv2.circle(frame, (zx + zw // 2, zy + 15), int(4 + pulse * 3), COLOR_AMBER, -1)

        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    def _draw_hands_hud(
        self,
        frame: np.ndarray,
        hands: List[HandData],
        classifier: MLGestureClassifier,
        cutoff: float,
        pitch_bend: float,
        pulse: float,
    ):
        for hand in hands:
            pts = hand.landmarks_px

            # Conexões do esqueleto da mão
            connections = [
                # Polegar
                (0, 1), (1, 2), (2, 3), (3, 4),
                # Indicador
                (0, 5), (5, 6), (6, 7), (7, 8),
                # Médio
                (0, 9), (9, 10), (10, 11), (11, 12),
                # Anelar
                (0, 13), (13, 14), (14, 15), (15, 16),
                # Mínimo
                (0, 17), (17, 18), (18, 19), (19, 20),
                # Base da palma
                (5, 9), (9, 13), (13, 17),
            ]

            color_skel = COLOR_CYAN if hand.label == "Right" else COLOR_MAGENTA

            # Desenha as linhas neon
            for p1, p2 in connections:
                cv2.line(frame, pts[p1], pts[p2], color_skel, 2, cv2.LINE_AA)

            # Desenha nós
            for i, p in enumerate(pts):
                r = 4
                if i in [4, 8, 12, 16, 20]:
                    r = 6  # Pontas dos dedos maiores
                cv2.circle(frame, p, r, (255, 255, 255), -1)
                cv2.circle(frame, p, r + 2, color_skel, 1, cv2.LINE_AA)

            # Retícula holográfica no indicador (ponto 8)
            tip8 = pts[8]
            reticle_r = int(14 + pulse * 4)
            cv2.circle(frame, tip8, reticle_r, color_skel, 1, cv2.LINE_AA)
            cv2.line(frame, (tip8[0] - reticle_r - 4, tip8[1]), (tip8[0] + reticle_r + 4, tip8[1]), color_skel, 1)
            cv2.line(frame, (tip8[0], tip8[1] - reticle_r - 4), (tip8[0], tip8[1] + reticle_r + 4), color_skel, 1)

            # Classificação por Machine Learning da postura da mão
            gesture_name, conf = classifier.predict(hand)

            # Rótulo de status flutuante acima do pulso
            wrist_x, wrist_y = pts[0]
            label_y = max(wrist_y + 35, 30)

            if hand.label == "Right":
                tag = f"DIR [MELODIA] | IA: {gesture_name} ({int(conf*100)}%)"
                tag_color = COLOR_CYAN
            else:
                tag = f"ESQ [MODULACAO] | Filtro: {int(cutoff*100)}% | Bend: {pitch_bend:+.1f}st"
                tag_color = COLOR_MAGENTA

            cv2.putText(
                frame, tag, (wrist_x - 80, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_DARK_BG, 3, cv2.LINE_AA
            )
            cv2.putText(
                frame, tag, (wrist_x - 80, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, tag_color, 1, cv2.LINE_AA
            )

    def _draw_modulation_gauges(self, frame: np.ndarray, cutoff: float, pitch_bend: float, w: int, h: int):
        # 1. Radar Circular de Cutoff (canto inferior esquerdo)
        radar_cx, radar_cy = 90, h - 90
        radar_max_r = 50
        radar_r = int(10 + cutoff * (radar_max_r - 10))

        # Círculo base do radar
        cv2.circle(frame, (radar_cx, radar_cy), radar_max_r, (60, 60, 80), 1, cv2.LINE_AA)
        cv2.circle(frame, (radar_cx, radar_cy), radar_max_r // 2, (50, 50, 70), 1, cv2.LINE_AA)
        # Círculo dinâmico da modulação
        cv2.circle(frame, (radar_cx, radar_cy), radar_r, COLOR_MAGENTA, 2, cv2.LINE_AA)
        cv2.putText(
            frame, f"CUTOFF: {int(cutoff*100)}%",
            (radar_cx - 45, radar_cy + radar_max_r + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_MAGENTA, 1, cv2.LINE_AA
        )

        # 2. Barra Vertical de Pitch Bend (canto lateral direito)
        bar_x = w - 45
        bar_y_top = int(h * 0.45)
        bar_h = int(h * 0.35)
        bar_mid_y = bar_y_top + bar_h // 2

        # Fundo da barra
        cv2.rectangle(frame, (bar_x, bar_y_top), (bar_x + 12, bar_y_top + bar_h), (50, 60, 70), -1)
        cv2.line(frame, (bar_x - 5, bar_mid_y), (bar_x + 17, bar_mid_y), COLOR_WHITE, 1)

        # Indicador de Pitch Bend (-2 a +2 semitons)
        clamped_pb = np.clip(pitch_bend, -2.0, 2.0)
        indicator_y = int(bar_mid_y - (clamped_pb / 2.0) * (bar_h // 2))
        cv2.circle(frame, (bar_x + 6, indicator_y), 7, COLOR_AMBER, -1)
        cv2.putText(
            frame, f"{pitch_bend:+.1f}st",
            (bar_x - 45, indicator_y + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_AMBER, 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, "PITCH BEND",
            (bar_x - 65, bar_y_top - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_WHITE, 1, cv2.LINE_AA
        )

    def _draw_tutor_panel(self, frame: np.ndarray, tutor: IntelligentMusicTutor, w: int, h: int):
        panel_y = int(h * 0.06)
        panel_h = 36

        # Fundo glass do tutor
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, panel_y), (w - 20, panel_y + panel_h), COLOR_DARK_BG, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Cor do status
        if tutor.feedback_type == "success":
            status_color = COLOR_GREEN
        elif tutor.feedback_type == "warning":
            status_color = COLOR_AMBER
        else:
            status_color = COLOR_CYAN

        cv2.rectangle(frame, (20, panel_y), (w - 20, panel_y + panel_h), status_color, 1)

        mode_tag = f"[TUTOR: {tutor.song.title}]" if tutor.is_active else "[MODO LIVRE]"
        accuracy_tag = f"Precisao: {tutor.get_accuracy()}%" if tutor.is_active else ""

        full_msg = f"{mode_tag} {tutor.feedback_message}  {accuracy_tag}"
        cv2.putText(
            frame, full_msg, (32, panel_y + 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 1, cv2.LINE_AA
        )

    def _draw_header_footer(
        self,
        frame: np.ndarray,
        engine_online: bool,
        current_waveform: str,
        scale_name: str,
        fps: float,
        w: int,
        h: int,
    ):
        # Header superior
        header_text = "GESTURES MUSIC AI // RUST DSP + OPENCV"
        cv2.putText(frame, header_text, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_WHITE, 2, cv2.LINE_AA)

        # Status do motor Rust
        eng_status = "RUST ENGINE: ONLINE" if engine_online else "RUST ENGINE: OFFLINE"
        eng_color = COLOR_GREEN if engine_online else COLOR_RED
        cv2.putText(frame, eng_status, (w - 250, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, eng_color, 1, cv2.LINE_AA)

        # FPS e Configuração
        info_text = f"FPS: {fps:.0f} | ONDA: {current_waveform.upper()} | {scale_name}"
        cv2.putText(frame, info_text, (w - 380, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_GRAY, 1, cv2.LINE_AA)

        # Rodapé com comandos
        footer = "[ESPACO] Modo Tutor | [T] Trocar Musica | [W] Onda (Saw/Sine/Square) | [E] Escala | [Q] Sair"
        cv2.putText(frame, footer, (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1, cv2.LINE_AA)

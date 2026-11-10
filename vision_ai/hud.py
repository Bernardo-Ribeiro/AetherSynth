import math
import time
from typing import List, Optional, Tuple
import cv2
import numpy as np
from gestures import HolographicVirtualKeyboard, MLGestureClassifier, get_note_name
from tracker import HandData
from tutor import IntelligentMusicTutor


# =====================================================================
# PALETA DE CORES DE ALTO GOSTO (TASTE / OBSIDIAN & ELECTRIC ICE)
# Cores em formato BGR do OpenCV correspondentes ao design do logo oficial
# =====================================================================
COLOR_BG_DARK       = (14, 11, 8)       # #080B0E — Obsidiana Profundo
COLOR_GLASS_FILL    = (28, 20, 15)      # #0F141C — Vidro Escuro Translúcido
COLOR_GLASS_ACTIVE  = (50, 35, 20)      # #142332 — Vidro Iluminado
COLOR_GLASS_BORDER  = (65, 48, 35)      # #233041 — Borda Suave de Vidro (1px)
COLOR_ICE_BLUE      = (248, 189, 56)    # #38BDF8 — Azul Gelo Elétrico (Marca AetherSynth)
COLOR_ICE_LIGHT     = (255, 224, 130)   # #82E0FF — Destaque de Crista da Onda
COLOR_INDIGO        = (248, 140, 129)   # #818CF8 — Índigo Refinado
COLOR_EMERALD       = (153, 211, 52)    # #34D399 — Verde Esmeralda (Status Ativo)
COLOR_AMBER         = (56, 189, 248)    # #F8BD38 — Âmbar Suave (Alvo do Tutor)
COLOR_RED           = (80, 80, 245)     # #F55050 — Vermelho Erro
COLOR_WHITE         = (255, 255, 255)   # #FFFFFF — Branco Puro
COLOR_SLATE_LIGHT   = (226, 232, 240)   # #E2E8F0 — Texto Claro Legível
COLOR_MUTED         = (184, 163, 148)   # #94A3B8 — Cinza Ardósia Textos Secundários
COLOR_MUTED_DARK    = (120, 100, 85)    # #556478 — Guias e Trilhas


def draw_text_with_shadow(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font_scale: float = 0.5,
    color: Tuple[int, int, int] = COLOR_WHITE,
    thickness: int = 1,
    shadow_color: Tuple[int, int, int] = (0, 0, 0),
    shadow_offset: int = 1,
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
):
    """Renderiza texto com drop shadow de 1px garantindo legibilidade absoluta em qualquer iluminação de câmera."""
    x, y = pos
    if shadow_offset > 0:
        cv2.putText(
            img, text, (x + shadow_offset, y + shadow_offset),
            font_face, font_scale, shadow_color, thickness + 1, cv2.LINE_AA
        )
    cv2.putText(
        img, text, (x, y),
        font_face, font_scale, color, thickness, cv2.LINE_AA
    )


def draw_glass_card(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    bg_color: Tuple[int, int, int] = COLOR_GLASS_FILL,
    border_color: Optional[Tuple[int, int, int]] = COLOR_GLASS_BORDER,
    alpha: float = 0.68,
    radius: int = 12,
):
    """Renderiza um card com acabamento Liquid Glass e cantos arredondados sem artefatos ou cruzamentos."""
    x1, y1 = pt1
    x2, y2 = pt2
    if x2 <= x1 or y2 <= y1:
        return

    ih, iw, _ = img.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(iw, x2), min(ih, y2)
    rw, rh = x2 - x1, y2 - y1
    if rw <= 0 or rh <= 0:
        return

    roi = img[y1:y2, x1:x2]
    overlay = roi.copy()
    r = min(radius, rw // 2, rh // 2)

    # 1. Preenchimento arredondado no overlay
    if r > 0:
        cv2.rectangle(overlay, (r, 0), (rw - r, rh), bg_color, -1)
        cv2.rectangle(overlay, (0, r), (rw, rh - r), bg_color, -1)
        cv2.circle(overlay, (r, r), r, bg_color, -1)
        cv2.circle(overlay, (rw - r, r), r, bg_color, -1)
        cv2.circle(overlay, (r, rh - r), r, bg_color, -1)
        cv2.circle(overlay, (rw - r, rh - r), r, bg_color, -1)
    else:
        cv2.rectangle(overlay, (0, 0), (rw, rh), bg_color, -1)

    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)

    # 2. Borda de vidro limpa (4 linhas retas e 4 arcos de 90° sem círculos internos ou linhas transversais)
    if border_color is not None and r > 0:
        # Segmentos retos
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), border_color, 1, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), border_color, 1, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), border_color, 1, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), border_color, 1, cv2.LINE_AA)
        # Cantos arredondados (arcos de 90 graus)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 0, 180, 270, border_color, 1, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 0, 270, 360, border_color, 1, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, border_color, 1, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 0, 90, 180, border_color, 1, cv2.LINE_AA)
    elif border_color is not None:
        cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 1, cv2.LINE_AA)


class HolographicHUD:
    """HUD de Realidade Aumentada de alto bom gosto (Taste-Driven) para o AetherSynth."""

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
        self.pulse_phase = (self.pulse_phase + 0.08) % (2 * math.pi)
        pulse = 0.5 + 0.5 * math.sin(self.pulse_phase)

        # 1. Pads Acústicos Flutuantes (Teclas virtuais)
        self._draw_acoustic_pads(frame, keyboard, tutor, active_note, pulse)

        # 2. Rastreamento Minimalista das Mãos
        self._draw_hands_minimal(frame, hands, classifier, cutoff, pitch_bend, pulse)

        # 3. Widget de Cutoff (Filtro Mão Esquerda)
        self._draw_cutoff_widget(frame, cutoff, h)

        # 4. Widget de Pitch Bend (Fader Mão Esquerda)
        self._draw_pitch_bend_widget(frame, pitch_bend, w, h)

        # 5. Barra do Tutor Inteligente
        self._draw_tutor_island(frame, tutor, pulse, w)

        # 6. Header Superior Integrado e Footer Dock
        self._draw_header_island(frame, engine_online, current_waveform, keyboard.scale_name, fps, w)
        self._draw_footer_dock(frame, w, h)

        return frame

    def _draw_acoustic_pads(
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

        for zone in zones:
            zx, zy, zw, zh = zone["rect"]
            note = zone["note"]
            name = zone["name"]
            is_active = (active_note == note)
            is_target = (expected_note == note)

            if is_active:
                # Pad Tocado: Iluminação Ice-Blue pura com pulso
                pad_bg = (95, 65, 28)
                pad_border = COLOR_ICE_BLUE
                pad_alpha = 0.85
                text_color = COLOR_WHITE
                octave_color = COLOR_ICE_LIGHT
            elif is_target:
                # Alvo do Tutor: Pulsação âmbar/dourada de alta visibilidade
                pad_bg = (30, 50, 70)
                pad_border = COLOR_AMBER
                pad_alpha = 0.72 + (pulse * 0.15)
                text_color = COLOR_WHITE
                octave_color = COLOR_AMBER
            else:
                # Pad em repouso: Vidro escuro translúcido com alto contraste
                pad_bg = COLOR_GLASS_FILL
                pad_border = (80, 60, 45)
                pad_alpha = 0.68
                text_color = COLOR_WHITE
                octave_color = COLOR_ICE_BLUE

            # Desenha o pad de vidro com cantos arredondados
            draw_glass_card(frame, (zx, zy), (zx + zw, zy + zh), pad_bg, pad_border, pad_alpha, radius=14)

            # Se estiver ativo, desenha linha de brilho na borda superior
            if is_active:
                cv2.line(frame, (zx + 14, zy + 1), (zx + zw - 14, zy + 1), COLOR_ICE_LIGHT, 2, cv2.LINE_AA)

            # Se for o alvo do tutor, desenha ponto de foco luminoso
            if is_target:
                cv2.circle(frame, (zx + zw // 2, zy + 16), int(4 + pulse * 2.5), COLOR_AMBER, -1, cv2.LINE_AA)

            # Tipografia refinada do nome da nota (ex: "Dó4") com sombra de alto contraste
            base_name = name[:-1] if name[-1].isdigit() else name
            octave = name[-1] if name[-1].isdigit() else ""

            text_size = cv2.getTextSize(base_name, cv2.FONT_HERSHEY_SIMPLEX, 0.80, 2)[0]
            tx = zx + (zw - text_size[0]) // 2 - 4
            ty = zy + (zh // 2) + 7

            draw_text_with_shadow(
                frame, base_name, (tx, ty),
                font_scale=0.80, color=text_color, thickness=2
            )
            if octave:
                draw_text_with_shadow(
                    frame, octave, (tx + text_size[0] + 4, ty - 10),
                    font_scale=0.48, color=octave_color, thickness=1
                )

    def _draw_hands_minimal(
        self,
        frame: np.ndarray,
        hands: List[HandData],
        classifier: MLGestureClassifier,
        cutoff: float,
        pitch_bend: float,
        pulse: float,
    ):
        """Esqueleto minimalista e refinado das mãos (linhas finas, nós discretos, retícula precisa)."""
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17),
        ]

        for hand in hands:
            pts = hand.landmarks_px
            is_right = (hand.label == "Right")
            line_color = COLOR_ICE_BLUE if is_right else COLOR_INDIGO

            # Conexões elegantes (espessura 1.5px suave)
            for p1, p2 in connections:
                cv2.line(frame, pts[p1], pts[p2], line_color, 1, cv2.LINE_AA)

            # Nós dos dedos discretos
            for i, p in enumerate(pts):
                if i in [4, 8, 12, 16, 20]:
                    cv2.circle(frame, p, 4, COLOR_WHITE, -1, cv2.LINE_AA)
                    cv2.circle(frame, p, 6, line_color, 1, cv2.LINE_AA)
                else:
                    cv2.circle(frame, p, 2, line_color, -1, cv2.LINE_AA)

            # Retícula de mira precisa na ponta do indicador (dedo 8)
            tip8 = pts[8]
            reticle_r = int(12 + pulse * 3)
            cv2.circle(frame, tip8, reticle_r, COLOR_ICE_BLUE if is_right else COLOR_INDIGO, 1, cv2.LINE_AA)
            cv2.circle(frame, tip8, 2, COLOR_WHITE, -1, cv2.LINE_AA)

            # Tag informativa em card de vidro flutuante de alta visibilidade
            gesture_name, conf = classifier.predict(hand)
            wrist_x, wrist_y = pts[0]
            tag_y = max(wrist_y + 30, 42)

            if is_right:
                tag_text = f"MELODIA  |  {gesture_name.split(' ')[0]}"
                badge_w = 180
            else:
                tag_text = f"MODULACAO  |  CUTOFF {int(cutoff*100)}%"
                badge_w = 205

            draw_glass_card(
                frame,
                (wrist_x - badge_w // 2, tag_y),
                (wrist_x + badge_w // 2, tag_y + 26),
                bg_color=COLOR_BG_DARK,
                border_color=line_color,
                alpha=0.82,
                radius=8,
            )
            draw_text_with_shadow(
                frame, tag_text, (wrist_x - badge_w // 2 + 12, tag_y + 18),
                font_scale=0.44, color=COLOR_WHITE, thickness=1
            )

    def _draw_cutoff_widget(self, frame: np.ndarray, cutoff: float, h: int):
        """Widget circular refinado de Cutoff no canto inferior esquerdo."""
        card_x1, card_y1 = 30, h - 180
        card_x2, card_y2 = 180, h - 52

        draw_glass_card(frame, (card_x1, card_y1), (card_x2, card_y2), COLOR_BG_DARK, COLOR_GLASS_BORDER, alpha=0.75, radius=14)

        # Título nítido
        draw_text_with_shadow(
            frame, "FILTRO CUTOFF", (card_x1 + 16, card_y1 + 22),
            font_scale=0.42, color=COLOR_SLATE_LIGHT, thickness=1
        )

        # Dial Circular
        cx, cy = (card_x1 + card_x2) // 2, card_y1 + 72
        radius = 32

        # Trilha de fundo
        cv2.circle(frame, (cx, cy), radius, COLOR_MUTED_DARK, 2, cv2.LINE_AA)

        # Arco ativo em Azul Gelo
        start_angle = 135
        span_angle = int(cutoff * 270)
        end_angle = start_angle + span_angle

        for deg in range(start_angle, end_angle, 6):
            rad = math.radians(deg)
            px = int(cx + radius * math.cos(rad))
            py = int(cy + radius * math.sin(rad))
            cv2.circle(frame, (px, py), 2, COLOR_ICE_BLUE, -1, cv2.LINE_AA)

        # Valor percentual central em destaque
        val_str = f"{int(cutoff * 100)}%"
        t_sz = cv2.getTextSize(val_str, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)[0]
        draw_text_with_shadow(
            frame, val_str, (cx - t_sz[0] // 2, cy + 7),
            font_scale=0.58, color=COLOR_WHITE, thickness=2
        )

        # Subtítulo legível
        draw_text_with_shadow(
            frame, "MAO ESQUERDA", (card_x1 + 18, card_y2 - 12),
            font_scale=0.36, color=COLOR_ICE_LIGHT, thickness=1
        )

    def _draw_pitch_bend_widget(self, frame: np.ndarray, pitch_bend: float, w: int, h: int):
        """Slider vertical minimalista de Pitch Bend no lado direito."""
        card_x1 = w - 75
        card_x2 = w - 25
        card_y1 = int(h * 0.38)
        card_y2 = int(h * 0.78)

        draw_glass_card(frame, (card_x1, card_y1), (card_x2, card_y2), COLOR_BG_DARK, COLOR_GLASS_BORDER, alpha=0.75, radius=12)

        # Label no topo
        draw_text_with_shadow(
            frame, "BEND", (card_x1 + 8, card_y1 - 10),
            font_scale=0.42, color=COLOR_SLATE_LIGHT, thickness=1
        )

        # Trilha central
        mid_x = (card_x1 + card_x2) // 2
        track_top = card_y1 + 18
        track_bot = card_y2 - 18
        mid_y = (track_top + track_bot) // 2

        cv2.line(frame, (mid_x, track_top), (mid_x, track_bot), COLOR_MUTED_DARK, 2, cv2.LINE_AA)
        cv2.line(frame, (mid_x - 8, mid_y), (mid_x + 8, mid_y), COLOR_ICE_BLUE, 1, cv2.LINE_AA)  # Marca neutra (0)

        # Knob Fader
        clamped_pb = np.clip(pitch_bend, -2.0, 2.0)
        knob_y = int(mid_y - (clamped_pb / 2.0) * (mid_y - track_top))

        is_bent = abs(clamped_pb) > 0.08
        knob_color = COLOR_ICE_BLUE if is_bent else COLOR_WHITE
        cv2.circle(frame, (mid_x, knob_y), 7, knob_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (mid_x, knob_y), 9, COLOR_ICE_LIGHT if is_bent else COLOR_SLATE_LIGHT, 1, cv2.LINE_AA)

        # Valor formatado de forma limpa (sem "-0.0")
        if abs(clamped_pb) < 0.05:
            val_str = "0.0"
        else:
            val_str = f"{clamped_pb:+.1f}"

        draw_text_with_shadow(
            frame, val_str, (card_x1 - 42, knob_y + 5),
            font_scale=0.42, color=knob_color, thickness=1
        )

    def _draw_tutor_island(self, frame: np.ndarray, tutor: IntelligentMusicTutor, pulse: float, w: int):
        """Ilha de feedback pedagógico e status do Tutor."""
        card_h = 36
        card_y = 66

        if tutor.is_active:
            # Modo Tutor Ativo: Card centralizado de destaque com acento Ice-Blue
            card_w = min(int(w * 0.74), 880)
            card_x = (w - card_w) // 2

            expected = tutor.get_expected_note()
            note_target = expected.name if expected else "Fim"
            status_text = f"TUTOR: {tutor.song.title.upper()}  |  TOQUE: {note_target}  |  {tutor.feedback_message}"

            draw_glass_card(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (24, 18, 12), COLOR_ICE_BLUE, alpha=0.82, radius=10)

            # Ponto luminoso de gravação/tutor
            cv2.circle(frame, (card_x + 18, card_y + 18), int(4 + pulse * 2.5), COLOR_ICE_BLUE, -1, cv2.LINE_AA)
            draw_text_with_shadow(
                frame, status_text, (card_x + 36, card_y + 23),
                font_scale=0.46, color=COLOR_WHITE, thickness=1
            )

            # Badge de Precisão no lado direito
            acc_str = f"{tutor.get_accuracy()}%"
            draw_text_with_shadow(
                frame, acc_str, (card_x + card_w - 60, card_y + 23),
                font_scale=0.46, color=COLOR_EMERALD, thickness=2
            )
        else:
            # Modo Livre: Tag translúcida nítida e convidativa
            card_w = 440
            card_x = (w - card_w) // 2
            hint = "MODO LIVRE  |  Pressione [ESPACO] para Tutor"

            draw_glass_card(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), COLOR_BG_DARK, COLOR_GLASS_BORDER, alpha=0.72, radius=8)
            draw_text_with_shadow(
                frame, hint, (card_x + 20, card_y + 23),
                font_scale=0.42, color=COLOR_WHITE, thickness=1
            )

    def _draw_header_island(
        self,
        frame: np.ndarray,
        engine_online: bool,
        current_waveform: str,
        scale_name: str,
        fps: float,
        w: int,
    ):
        """Barra de topo unificada (Floating Header Island) — Alta legibilidade e hierarquia."""
        bar_h = 44
        bar_y = 14
        margin_x = 24
        bar_w = w - (2 * margin_x)

        # Fundo de vidro único para todo o header
        draw_glass_card(frame, (margin_x, bar_y), (margin_x + bar_w, bar_y + bar_h), COLOR_BG_DARK, COLOR_GLASS_BORDER, alpha=0.82, radius=12)

        # 1. Badge de FPS à esquerda com alto contraste
        fps_val = int(round(fps))
        fps_color = COLOR_EMERALD if fps_val >= 28 else (COLOR_AMBER if fps_val >= 20 else COLOR_RED)
        draw_text_with_shadow(
            frame, f"FPS {fps_val}", (margin_x + 18, bar_y + 28),
            font_scale=0.48, color=fps_color, thickness=2
        )

        # Divisor sutil
        cv2.line(frame, (margin_x + 92, bar_y + 10), (margin_x + 92, bar_y + 34), COLOR_GLASS_BORDER, 1)

        # 2. Título da Marca Oficial (AetherSynth)
        draw_text_with_shadow(
            frame, "AETHERSYNTH", (margin_x + 110, bar_y + 28),
            font_scale=0.56, color=COLOR_WHITE, thickness=2
        )
        draw_text_with_shadow(
            frame, "// GESTURAL SYNTH", (margin_x + 256, bar_y + 28),
            font_scale=0.42, color=COLOR_ICE_LIGHT, thickness=1
        )

        # 3. Badges de Configuração à Direita (Sem Sobreposição)
        rx = margin_x + bar_w - 20

        # Status do motor Rust DSP
        eng_color = COLOR_EMERALD if engine_online else COLOR_RED
        eng_text = "RUST DSP"
        eng_sz = cv2.getTextSize(eng_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0]
        rx -= eng_sz[0]
        cv2.circle(frame, (rx - 12, bar_y + 24), 5, eng_color, -1, cv2.LINE_AA)
        draw_text_with_shadow(frame, eng_text, (rx, bar_y + 28), font_scale=0.44, color=COLOR_WHITE, thickness=1)
        rx -= 34

        # Divisor
        cv2.line(frame, (rx, bar_y + 10), (rx, bar_y + 34), COLOR_GLASS_BORDER, 1)
        rx -= 20

        # Forma de onda
        wave_str = f"ONDA: {current_waveform.upper()}"
        wave_sz = cv2.getTextSize(wave_str, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0]
        rx -= wave_sz[0]
        draw_text_with_shadow(frame, wave_str, (rx, bar_y + 28), font_scale=0.44, color=COLOR_ICE_BLUE, thickness=1)
        rx -= 24

        # Divisor
        cv2.line(frame, (rx, bar_y + 10), (rx, bar_y + 34), COLOR_GLASS_BORDER, 1)
        rx -= 20

        # Escala ativa
        scale_str = f"ESCALA: {scale_name.split(' ')[0]}"
        scale_sz = cv2.getTextSize(scale_str, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0]
        rx -= scale_sz[0]
        draw_text_with_shadow(frame, scale_str, (rx, bar_y + 28), font_scale=0.44, color=COLOR_WHITE, thickness=1)

    def _draw_footer_dock(self, frame: np.ndarray, w: int, h: int):
        """Dock inferior limpo e centralizado com atalhos de teclado destacados."""
        dock_w = 680
        dock_h = 32
        dock_x = (w - dock_w) // 2
        dock_y = h - 42

        draw_glass_card(frame, (dock_x, dock_y), (dock_x + dock_w, dock_y + dock_h), COLOR_BG_DARK, COLOR_GLASS_BORDER, alpha=0.75, radius=8)

        # Segmentos coloridos: teclas em Azul Gelo (#38BDF8) e ações em Branco (#FFFFFF)
        segments = [
            ("[ESPACO]", COLOR_ICE_BLUE), (" Tutor    ", COLOR_WHITE),
            ("[T]", COLOR_ICE_BLUE), (" Musica    ", COLOR_WHITE),
            ("[W]", COLOR_ICE_BLUE), (" Onda    ", COLOR_WHITE),
            ("[E]", COLOR_ICE_BLUE), (" Escala    ", COLOR_WHITE),
            ("[Q]", COLOR_ICE_BLUE), (" Sair", COLOR_WHITE),
        ]

        total_width = 0
        sizes = []
        for text, _ in segments:
            sz = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
            sizes.append(sz[0])
            total_width += sz[0]

        curr_x = dock_x + (dock_w - total_width) // 2
        text_y = dock_y + 21

        for (text, col), sz_w in zip(segments, sizes):
            draw_text_with_shadow(
                frame, text, (curr_x, text_y),
                font_scale=0.42, color=col, thickness=1
            )
            curr_x += sz_w

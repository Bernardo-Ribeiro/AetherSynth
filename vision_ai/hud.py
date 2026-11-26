import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from gestures import (
    ChordDetection,
    HolographicVirtualKeyboard,
    MLPChordClassifier,
    SHAPE_CLASSES,
)
from harmony_ai import DetectedKey
from tracker import HandData
from tutor import ChordProgressionGuide


# Paleta Obsidian & Electric Ice (BGR)
COLOR_BG_DARK = (14, 11, 8)
COLOR_BG_MENU = (22, 16, 12)
COLOR_GLASS_FILL = (28, 20, 15)
COLOR_GLASS_ACTIVE = (62, 42, 22)
COLOR_GLASS_BORDER = (72, 54, 40)
COLOR_ICE_BLUE = (248, 189, 56)     # #38BDF8
COLOR_ICE_LIGHT = (255, 224, 130)   # #82E0FF
COLOR_INDIGO = (248, 140, 129)      # #818CF8
COLOR_EMERALD = (153, 211, 52)      # #34D399
COLOR_AMBER = (56, 189, 248)        # #F8BD38
COLOR_RED = (80, 80, 245)           # #F55050
COLOR_WHITE = (255, 255, 255)
COLOR_SLATE_LIGHT = (226, 232, 240)
COLOR_MUTED = (184, 163, 148)
COLOR_MUTED_DARK = (120, 100, 85)


@dataclass
class UIButton:
    id: str
    rect: Tuple[int, int, int, int]  # (x, y, w, h)
    label: str
    active: bool = False
    accent: Tuple[int, int, int] = COLOR_ICE_BLUE
    subtitle: str = ""


class MouseController:
    """Gerenciador de hover e cliques do mouse para a interface estilo jogo."""

    def __init__(self):
        self.cursor: Tuple[int, int] = (-1, -1)
        self.is_down: bool = False
        self.buttons: List[UIButton] = []
        self._pending_clicks: List[str] = []

    def clear_buttons(self):
        self.buttons.clear()

    def register_button(self, btn: UIButton):
        self.buttons.append(btn)

    def hit_test(self, x: int, y: int) -> Optional[UIButton]:
        for btn in reversed(self.buttons):
            bx, by, bw, bh = btn.rect
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return btn
        return None

    def is_hovered(self, btn_id: str) -> bool:
        x, y = self.cursor
        hit = self.hit_test(x, y)
        return hit is not None and hit.id == btn_id

    def on_mouse(self, event: int, x: int, y: int, flags: int, param=None):
        self.cursor = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_down = True
            hit = self.hit_test(x, y)
            if hit is not None:
                self._pending_clicks.append(hit.id)
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_down = False

    def pop_clicks(self) -> List[str]:
        if not self._pending_clicks:
            return []
        out = list(self._pending_clicks)
        self._pending_clicks.clear()
        return out


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
    x, y = pos
    if shadow_offset > 0:
        cv2.putText(
            img, text, (x + shadow_offset, y + shadow_offset),
            font_face, font_scale, shadow_color, thickness + 1, cv2.LINE_AA
        )
    cv2.putText(img, text, (x, y), font_face, font_scale, color, thickness, cv2.LINE_AA)


def draw_glass_card(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    bg_color: Tuple[int, int, int] = COLOR_GLASS_FILL,
    border_color: Optional[Tuple[int, int, int]] = COLOR_GLASS_BORDER,
    alpha: float = 0.70,
    radius: int = 12,
    border_thickness: int = 1,
):
    x1, y1 = pt1
    x2, y2 = pt2
    if x2 <= x1 or y2 <= y1:
        return

    ih, iw, _ = img.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(iw, x2), min(ih, y2)
    rw, rh = x2 - x1, y2 - y1
    if rw <= 2 or rh <= 2:
        return

    roi = img[y1:y2, x1:x2]
    r = min(radius, rw // 2, rh // 2)

    if r <= 0:
        bg = np.empty_like(roi)
        bg[:] = bg_color
        cv2.addWeighted(bg, alpha, roi, 1.0 - alpha, 0, roi)
        if border_color is not None:
            cv2.rectangle(img, (x1, y1), (x2, y2), border_color, border_thickness, cv2.LINE_AA)
        return

    overlay = roi.copy()
    cv2.rectangle(overlay, (r, 0), (rw - r, rh), bg_color, -1)
    cv2.rectangle(overlay, (0, r), (rw, rh - r), bg_color, -1)
    cv2.circle(overlay, (r, r), r, bg_color, -1)
    cv2.circle(overlay, (rw - r, r), r, bg_color, -1)
    cv2.circle(overlay, (r, rh - r), r, bg_color, -1)
    cv2.circle(overlay, (rw - r, rh - r), r, bg_color, -1)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)

    if border_color is not None:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), border_color, border_thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), border_color, border_thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), border_color, border_thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), border_color, border_thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 0, 180, 270, border_color, border_thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 0, 270, 360, border_color, border_thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, border_color, border_thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 0, 90, 180, border_color, border_thickness, cv2.LINE_AA)


def draw_vector_arrow(
    img: np.ndarray, center: Tuple[int, int], direction: str, color: Tuple[int, int, int], size: int = 12
):
    cx, cy = center
    if direction == "up":
        cv2.arrowedLine(img, (cx, cy + size), (cx, cy - size), color, 2, cv2.LINE_AA, tipLength=0.45)
    elif direction == "down":
        cv2.arrowedLine(img, (cx, cy - size), (cx, cy + size), color, 2, cv2.LINE_AA, tipLength=0.45)
    elif direction == "right":
        cv2.arrowedLine(img, (cx - size, cy), (cx + size, cy), color, 2, cv2.LINE_AA, tipLength=0.45)
    elif direction == "left":
        cv2.arrowedLine(img, (cx + size, cy), (cx - size, cy), color, 2, cv2.LINE_AA, tipLength=0.45)


def draw_hand_sign_icon(
    img: np.ndarray,
    center: Tuple[int, int],
    size: int,
    shape_idx: int,
    orientation: str = "up",
    accent: Tuple[int, int, int] = COLOR_ICE_BLUE,
    success: bool = False,
):
    """Desenha icone vetorial nitido mostrando como posicionar dedos e orientacao da mao."""
    cx, cy = center
    active_col = COLOR_EMERALD if success else accent
    curled_col = (95, 80, 68)

    # Mascote da postura: [Indicador, Medio, Anelar, Minimo] + modo do polegar
    finger_table = {
        0: ([False, False, False, False], "tuck"),   # Punho
        1: ([True, False, False, False], "tuck"),    # Do
        2: ([True, True, False, False], "tuck"),     # Re
        3: ([True, True, True, False], "tuck"),      # Mi
        4: ([True, True, True, True], "tuck"),       # Fa
        5: ([True, True, True, True], "open"),       # Sol
        6: ([False, False, False, True], "open"),    # La (hang-loose)
        7: ([False, False, False, False], "up"),     # Si (polegar)
    }
    ext, thumb_mode = finger_table.get(shape_idx, ([False, False, False, False], "tuck"))

    def xform(u: float, v: float) -> Tuple[int, int]:
        if orientation == "down":
            ru, rv = -u, -v
        elif orientation == "right":
            ru, rv = -v, u
        elif orientation == "left":
            ru, rv = v, -u
        else:
            ru, rv = u, v
        return (int(round(cx + ru * size)), int(round(cy + rv * size)))

    # Pulso e palma
    wrist_poly = np.array([xform(-0.22, 0.50), xform(0.22, 0.50), xform(0.20, 0.78), xform(-0.20, 0.78)], dtype=np.int32)
    palm_poly = np.array(
        [xform(-0.36, 0.04), xform(0.36, 0.04), xform(0.32, 0.54), xform(-0.32, 0.54)],
        dtype=np.int32,
    )
    cv2.fillPoly(img, [wrist_poly], (44, 34, 24), cv2.LINE_AA)
    cv2.fillPoly(img, [palm_poly], (58, 44, 30), cv2.LINE_AA)
    cv2.polylines(img, [palm_poly], True, active_col, 2, cv2.LINE_AA)

    # 4 dedos (Indicador, Medio, Anelar, Minimo)
    x_cols = [-0.26, -0.09, 0.09, 0.26]
    top_vs = [-0.76, -0.84, -0.78, -0.64]
    thick_ext = max(3, int(size * 0.12))
    thick_curled = max(2, int(size * 0.09))

    for is_up, u_x, top_v in zip(ext, x_cols, top_vs):
        p_base = xform(u_x, 0.06)
        if is_up:
            p_tip = xform(u_x, top_v)
            cv2.line(img, p_base, p_tip, active_col, thick_ext, cv2.LINE_AA)
            cv2.circle(img, p_tip, max(2, thick_ext // 2 + 1), COLOR_WHITE, -1, cv2.LINE_AA)
        else:
            p_fold = xform(u_x, -0.10)
            cv2.line(img, p_base, p_fold, curled_col, thick_curled, cv2.LINE_AA)
            cv2.circle(img, p_fold, max(2, thick_curled // 2), (130, 110, 95), -1, cv2.LINE_AA)

    # Polegar
    p_th_base = xform(-0.34, 0.28)
    if thumb_mode == "open":
        p_th_tip = xform(-0.80, -0.20)
        cv2.line(img, p_th_base, p_th_tip, active_col, thick_ext, cv2.LINE_AA)
        cv2.circle(img, p_th_tip, max(2, thick_ext // 2 + 1), COLOR_WHITE, -1, cv2.LINE_AA)
    elif thumb_mode == "up":
        p_th_mid = xform(-0.42, 0.02)
        p_th_tip = xform(-0.42, -0.70)
        cv2.line(img, p_th_base, p_th_mid, active_col, thick_ext, cv2.LINE_AA)
        cv2.line(img, p_th_mid, p_th_tip, active_col, thick_ext, cv2.LINE_AA)
        cv2.circle(img, p_th_tip, max(2, thick_ext // 2 + 1), COLOR_WHITE, -1, cv2.LINE_AA)
    else:
        p_th_in = xform(0.06, 0.22)
        cv2.line(img, p_th_base, p_th_in, curled_col, thick_curled, cv2.LINE_AA)

    if success:
        cv2.circle(img, (cx, cy), int(size * 0.96), COLOR_EMERALD, 2, cv2.LINE_AA)
        bx, by = cx + int(size * 0.62), cy - int(size * 0.62)
        cv2.circle(img, (bx, by), 11, COLOR_EMERALD, -1, cv2.LINE_AA)
        cv2.line(img, (bx - 5, by), (bx - 1, by + 4), COLOR_BG_DARK, 2, cv2.LINE_AA)
        cv2.line(img, (bx - 1, by + 4), (bx + 6, by - 4), COLOR_BG_DARK, 2, cv2.LINE_AA)


class HolographicHUD:
    """Interface Liquid Glass estilo Jogo (100% clicavel via mouse + baixa carga de CPU)."""

    def __init__(self):
        self.pulse_phase = 0.0
        self._menu_bg_cache: Optional[np.ndarray] = None

    def _get_menu_backdrop(self, w: int, h: int, t: float) -> np.ndarray:
        if self._menu_bg_cache is None or self._menu_bg_cache.shape[:2] != (h, w):
            yy, xx = np.mgrid[0:h, 0:w]
            cx, cy = w * 0.65, h * 0.35
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(w, h)
            glow = np.clip(1.0 - dist * 1.4, 0.0, 1.0) ** 2
            base = np.zeros((h, w, 3), dtype=np.uint8)
            base[:, :, 0] = np.clip(18 + glow * 65, 0, 255).astype(np.uint8)
            base[:, :, 1] = np.clip(13 + glow * 35, 0, 255).astype(np.uint8)
            base[:, :, 2] = np.clip(10 + glow * 16, 0, 255).astype(np.uint8)
            for gx in range(0, w, 64):
                cv2.line(base, (gx, 0), (gx, h), (26, 20, 16), 1)
            for gy in range(0, h, 64):
                cv2.line(base, (0, gy), (w, gy), (26, 20, 16), 1)
            self._menu_bg_cache = base

        canvas = self._menu_bg_cache.copy()
        # Onda sonora animada decorativa no fundo
        pts = []
        mid_y = int(h * 0.80)
        for x in range(0, w + 16, 16):
            y = int(mid_y + 24 * math.sin(x * 0.012 + t * 2.2) + 10 * math.cos(x * 0.028 - t * 1.4))
            pts.append((x, y))
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (75, 55, 25), 2, cv2.LINE_AA)
        return canvas

    def _draw_button(
        self,
        frame: np.ndarray,
        mouse: MouseController,
        btn: UIButton,
        font_scale: float = 0.48,
        radius: int = 10,
    ):
        mouse.register_button(btn)
        x, y, bw, bh = btn.rect
        hovered = mouse.is_hovered(btn.id)
        if btn.active:
            bg = (70, 50, 24)
            border = btn.accent
            alpha = 0.88
            thick = 2
        elif hovered:
            bg = (52, 38, 22)
            border = COLOR_ICE_LIGHT
            alpha = 0.86
            thick = 2
        else:
            bg = COLOR_GLASS_FILL
            border = COLOR_GLASS_BORDER if not hovered else btn.accent
            alpha = 0.74
            thick = 1

        draw_glass_card(frame, (x, y), (x + bw, y + bh), bg, border, alpha=alpha, radius=radius, border_thickness=thick)

        if btn.subtitle:
            draw_text_with_shadow(frame, btn.label, (x + 22, y + 32), font_scale=font_scale, color=COLOR_WHITE, thickness=2)
            draw_text_with_shadow(frame, btn.subtitle, (x + 22, y + 54), font_scale=0.38, color=COLOR_MUTED, thickness=1)
        else:
            tsz = cv2.getTextSize(btn.label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            tx = x + max(8, (bw - tsz[0]) // 2)
            ty = y + (bh + tsz[1]) // 2 - 2
            col = btn.accent if btn.active else (COLOR_ICE_LIGHT if hovered else COLOR_WHITE)
            draw_text_with_shadow(frame, btn.label, (tx, ty), font_scale=font_scale, color=col, thickness=1 if not btn.active else 2)

    def draw_main_menu(
        self,
        w: int,
        h: int,
        t: float,
        mouse: MouseController,
        engine_online: bool,
        mlp_val_acc: float,
        current_waveform: str,
    ) -> np.ndarray:
        mouse.clear_buttons()
        frame = self._get_menu_backdrop(w, h, t)

        # Cabecalho de marca
        draw_text_with_shadow(frame, "AETHERSYNTH", (68, 92), font_scale=1.35, color=COLOR_WHITE, thickness=3)
        draw_text_with_shadow(
            frame,
            "STAGE CHORD AI  //  LOOP SESSION  //  RUST LOCK-FREE DSP",
            (72, 126),
            font_scale=0.52,
            color=COLOR_ICE_BLUE,
            thickness=1,
        )

        # Status pills no topo direito
        dsp_col = COLOR_EMERALD if engine_online else COLOR_RED
        dsp_txt = "RUST DSP ONLINE" if engine_online else "RUST DSP OFFLINE"
        draw_glass_card(frame, (w - 430, 50), (w - 64, 112), COLOR_BG_DARK, COLOR_GLASS_BORDER, alpha=0.82, radius=12)
        cv2.circle(frame, (w - 404, 72), 6, dsp_col, -1, cv2.LINE_AA)
        draw_text_with_shadow(frame, dsp_txt, (w - 388, 77), font_scale=0.46, color=COLOR_WHITE, thickness=1)
        draw_text_with_shadow(
            frame,
            f"MLP NEURAL: {mlp_val_acc * 100:.1f}% VAL  |  ONDA: {current_waveform.upper()}",
            (w - 404, 100),
            font_scale=0.38,
            color=COLOR_ICE_LIGHT,
            thickness=1,
        )

        # Coluna esquerda: botoes principais estilo jogo
        menu_items = [
            ("btn_start_loop", "INICIAR LOOP SESSION", "Grave bases por sinais de palco e crie melodias no tom", COLOR_ICE_BLUE, True),
            ("btn_start_challenge", "MODO TREINO (ACORDES)", "Aprenda os sinais com icone guia e feedback visual", COLOR_AMBER, False),
            ("btn_calibrate", "CALIBRAR IA (MINHA MAO)", "Ensine a rede neural com amostras reais da sua camera", COLOR_EMERALD, False),
            ("btn_how_to", "COMO JOGAR & TIMBRE", "Guia visual dos 7 sinais de acordes e escolha de onda", COLOR_INDIGO, False),
            ("btn_quit", "SAIR DO JOGO", "Encerrar AetherSynth", COLOR_RED, False),
        ]
        bx, by, bw, bh, gap = 68, 175, 510, 74, 14
        for idx, (bid, label, sub, acc, highlight) in enumerate(menu_items):
            ry = by + idx * (bh + gap)
            self._draw_button(
                frame,
                mouse,
                UIButton(bid, (bx, ry, bw, bh), label, active=highlight, accent=acc, subtitle=sub),
                font_scale=0.58,
                radius=14,
            )

        # Coluna direita: cartao resumo de Sinais de Palco
        px1, py1, px2, py2 = 630, 175, w - 64, 602
        draw_glass_card(frame, (px1, py1), (px2, py2), COLOR_BG_DARK, COLOR_GLASS_BORDER, alpha=0.80, radius=16)
        draw_text_with_shadow(frame, "SINAIS DE ACORDES (MAO DIREITA)", (px1 + 26, py1 + 36), font_scale=0.54, color=COLOR_ICE_BLUE, thickness=2)

        signs = [
            ("1 dedo (Indicador)", "DO (C)"),
            ("2 dedos (Paz)", "RE (D)"),
            ("3 dedos", "MI (E)"),
            ("4 dedos (sem polegar)", "FA (F)"),
            ("5 dedos abertos", "SOL (G)"),
            ("Hang-loose (Polegar+Min)", "LA (A)"),
            ("Apenas Polegar", "SI (B)"),
            ("Punho fechado", "Silencio"),
        ]
        for i, (desc, chord) in enumerate(signs):
            sy = py1 + 78 + i * 33
            draw_text_with_shadow(frame, chord, (px1 + 28, sy), font_scale=0.48, color=COLOR_WHITE, thickness=2)
            draw_text_with_shadow(frame, desc, (px1 + 150, sy), font_scale=0.44, color=COLOR_SLATE_LIGHT, thickness=1)

        cv2.line(frame, (px1 + 26, py1 + 345), (px2 - 26, py1 + 345), COLOR_GLASS_BORDER, 1)
        draw_text_with_shadow(frame, "ORIENTACAO DA MAO:", (px1 + 28, py1 + 375), font_scale=0.46, color=COLOR_AMBER, thickness=2)
        draw_text_with_shadow(frame, "Cima = Maior   |   Baixo = Menor   |   Dir = #   |   Esq = b", (px1 + 28, py1 + 402), font_scale=0.42, color=COLOR_WHITE, thickness=1)
        draw_text_with_shadow(frame, "MAO ESQUERDA: Pulsar no ar altera BPM / Abrir ajusta Cutoff", (px1 + 28, py2 - 18), font_scale=0.40, color=COLOR_ICE_LIGHT, thickness=1)

        draw_text_with_shadow(frame, "Clique com o mouse em qualquer opcao para comecar (camera ativa apenas no palco).", (68, h - 30), font_scale=0.44, color=COLOR_MUTED, thickness=1)
        return frame

    def draw_how_to_play(
        self,
        w: int,
        h: int,
        t: float,
        mouse: MouseController,
        current_waveform: str,
    ) -> np.ndarray:
        mouse.clear_buttons()
        frame = self._get_menu_backdrop(w, h, t)

        self._draw_button(frame, mouse, UIButton("btn_menu", (40, 26, 200, 42), "< MENU PRINCIPAL", accent=COLOR_ICE_BLUE), font_scale=0.44)
        draw_text_with_shadow(frame, "COMO JOGAR & ESCOLHA DE TIMBRE", (270, 55), font_scale=0.78, color=COLOR_WHITE, thickness=2)

        # Painel de orientacao
        draw_glass_card(frame, (40, 92), (w - 40, 260), COLOR_BG_DARK, COLOR_GLASS_BORDER, alpha=0.82, radius=14)
        draw_text_with_shadow(frame, "1. ORIENTACAO DA MAO DEFINE QUALIDADE E ACIDENTE", (68, 128), font_scale=0.54, color=COLOR_ICE_BLUE, thickness=2)

        orients = [
            ("up", "PARA CIMA", "Acorde Maior (ex: C, G, F)", COLOR_ICE_BLUE),
            ("down", "PARA BAIXO", "Acorde Menor (ex: Am, Dm, Em)", COLOR_INDIGO),
            ("right", "PARA DIREITA", "Sustenido # (ex: Fa -> F#)", COLOR_AMBER),
            ("left", "PARA ESQUERDA", "Bemol b (ex: Si -> Bb)", COLOR_EMERALD),
        ]
        col_w = (w - 120) // 4
        for idx, (d, title, ex, col) in enumerate(orients):
            cx = 68 + idx * col_w + 30
            cy = 185
            cv2.circle(frame, (cx, cy), 22, COLOR_GLASS_FILL, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 22, col, 1, cv2.LINE_AA)
            draw_vector_arrow(frame, (cx, cy), d, col, size=11)
            draw_text_with_shadow(frame, title, (cx + 36, cy - 2), font_scale=0.45, color=COLOR_WHITE, thickness=2)
            draw_text_with_shadow(frame, ex, (cx + 36, cy + 20), font_scale=0.38, color=COLOR_SLATE_LIGHT, thickness=1)

        # Painel de Loop Session + Pulso
        draw_glass_card(frame, (40, 282), (w - 40, 480), COLOR_BG_DARK, COLOR_GLASS_BORDER, alpha=0.82, radius=14)
        draw_text_with_shadow(frame, "2. LOOP SESSION + IA DETECTOR DE TOM + PULSO RITMICO", (68, 320), font_scale=0.54, color=COLOR_EMERALD, thickness=2)
        tips = [
            "- Clique em [GRAVAR BASE] no palco: o relogio inicia quando voce fizer o primeiro acorde.",
            "- Faca sua progressao (ex: C -> G -> Am -> F) e clique em [FECHAR LOOP].",
            "- O motor Rust quantiza em compassos de 4 tempos e a IA detecta o TOM automaticamente.",
            "- Os 8 pads do topo travam na escala do tom detectado (zero notas erradas!) e o modo muda para MELODIA.",
            "- Balance a mao esquerda para baixo no ar (pulso) para alterar o BPM do loop em tempo real.",
        ]
        for i, line in enumerate(tips):
            draw_text_with_shadow(frame, line, (68, 358 + i * 24), font_scale=0.44, color=COLOR_SLATE_LIGHT, thickness=1)

        # Seletores de forma de onda
        draw_glass_card(frame, (40, 505), (w - 40, 648), COLOR_BG_DARK, COLOR_GLASS_BORDER, alpha=0.82, radius=14)
        draw_text_with_shadow(frame, "3. TIMBRE DO SINTETIZADOR (CLIQUE PARA ESCOLHER)", (68, 540), font_scale=0.52, color=COLOR_AMBER, thickness=2)
        waves = [("saw", "SAWTOOTH (Rico)"), ("sine", "SINE (Suave)"), ("square", "SQUARE (Retro)"), ("triangle", "TRIANGLE (Doce)")]
        wb_w = (w - 140) // 4
        for idx, (wf, wlbl) in enumerate(waves):
            wx = 68 + idx * (wb_w + 14)
            self._draw_button(
                frame, mouse,
                UIButton(f"wf_{wf}", (wx, 566, wb_w, 56), wlbl, active=(current_waveform == wf), accent=COLOR_ICE_BLUE),
                font_scale=0.46,
            )

        self._draw_button(frame, mouse, UIButton("btn_start_loop", (w - 310, h - 56, 270, 42), "IR PARA O PALCO >", active=True, accent=COLOR_EMERALD), font_scale=0.46)
        return frame

    def draw_calibration(
        self,
        frame: np.ndarray,
        hands: List[HandData],
        mouse: MouseController,
        classifier: MLPChordClassifier,
        selected_cls: int,
        is_recording: bool,
        rec_progress: int,
        rec_target: int,
        status_msg: str,
        camera_error: Optional[str],
    ) -> np.ndarray:
        h, w, _ = frame.shape
        mouse.clear_buttons()

        # Header
        draw_glass_card(frame, (24, 14), (w - 24, 62), COLOR_BG_DARK, COLOR_GLASS_BORDER, alpha=0.85, radius=12)
        self._draw_button(frame, mouse, UIButton("btn_menu", (36, 22, 160, 32), "< VOLTAR", accent=COLOR_ICE_BLUE), font_scale=0.42, radius=8)
        draw_text_with_shadow(frame, "CALIBRACAO AO VIVO DA REDE NEURAL MLP", (220, 45), font_scale=0.58, color=COLOR_WHITE, thickness=2)
        draw_text_with_shadow(
            frame,
            f"ACURACIA VALIDACAO: {classifier.val_accuracy * 100:.1f}%",
            (w - 310, 45),
            font_scale=0.46,
            color=COLOR_EMERALD,
            thickness=2,
        )

        if camera_error:
            draw_glass_card(frame, (280, 240), (w - 280, 390), COLOR_BG_DARK, COLOR_RED, alpha=0.88, radius=14)
            draw_text_with_shadow(frame, camera_error, (330, 305), font_scale=0.56, color=COLOR_RED, thickness=2)
            draw_text_with_shadow(frame, "Conecte uma webcam e tente novamente.", (330, 345), font_scale=0.46, color=COLOR_WHITE, thickness=1)
        else:
            # Desenha esqueleto minimalista e previsao atual
            for hand in hands:
                self._draw_skeleton_only(frame, hand.landmarks_px, COLOR_ICE_BLUE)
                p_idx, p_name, p_conf = classifier.predict_shape(hand.landmarks_norm)
                wx, wy = hand.landmarks_px[0]
                draw_glass_card(frame, (wx - 110, max(75, wy + 18)), (wx + 110, max(105, wy + 48)), COLOR_BG_DARK, COLOR_ICE_BLUE, alpha=0.85, radius=8)
                draw_text_with_shadow(frame, f"IA: {p_name} ({int(p_conf * 100)}%)", (wx - 96, max(96, wy + 39)), font_scale=0.45, color=COLOR_WHITE, thickness=1)

        # Painel direito de selecao das 8 classes
        px = w - 340
        draw_glass_card(frame, (px, 78), (w - 24, h - 92), COLOR_BG_DARK, COLOR_GLASS_BORDER, alpha=0.84, radius=14)
        draw_text_with_shadow(frame, "ESCOLHA O GESTOS/ACORDE:", (px + 18, 108), font_scale=0.44, color=COLOR_ICE_LIGHT, thickness=2)

        counts = classifier.user_sample_counts()
        for idx, cname in enumerate(SHAPE_CLASSES):
            by = 124 + idx * 58
            lbl = f"{idx}: {cname}  ({counts[idx]} am.)"
            self._draw_button(
                frame, mouse,
                UIButton(f"cal_cls_{idx}", (px + 16, by, 284, 48), lbl, active=(idx == selected_cls), accent=COLOR_ICE_BLUE),
                font_scale=0.44,
                radius=10,
            )

        # Preview do icone da classe selecionada no canto inferior esquerdo
        draw_glass_card(frame, (24, h - 228), (180, h - 90), COLOR_BG_DARK, COLOR_ICE_BLUE, alpha=0.84, radius=12)
        draw_text_with_shadow(frame, "SINAL ALVO", (54, h - 206), font_scale=0.38, color=COLOR_ICE_LIGHT, thickness=1)
        draw_hand_sign_icon(frame, (102, h - 148), 44, selected_cls, "up", COLOR_ICE_BLUE, success=False)

        # Barra inferior de acoes
        draw_glass_card(frame, (24, h - 78), (w - 24, h - 16), COLOR_BG_DARK, COLOR_GLASS_BORDER, alpha=0.86, radius=12)
        rec_lbl = f"GRAVANDO {rec_progress}/{rec_target}..." if is_recording else f"GRAVAR AMOSTRAS ({SHAPE_CLASSES[selected_cls]})"
        self._draw_button(frame, mouse, UIButton("btn_cal_record", (40, h - 66, 280, 42), rec_lbl, active=is_recording, accent=COLOR_AMBER), font_scale=0.44)
        self._draw_button(frame, mouse, UIButton("btn_cal_train", (336, h - 66, 230, 42), "TREINAR REDE MLP", accent=COLOR_EMERALD), font_scale=0.44)
        self._draw_button(frame, mouse, UIButton("btn_cal_reset", (582, h - 66, 180, 42), "MODELO PADRAO", accent=COLOR_RED), font_scale=0.44)
        draw_text_with_shadow(frame, status_msg, (780, h - 40), font_scale=0.42, color=COLOR_SLATE_LIGHT, thickness=1)
        return frame

    def draw_stage(
        self,
        frame: np.ndarray,
        mouse: MouseController,
        hands: List[HandData],
        right_hand: Optional[HandData],
        left_hand: Optional[HandData],
        keyboard: HolographicVirtualKeyboard,
        chord_det: Optional[ChordDetection],
        detected_key: Optional[DetectedKey],
        guide: ChordProgressionGuide,
        active_note: Optional[int],
        right_mode: str,  # "CHORD" | "MELODY"
        engine_online: bool,
        engine_state: Dict,
        loop_mirror_chords: List[Dict],
        current_waveform: str,
        fps: float,
        cutoff: float,
        pitch_bend: float,
        bpm: float,
        pulse_glow: float,
        camera_error: Optional[str] = None,
        now: Optional[float] = None,
    ) -> np.ndarray:
        if now is None:
            now = time.monotonic()
        h, w, _ = frame.shape
        mouse.clear_buttons()
        self.pulse_phase = (self.pulse_phase + 0.08) % (2 * math.pi)
        pulse = 0.5 + 0.5 * math.sin(self.pulse_phase)

        # 1. Pads Holograficos no topo (mostram escala do tom travado e destacam chord tones!)
        self._draw_acoustic_pads(frame, keyboard, active_note, right_mode == "MELODY", pulse)

        # 2. Erro de camera (se nao conectada)
        if camera_error:
            draw_glass_card(frame, (300, 290), (w - 300, 410), COLOR_BG_DARK, COLOR_RED, alpha=0.88, radius=14)
            draw_text_with_shadow(frame, camera_error, (350, 345), font_scale=0.56, color=COLOR_RED, thickness=2)
            draw_text_with_shadow(frame, "Use os botoes abaixo ou volte ao Menu.", (350, 380), font_scale=0.45, color=COLOR_WHITE, thickness=1)

        # 3. Esqueleto minimalista das maos
        self._draw_hands_stage(frame, hands, right_hand, right_mode, chord_det, cutoff, bpm, pulse)

        # 4. Widgets laterais (Cutoff com cv2.ellipse + Pitch Bend)
        self._draw_cutoff_widget(frame, cutoff, h)
        self._draw_pitch_bend_widget(frame, pitch_bend, w, h)

        # 5. Painel visual do Modo Treino (com icone da mao + celebracao antes de avancar)
        if guide.is_active:
            self._draw_training_card(frame, mouse, guide, pulse, w, now)

        # 6. Timeline do Loop Session (acima do dock)
        self._draw_loop_timeline(frame, engine_state, loop_mirror_chords, w, h)

        # 7. Header clicavel e Dock inferior clicavel
        self._draw_stage_header(
            frame, mouse, engine_online, current_waveform, keyboard, detected_key, bpm, pulse_glow, fps, w
        )
        self._draw_stage_dock(frame, mouse, engine_state, right_mode, guide.is_active, w, h)
        return frame

    def _draw_acoustic_pads(
        self,
        frame: np.ndarray,
        keyboard: HolographicVirtualKeyboard,
        active_note: Optional[int],
        melody_mode: bool,
        pulse: float,
    ):
        h, w, _ = frame.shape
        zones = keyboard.get_zones(w, h)
        for zone in zones:
            zx, zy, zw, zh = zone["rect"]
            note = zone["note"]
            pc = zone["pc"]
            name = zone["name"]
            is_active = (active_note == note)
            is_chord_tone = pc in keyboard.chord_tone_pcs

            if is_active:
                pad_bg = (95, 65, 28)
                pad_border = COLOR_ICE_BLUE
                pad_alpha = 0.88 if melody_mode else 0.72
            elif is_chord_tone:
                pad_bg = (34, 48, 64)
                pad_border = COLOR_AMBER
                pad_alpha = (0.74 + pulse * 0.10) if melody_mode else 0.56
            else:
                pad_bg = COLOR_GLASS_FILL
                pad_border = (78, 60, 45)
                pad_alpha = 0.65 if melody_mode else 0.42

            draw_glass_card(frame, (zx, zy), (zx + zw, zy + zh), pad_bg, pad_border, pad_alpha, radius=12)
            if is_chord_tone:
                cv2.circle(frame, (zx + zw - 14, zy + 14), 4, COLOR_AMBER, -1, cv2.LINE_AA)

            base_name = name[:-1] if name[-1].isdigit() else name
            octave = name[-1] if name[-1].isdigit() else ""
            tsz = cv2.getTextSize(base_name, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)[0]
            tx = zx + (zw - tsz[0]) // 2 - 4
            ty = zy + (zh // 2) + 8
            draw_text_with_shadow(frame, base_name, (tx, ty), font_scale=0.72, color=COLOR_WHITE, thickness=2)
            if octave:
                draw_text_with_shadow(
                    frame, octave, (tx + tsz[0] + 3, ty - 8),
                    font_scale=0.42,
                    color=COLOR_AMBER if is_chord_tone else COLOR_ICE_LIGHT,
                    thickness=1,
                )

    @staticmethod
    def _draw_skeleton_only(frame: np.ndarray, pts: List[Tuple[int, int]], line_color: Tuple[int, int, int]):
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17),
        ]
        for p1, p2 in connections:
            cv2.line(frame, pts[p1], pts[p2], line_color, 1, cv2.LINE_AA)
        for i, p in enumerate(pts):
            if i in (4, 8, 12, 16, 20):
                cv2.circle(frame, p, 4, COLOR_WHITE, -1, cv2.LINE_AA)
                cv2.circle(frame, p, 6, line_color, 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, p, 2, line_color, -1, cv2.LINE_AA)

    def _draw_hands_stage(
        self,
        frame: np.ndarray,
        hands: List[HandData],
        right_hand: Optional[HandData],
        right_mode: str,
        chord_det: Optional[ChordDetection],
        cutoff: float,
        bpm: float,
        pulse: float,
    ):
        for hand in hands:
            pts = hand.landmarks_px
            is_right = hand is right_hand
            line_color = COLOR_ICE_BLUE if is_right else COLOR_INDIGO
            self._draw_skeleton_only(frame, pts, line_color)

            tip8 = pts[8]
            if is_right and right_mode == "MELODY":
                cv2.circle(frame, tip8, int(11 + pulse * 3), line_color, 1, cv2.LINE_AA)
                cv2.circle(frame, tip8, 2, COLOR_WHITE, -1, cv2.LINE_AA)

            wx, wy = pts[0]
            tag_y = max(wy + 24, 90)
            if is_right:
                if right_mode == "CHORD" and chord_det is not None:
                    tag_text = f"ACORDE: {chord_det.chord_label} ({int(chord_det.confidence * 100)}%)"
                    badge_w = 265
                    draw_glass_card(frame, (wx - badge_w // 2, tag_y), (wx + badge_w // 2, tag_y + 28), COLOR_BG_DARK, line_color, alpha=0.84, radius=8)
                    draw_text_with_shadow(frame, tag_text, (wx - badge_w // 2 + 12, tag_y + 19), font_scale=0.42, color=COLOR_WHITE, thickness=1)
                    if chord_det.arrow_dir != "none":
                        draw_vector_arrow(frame, (wx + badge_w // 2 - 18, tag_y + 14), chord_det.arrow_dir, COLOR_AMBER, size=7)
                else:
                    tag_text = "MELODIA  |  NO TOM"
                    badge_w = 185
                    draw_glass_card(frame, (wx - badge_w // 2, tag_y), (wx + badge_w // 2, tag_y + 28), COLOR_BG_DARK, line_color, alpha=0.84, radius=8)
                    draw_text_with_shadow(frame, tag_text, (wx - badge_w // 2 + 14, tag_y + 19), font_scale=0.42, color=COLOR_WHITE, thickness=1)
            else:
                tag_text = f"PULSO {int(round(bpm))} BPM | CUTOFF {int(cutoff * 100)}%"
                badge_w = 248
                draw_glass_card(frame, (wx - badge_w // 2, tag_y), (wx + badge_w // 2, tag_y + 28), COLOR_BG_DARK, line_color, alpha=0.84, radius=8)
                draw_text_with_shadow(frame, tag_text, (wx - badge_w // 2 + 12, tag_y + 19), font_scale=0.40, color=COLOR_WHITE, thickness=1)

    def _draw_cutoff_widget(self, frame: np.ndarray, cutoff: float, h: int):
        x1, y1, x2, y2 = 28, h - 196, 174, h - 72
        draw_glass_card(frame, (x1, y1), (x2, y2), COLOR_BG_DARK, COLOR_GLASS_BORDER, alpha=0.78, radius=14)
        draw_text_with_shadow(frame, "FILTRO CUTOFF", (x1 + 16, y1 + 22), font_scale=0.40, color=COLOR_SLATE_LIGHT, thickness=1)

        cx, cy = (x1 + x2) // 2, y1 + 70
        radius = 30
        cv2.ellipse(frame, (cx, cy), (radius, radius), 0, 135, 405, COLOR_MUTED_DARK, 3, cv2.LINE_AA)
        span = int(max(1, cutoff * 270))
        cv2.ellipse(frame, (cx, cy), (radius, radius), 0, 135, 135 + span, COLOR_ICE_BLUE, 3, cv2.LINE_AA)

        val_str = f"{int(cutoff * 100)}%"
        tsz = cv2.getTextSize(val_str, cv2.FONT_HERSHEY_SIMPLEX, 0.54, 2)[0]
        draw_text_with_shadow(frame, val_str, (cx - tsz[0] // 2, cy + 7), font_scale=0.54, color=COLOR_WHITE, thickness=2)
        draw_text_with_shadow(frame, "MAO ESQUERDA", (x1 + 22, y2 - 10), font_scale=0.35, color=COLOR_ICE_LIGHT, thickness=1)

    def _draw_pitch_bend_widget(self, frame: np.ndarray, pitch_bend: float, w: int, h: int):
        x1, x2 = w - 74, w - 26
        y1, y2 = int(h * 0.38), int(h * 0.72)
        draw_glass_card(frame, (x1, y1), (x2, y2), COLOR_BG_DARK, COLOR_GLASS_BORDER, alpha=0.78, radius=12)
        draw_text_with_shadow(frame, "BEND", (x1 + 6, y1 - 8), font_scale=0.40, color=COLOR_SLATE_LIGHT, thickness=1)

        mid_x = (x1 + x2) // 2
        top, bot = y1 + 16, y2 - 16
        mid_y = (top + bot) // 2
        cv2.line(frame, (mid_x, top), (mid_x, bot), COLOR_MUTED_DARK, 2, cv2.LINE_AA)
        cv2.line(frame, (mid_x - 8, mid_y), (mid_x + 8, mid_y), COLOR_ICE_BLUE, 1, cv2.LINE_AA)

        clamped = float(np.clip(pitch_bend, -2.0, 2.0))
        knob_y = int(mid_y - (clamped / 2.0) * (mid_y - top))
        is_bent = abs(clamped) > 0.08
        kcol = COLOR_ICE_BLUE if is_bent else COLOR_WHITE
        cv2.circle(frame, (mid_x, knob_y), 7, kcol, -1, cv2.LINE_AA)
        val_str = "0.0" if abs(clamped) < 0.05 else f"{clamped:+.1f}"
        draw_text_with_shadow(frame, val_str, (x1 - 42, knob_y + 5), font_scale=0.40, color=kcol, thickness=1)

    def _draw_training_card(
        self, frame: np.ndarray, mouse: MouseController, guide: ChordProgressionGuide, pulse: float, w: int, now: float
    ):
        x1, y1, x2, y2 = 28, 262, 410, 512
        celebrating = guide.phase == "CELEBRATING"
        step = guide.get_display_step()

        if celebrating:
            border_col = COLOR_EMERALD
            bg_col = (22, 44, 24)
            thick = 2
        elif guide.feedback_type == "warning":
            border_col = COLOR_AMBER
            bg_col = (24, 22, 16)
            thick = 2
        else:
            border_col = COLOR_ICE_BLUE
            bg_col = COLOR_BG_DARK
            thick = 1

        draw_glass_card(frame, (x1, y1), (x2, y2), bg_col, border_col, alpha=0.86, radius=14, border_thickness=thick)

        # Cabecalho do card
        cv2.circle(frame, (x1 + 18, y1 + 22), int(4 + pulse * 2.0), COLOR_EMERALD if celebrating else COLOR_AMBER, -1, cv2.LINE_AA)
        draw_text_with_shadow(frame, "MODO TREINO", (x1 + 30, y1 + 27), font_scale=0.44, color=COLOR_WHITE, thickness=2)
        draw_text_with_shadow(frame, f"{guide.score} PTS | {guide.get_accuracy()}%", (x1 + 154, y1 + 27), font_scale=0.40, color=COLOR_EMERALD, thickness=1)
        self._draw_button(
            frame, mouse,
            UIButton("btn_next_song", (x2 - 102, y1 + 8, 92, 26), "PROXIMA >", accent=COLOR_AMBER),
            font_scale=0.35,
            radius=6,
        )

        # Pills dos passos da progressao
        pill_x = x1 + 14
        pill_y = y1 + 42
        steps = guide.song.steps
        pw = max(48, (x2 - x1 - 28 - (len(steps) - 1) * 6) // max(1, len(steps)))
        for idx, st in enumerate(steps):
            px1 = pill_x + idx * (pw + 6)
            is_cur = idx == guide.current_step
            is_done = idx < guide.current_step
            if is_cur and celebrating:
                pbg, pbd, pcol = (34, 88, 42), COLOR_EMERALD, COLOR_EMERALD
            elif is_cur:
                pbg, pbd, pcol = (64, 46, 22), COLOR_ICE_BLUE, COLOR_WHITE
            elif is_done:
                pbg, pbd, pcol = (28, 48, 30), (80, 140, 80), COLOR_SLATE_LIGHT
            else:
                pbg, pbd, pcol = (24, 20, 16), COLOR_GLASS_BORDER, COLOR_MUTED
            draw_glass_card(frame, (px1, pill_y), (px1 + pw, pill_y + 24), pbg, pbd, alpha=0.85, radius=6)
            tsz = cv2.getTextSize(st.cifra, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
            draw_text_with_shadow(frame, st.cifra, (px1 + (pw - tsz[0]) // 2, pill_y + 17), font_scale=0.42, color=pcol, thickness=1)

        if step is None:
            return

        # Sub-painel esquerdo: Icone visual de como a mao deve ficar
        ix1, iy1, ix2, iy2 = x1 + 14, y1 + 76, x1 + 148, y2 - 38
        draw_glass_card(frame, (ix1, iy1), (ix2, iy2), (18, 15, 12), COLOR_EMERALD if celebrating else COLOR_GLASS_BORDER, alpha=0.88, radius=12)
        draw_hand_sign_icon(
            frame,
            ((ix1 + ix2) // 2, (iy1 + iy2) // 2 - 4),
            48,
            step.shape_idx,
            step.arrow_dir,
            accent=COLOR_ICE_BLUE,
            success=celebrating,
        )
        orient_txt = {"up": "CIMA (Maior)", "down": "BAIXO (Menor)", "right": "DIR (#)", "left": "ESQ (b)"}.get(step.arrow_dir, "")
        draw_text_with_shadow(frame, orient_txt, (ix1 + 10, iy2 - 8), font_scale=0.35, color=COLOR_AMBER if not celebrating else COLOR_EMERALD, thickness=1)

        # Sub-painel direito: Acorde alvo + feedback de acerto
        tx = ix2 + 16
        if celebrating:
            draw_text_with_shadow(frame, "ACERTOU!", (tx, iy1 + 30), font_scale=0.72, color=COLOR_EMERALD, thickness=2)
            draw_text_with_shadow(frame, f"Sinal {step.cifra} confirmado!", (tx, iy1 + 58), font_scale=0.48, color=COLOR_WHITE, thickness=1)
            draw_text_with_shadow(frame, "Proximo acorde em instantes...", (tx, iy1 + 84), font_scale=0.39, color=COLOR_SLATE_LIGHT, thickness=1)
            # Barra de progresso do hold antes de avancar
            prog = guide.celebration_progress(now)
            bar_x1, bar_y1, bar_x2, bar_y2 = tx, iy2 - 20, x2 - 16, iy2 - 8
            cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (32, 42, 34), -1)
            fill_x = int(bar_x1 + (bar_x2 - bar_x1) * prog)
            cv2.rectangle(frame, (bar_x1, bar_y1), (fill_x, bar_y2), COLOR_EMERALD, -1)
        else:
            draw_text_with_shadow(frame, f"FACA: {step.cifra}", (tx, iy1 + 32), font_scale=0.76, color=COLOR_WHITE, thickness=2)
            draw_text_with_shadow(frame, step.shape_hint, (tx, iy1 + 60), font_scale=0.42, color=COLOR_ICE_LIGHT, thickness=1)
            draw_vector_arrow(frame, (tx + 12, iy1 + 88), step.arrow_dir, COLOR_AMBER, size=9)
            draw_text_with_shadow(frame, f"Direcao: {orient_txt}", (tx + 30, iy1 + 93), font_scale=0.39, color=COLOR_AMBER, thickness=1)

        # Linha inferior: mensagem de feedback / titulo da progressao
        msg_col = COLOR_EMERALD if celebrating else (COLOR_AMBER if guide.feedback_type == "warning" else COLOR_SLATE_LIGHT)
        draw_text_with_shadow(frame, guide.feedback_message[:48], (x1 + 16, y2 - 12), font_scale=0.39, color=msg_col, thickness=1)

    def _draw_loop_timeline(
        self, frame: np.ndarray, engine_state: Dict, loop_mirror_chords: List[Dict], w: int, h: int
    ):
        lstate = engine_state.get("loop_state", "idle")
        beat_pos = float(engine_state.get("beat_pos", 0.0))
        loop_len = float(engine_state.get("loop_len_beats", 0.0))

        box_w, box_h = 740, 44
        bx, by = (w - box_w) // 2, h - 108

        if lstate == "recording_base":
            border, state_lbl, state_col = COLOR_RED, "GRAVANDO BASE", COLOR_RED
        elif lstate == "overdubbing":
            border, state_lbl, state_col = COLOR_AMBER, "OVERDUB MELODIA", COLOR_AMBER
        elif lstate == "playing":
            border, state_lbl, state_col = COLOR_EMERALD, "LOOP TOCANDO", COLOR_EMERALD
        else:
            border, state_lbl, state_col = COLOR_GLASS_BORDER, "LOOP VAZIO", COLOR_MUTED

        draw_glass_card(frame, (bx, by), (bx + box_w, by + box_h), COLOR_BG_DARK, border, alpha=0.80, radius=10)
        cv2.circle(frame, (bx + 18, by + 22), 5, state_col, -1, cv2.LINE_AA)
        draw_text_with_shadow(frame, state_lbl, (bx + 32, by + 27), font_scale=0.42, color=state_col, thickness=2)

        track_x1, track_x2 = bx + 185, bx + box_w - 95
        track_w = track_x2 - track_x1
        cv2.rectangle(frame, (track_x1, by + 8), (track_x2, by + 36), (26, 22, 18), -1)

        if loop_mirror_chords and loop_len >= 4.0:
            for ev in loop_mirror_chords:
                sx = int(track_x1 + (ev["start_beat"] / loop_len) * track_w)
                ex = int(track_x1 + min(loop_len, ev["start_beat"] + ev["duration_beats"]) / loop_len * track_w)
                if ex > sx + 2:
                    cv2.rectangle(frame, (sx + 1, by + 10), (ex - 1, by + 34), (48, 36, 24), -1)
                    cv2.rectangle(frame, (sx + 1, by + 10), (ex - 1, by + 34), COLOR_ICE_BLUE, 1)
                    draw_text_with_shadow(frame, ev.get("short_label", "?"), (sx + 6, by + 27), font_scale=0.38, color=COLOR_WHITE, thickness=1)

            play_x = int(track_x1 + ((beat_pos % loop_len) / loop_len) * track_w)
            cv2.line(frame, (play_x, by + 6), (play_x, by + 38), COLOR_EMERALD, 2, cv2.LINE_AA)
            beat_txt = f"{beat_pos + 1.0:.1f}/{int(loop_len)}"
        elif loop_mirror_chords:
            summary = " - ".join(ev.get("short_label", "?") for ev in loop_mirror_chords[-6:])
            draw_text_with_shadow(frame, summary, (track_x1 + 10, by + 27), font_scale=0.40, color=COLOR_WHITE, thickness=1)
            beat_txt = f"{beat_pos:.1f}b"
        else:
            draw_text_with_shadow(frame, "Clique em [GRAVAR BASE] para criar progressao", (track_x1 + 14, by + 27), font_scale=0.39, color=COLOR_MUTED, thickness=1)
            beat_txt = "0/4"

        draw_text_with_shadow(frame, beat_txt, (track_x2 + 12, by + 27), font_scale=0.40, color=COLOR_SLATE_LIGHT, thickness=1)

    def _draw_stage_header(
        self,
        frame: np.ndarray,
        mouse: MouseController,
        engine_online: bool,
        current_waveform: str,
        keyboard: HolographicVirtualKeyboard,
        detected_key: Optional[DetectedKey],
        bpm: float,
        pulse_glow: float,
        fps: float,
        w: int,
    ):
        bar_h, bar_y, margin_x = 46, 12, 24
        bar_w = w - 2 * margin_x
        draw_glass_card(frame, (margin_x, bar_y), (margin_x + bar_w, bar_y + bar_h), COLOR_BG_DARK, COLOR_GLASS_BORDER, alpha=0.84, radius=12)

        self._draw_button(frame, mouse, UIButton("btn_menu", (margin_x + 10, bar_y + 7, 94, 32), "< MENU", accent=COLOR_ICE_BLUE), font_scale=0.40, radius=8)

        fps_val = int(round(fps))
        fps_col = COLOR_EMERALD if fps_val >= 26 else (COLOR_AMBER if fps_val >= 18 else COLOR_RED)
        draw_text_with_shadow(frame, f"FPS {fps_val}", (margin_x + 116, bar_y + 29), font_scale=0.45, color=fps_col, thickness=2)

        # Badge de Tom Detectado / Escala
        if detected_key is not None and detected_key.locked:
            key_txt = f"TOM: {detected_key.key_name} ({int(detected_key.confidence * 100)}%)"
            draw_glass_card(frame, (margin_x + 205, bar_y + 7), (margin_x + 485, bar_y + 39), (32, 42, 54), COLOR_EMERALD, alpha=0.82, radius=8)
            draw_text_with_shadow(frame, key_txt, (margin_x + 218, bar_y + 28), font_scale=0.42, color=COLOR_EMERALD, thickness=1)
        else:
            esc_lbl = f"ESCALA: {keyboard.scale_name.split(' ')[0]}"
            self._draw_button(frame, mouse, UIButton("btn_scale", (margin_x + 205, bar_y + 7, 210, 32), esc_lbl, accent=COLOR_ICE_BLUE), font_scale=0.40, radius=8)

        # Indicador de BPM + Pulso
        bpm_x = margin_x + 515
        p_rad = int(6 + pulse_glow * 5)
        p_col = COLOR_AMBER if pulse_glow > 0.1 else COLOR_ICE_BLUE
        cv2.circle(frame, (bpm_x, bar_y + 23), p_rad, p_col, -1 if pulse_glow > 0.1 else 1, cv2.LINE_AA)
        draw_text_with_shadow(frame, f"BPM {int(round(bpm))}", (bpm_x + 16, bar_y + 29), font_scale=0.46, color=COLOR_WHITE, thickness=2)

        # Botao Onda + Status DSP
        self._draw_button(
            frame, mouse,
            UIButton("btn_waveform", (w - margin_x - 290, bar_y + 7, 160, 32), f"ONDA: {current_waveform.upper()}", accent=COLOR_ICE_BLUE),
            font_scale=0.40,
            radius=8,
        )
        eng_col = COLOR_EMERALD if engine_online else COLOR_RED
        cv2.circle(frame, (w - margin_x - 108, bar_y + 24), 5, eng_col, -1, cv2.LINE_AA)
        draw_text_with_shadow(frame, "RUST DSP", (w - margin_x - 96, bar_y + 29), font_scale=0.42, color=COLOR_WHITE, thickness=1)

    def _draw_stage_dock(
        self,
        frame: np.ndarray,
        mouse: MouseController,
        engine_state: Dict,
        right_mode: str,
        guide_active: bool,
        w: int,
        h: int,
    ):
        lstate = engine_state.get("loop_state", "idle")
        if lstate == "idle":
            loop_lbl, loop_acc, loop_act = "GRAVAR BASE", COLOR_RED, False
        elif lstate == "recording_base":
            loop_lbl, loop_acc, loop_act = "FECHAR LOOP", COLOR_EMERALD, True
        elif lstate == "playing":
            loop_lbl, loop_acc, loop_act = "+ OVERDUB", COLOR_AMBER, False
        else:
            loop_lbl, loop_acc, loop_act = "PARAR OVERDUB", COLOR_AMBER, True

        mode_lbl = "MODO: ACORDES" if right_mode == "CHORD" else "MODO: MELODIA"
        buttons = [
            UIButton("btn_loop_toggle", (0, 0, 175, 38), loop_lbl, active=loop_act, accent=loop_acc),
            UIButton("btn_loop_clear", (0, 0, 110, 38), "LIMPAR", accent=COLOR_MUTED),
            UIButton("btn_mode_toggle", (0, 0, 185, 38), mode_lbl, active=(right_mode == "MELODY"), accent=COLOR_ICE_BLUE),
            UIButton("btn_toggle_guide", (0, 0, 130, 38), "TREINO", active=guide_active, accent=COLOR_AMBER),
            UIButton("btn_calibrate", (0, 0, 145, 38), "CALIBRAR IA", accent=COLOR_EMERALD),
            UIButton("btn_panic", (0, 0, 120, 38), "SILENCIAR", accent=COLOR_RED),
        ]

        gap = 10
        total_w = sum(b.rect[2] for b in buttons) + gap * (len(buttons) - 1)
        start_x = (w - total_w) // 2
        y = h - 52
        draw_glass_card(frame, (start_x - 12, y - 6), (start_x + total_w + 12, y + 44), COLOR_BG_DARK, COLOR_GLASS_BORDER, alpha=0.84, radius=12)

        cx = start_x
        for b in buttons:
            bw, bh = b.rect[2], b.rect[3]
            b.rect = (cx, y, bw, bh)
            self._draw_button(frame, mouse, b, font_scale=0.41, radius=8)
            cx += bw + gap

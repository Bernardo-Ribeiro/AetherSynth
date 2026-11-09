import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

from gestures import HolographicVirtualKeyboard, MLGestureClassifier, SCALES
from hud import HolographicHUD
from tracker import HandTracker
from tutor import IntelligentMusicTutor, SONGS
from udp_client import AudioEngineClient


WAVEFORMS = ["saw", "sine", "square", "triangle"]
SCALE_NAMES = list(SCALES.keys())


def main():
    print("=====================================================")
    print("  AETHERSYNTH - AI Gestural Synthesizer & Tutor")
    print("=====================================================")

    # 1. Conecta ao Audio Engine (Rust)
    audio = AudioEngineClient(host="127.0.0.1", port=8765)
    is_online = audio.ping()
    if is_online:
        print("[OK] Rust Audio Engine conectado via UDP (127.0.0.1:8765)")
    else:
        print("[AVISO] Rust Audio Engine offline. Certifique-se de iniciar o binario de audio.")

    # 2. Inicializa componentes
    tracker = HandTracker(max_hands=2, min_detection_conf=0.7, min_track_conf=0.6)
    classifier = MLGestureClassifier()
    keyboard = HolographicVirtualKeyboard(scale_name=SCALE_NAMES[0])
    tutor = IntelligentMusicTutor()
    hud = HolographicHUD()

    waveform_idx = 0
    scale_idx = 0
    audio.set_waveform(WAVEFORMS[waveform_idx])

    # 3. Abre a Webcam (640x480 nativo a 30 FPS para estabilidade maxima de CPU)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERRO] Nao foi possivel abrir a webcam (indice 0). Verifique se a camera esta conectada.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Resolução de renderização em Alta Definição (HD 720p) para a HUD
    HUD_W, HUD_H = 1280, 720

    cv2.namedWindow("AetherSynth", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AetherSynth", HUD_W, HUD_H)

    active_note = None
    last_mod_time = 0.0
    cutoff = 0.7
    pitch_bend = 0.0

    prev_frame_time = time.time()
    fps = 30.0

    print("\nSistema pronto! Toque nas teclas holograficas com a mao direita.")
    print("Use a mao esquerda para controlar o filtro e o pitch bend.")
    print("Pressione ESPACO para ligar/desligar o Modo Tutor. 'q' para sair.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[AVISO] Falha ao capturar quadro da webcam.")
                time.sleep(0.01)
                continue

            # Espelha o quadro para interacao intuitiva
            frame = cv2.flip(frame, 1)

            # Upscale leve para HUD em Alta Definição (1280x720) sem perda de FPS
            display_frame = cv2.resize(frame, (HUD_W, HUD_H), interpolation=cv2.INTER_LINEAR)
            h, w = HUD_H, HUD_W

            # Calcula FPS
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_frame_time, 1e-5))
            prev_frame_time = now

            # Processa rastreamento na resolucao nativa da camera mas projeta os pontos na HUD HD
            hands, _ = tracker.process_frame(frame, target_w=HUD_W, target_h=HUD_H)

            # Separação funcional das mãos
            right_hand = None  # Mão da Melodia (na tela espelhada, a mão do lado direito do usuário)
            left_hand = None   # Mão da Modulação

            for hand in hands:
                # Com a tela espelhada:
                # hand.label 'Left' do MediaPipe aparece no lado direito da tela (mão direita do usuário)
                # hand.label 'Right' aparece no lado esquerdo da tela (mão esquerda do usuário)
                if hand.label == "Left":
                    right_hand = hand
                else:
                    left_hand = hand

            # Se detectar apenas 1 mão, usa a posição na tela para decidir o papel
            if len(hands) == 1:
                h_only = hands[0]
                if h_only.center[0] > w * 0.4:
                    right_hand = h_only
                    left_hand = None
                else:
                    left_hand = h_only
                    right_hand = None

            # --- AÇÃO DA MÃO DA MELODIA (Tocar Notas) ---
            hit_note = None
            if right_hand is not None:
                hit_note = keyboard.check_hit(right_hand, w, h)

            if hit_note != active_note:
                if active_note is not None:
                    audio.note_off(active_note)
                if hit_note is not None:
                    audio.note_on(hit_note, velocity=0.85)
                    if tutor.is_active:
                        tutor.register_played_note(hit_note)
                active_note = hit_note

            # Se não houver mão direita e houver nota ativa, desliga
            if right_hand is None and active_note is not None:
                audio.note_off(active_note)
                active_note = None

            # --- AÇÃO DA MÃO DA MODULAÇÃO (Cutoff e Pitch Bend) ---
            if left_hand is not None:
                # 1. Abertura da mão controla o Cutoff (filtro passa-baixas)
                cutoff = float(np.clip(left_hand.openness, 0.05, 1.0))

                # 2. Altura vertical da mão controla o Pitch Bend (-2 a +2 semitons)
                # Centro vertical normalizado (0.0 topo, 1.0 fundo)
                norm_y = left_hand.center[1] / h
                # Centro neutro é 0.5; acima é pitch up, abaixo é pitch down
                pitch_bend = float(np.clip((0.5 - norm_y) * 4.0, -2.0, 2.0))

                # Envia modulação para o Rust a cada ~20ms para economizar tráfego sem perder fluidez
                if now - last_mod_time > 0.02:
                    audio.modulate(cutoff=cutoff, pitch_bend=pitch_bend, volume=0.85)
                    last_mod_time = now
            else:
                # Retorna gradualmente o pitch bend para o centro se a mão sair
                if abs(pitch_bend) > 0.05:
                    pitch_bend *= 0.8
                    audio.modulate(pitch_bend=pitch_bend)

            # Verifica status da engine a cada 3 segundos
            if int(now) % 3 == 0 and int(now) != getattr(main, "_last_ping_sec", 0):
                setattr(main, "_last_ping_sec", int(now))
                is_online = audio.ping()

            # --- RENDERIZAÇÃO DO HUD EM ALTA DEFINIÇÃO ---
            hud.draw(
                frame=display_frame,
                hands=hands,
                keyboard=keyboard,
                classifier=classifier,
                tutor=tutor,
                active_note=active_note,
                engine_online=is_online,
                current_waveform=WAVEFORMS[waveform_idx],
                fps=fps,
                cutoff=cutoff,
                pitch_bend=pitch_bend,
            )

            cv2.imshow("AetherSynth", display_frame)

            # --- TECLAS DE ATALHO ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' ou ESC
                break
            elif key == ord(" "):  # ESPAÇO: Alterna Modo Tutor
                if tutor.is_active:
                    tutor.stop()
                else:
                    tutor.start_song(0)
            elif key == ord("t") or key == ord("T"):  # 't': Próxima música do Tutor
                if not tutor.is_active:
                    tutor.start_song(0)
                else:
                    next_idx = (tutor.active_song_idx + 1) % len(SONGS)
                    tutor.start_song(next_idx)
            elif key == ord("w") or key == ord("W"):  # 'w': Alterna Forma de Onda
                waveform_idx = (waveform_idx + 1) % len(WAVEFORMS)
                audio.set_waveform(WAVEFORMS[waveform_idx])
            elif key == ord("e") or key == ord("E"):  # 'e': Alterna Escala
                scale_idx = (scale_idx + 1) % len(SCALE_NAMES)
                keyboard.set_scale(SCALE_NAMES[scale_idx])
            elif key == ord("r") or key == ord("R"):  # 'r': Reset notas
                audio.all_notes_off()
                active_note = None

    finally:
        print("\nEncerrando Gestures Music AI...")
        audio.all_notes_off()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

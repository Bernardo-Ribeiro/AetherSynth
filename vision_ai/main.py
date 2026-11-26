import os
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

from gestures import (
    ChordDetection,
    ChordInterpreter,
    HolographicVirtualKeyboard,
    MLPChordClassifier,
    SCALES,
    SHAPE_CLASSES,
)
from harmony_ai import HarmonyKeyDetector, chord_tones_for
from hud import HolographicHUD, MouseController
from rhythm import RhythmPulseDetector
from tracker import HandTracker, ThreadedCamera
from tutor import ChordProgressionGuide
from udp_client import AudioEngineClient


WAVEFORMS = ["saw", "sine", "square", "triangle"]
SCALE_NAMES = list(SCALES.keys())
HUD_W, HUD_H = 1280, 720


def _rescale_mirror_chords(events: List[Dict], raw_len: float) -> tuple[List[Dict], float]:
    raw_len = max(1.0, raw_len)
    q_len = max(4.0, round(raw_len / 4.0) * 4.0)
    scale = q_len / raw_len
    rescaled = []
    for ev in events:
        sb = max(0.0, min(q_len - 0.1, ev["start_beat"] * scale))
        dur = max(0.25, ev["duration_beats"] * scale)
        dur = min(dur, max(0.1, q_len - sb))
        item = dict(ev)
        item["start_beat"] = sb
        item["duration_beats"] = dur
        rescaled.append(item)
    return rescaled, q_len


def main():
    print("=====================================================")
    print("  AETHERSYNTH - Stage Chord AI & Loop Session")
    print("=====================================================")

    audio = AudioEngineClient(host="127.0.0.1", port=8765)
    audio.ping()
    last_ping_time = time.monotonic()

    tracker = HandTracker(max_hands=2, min_detection_conf=0.65, min_track_conf=0.55)
    classifier = MLPChordClassifier()
    chord_interp = ChordInterpreter(classifier)
    harmony_ai = HarmonyKeyDetector()
    rhythm_det = RhythmPulseDetector(initial_bpm=96.0)
    keyboard = HolographicVirtualKeyboard(scale_name=SCALE_NAMES[0])
    guide = ChordProgressionGuide()
    hud = HolographicHUD()
    mouse = MouseController()
    camera = ThreadedCamera(camera_index=0, width=640, height=480, fps=60)

    waveform_idx = 0
    scale_idx = 0
    audio.set_waveform(WAVEFORMS[waveform_idx])

    window_flags = cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_GUI_NORMAL", 0)
    cv2.namedWindow("AetherSynth", window_flags)
    cv2.resizeWindow("AetherSynth", HUD_W, HUD_H)
    cv2.setMouseCallback("AetherSynth", mouse.on_mouse)

    screen = "MAIN_MENU"  # MAIN_MENU | HOW_TO_PLAY | CALIBRATION | STAGE
    right_mode = "CHORD"  # CHORD | MELODY
    active_note: Optional[int] = None
    active_chord_key: Optional[tuple] = None
    last_chord_det: Optional[ChordDetection] = None

    cutoff = 0.7
    pitch_bend = 0.0
    last_mod_time = 0.0
    fps = 30.0
    prev_frame_time = time.monotonic()

    # Espelho local da gravacao de base para HarmonyKeyDetector e Timeline
    local_loop_state = "idle"
    local_base_armed = False
    local_rec_start_beat = 0.0
    local_beat_accum = 0.0
    local_last_tick = time.monotonic()
    local_open_chord: Optional[Dict] = None
    loop_mirror_chords: List[Dict] = []
    detected_key = None

    # Estado de calibracao ao vivo
    cal_selected_cls = 1
    cal_recording = False
    cal_rec_count = 0
    cal_rec_target = 30
    cal_status = "Escolha uma classe e clique em GRAVAR AMOSTRAS."
    camera_error: Optional[str] = None

    def ensure_camera(active: bool):
        nonlocal camera_error
        if active:
            if not camera.is_ready():
                ok = camera.start()
                camera_error = None if ok else camera.error_message()
        else:
            camera.stop()
            camera_error = None

    def release_live_outputs():
        nonlocal active_note, active_chord_key, last_chord_det
        if active_note is not None:
            audio.note_off(active_note)
            active_note = None
        if active_chord_key is not None:
            audio.chord_off()
            active_chord_key = None
        last_chord_det = None
        chord_interp.reset()

    def trigger_loop_toggle():
        nonlocal local_loop_state, local_base_armed, local_beat_accum, local_last_tick
        nonlocal local_open_chord, loop_mirror_chords, detected_key, right_mode

        audio.loop_toggle()
        now_t = time.monotonic()
        if local_loop_state == "idle":
            loop_mirror_chords = []
            local_open_chord = None
            local_beat_accum = 0.0
            local_last_tick = now_t
            local_loop_state = "recording_base"
            if active_chord_key is not None and last_chord_det is not None:
                local_base_armed = False
                local_open_chord = {
                    "start_beat": 0.0,
                    "duration_beats": 0.25,
                    "root_pc": last_chord_det.root_pc,
                    "is_minor": last_chord_det.is_minor,
                    "short_label": last_chord_det.short_label,
                }
            else:
                local_base_armed = True
        elif local_loop_state == "recording_base":
            if local_base_armed and not loop_mirror_chords and local_open_chord is None:
                local_loop_state = "idle"
                local_base_armed = False
                return
            if local_open_chord is not None:
                local_open_chord["duration_beats"] = max(0.15, local_beat_accum - local_open_chord["start_beat"])
                loop_mirror_chords.append(local_open_chord)
                local_open_chord = None
            if not loop_mirror_chords:
                local_loop_state = "idle"
                local_base_armed = False
                return
            loop_mirror_chords, _ = _rescale_mirror_chords(loop_mirror_chords, local_beat_accum)
            local_loop_state = "playing"
            detected_key = harmony_ai.detect_from_events(loop_mirror_chords, locked=True)
            if detected_key is not None:
                keyboard.set_custom_scale(detected_key.key_name, detected_key.scale_notes, locked=True)
            release_live_outputs()
            right_mode = "MELODY"
        elif local_loop_state == "playing":
            local_loop_state = "overdubbing"
        elif local_loop_state == "overdubbing":
            local_loop_state = "playing"

    def trigger_loop_clear():
        nonlocal local_loop_state, local_base_armed, local_beat_accum, local_open_chord
        nonlocal loop_mirror_chords, detected_key, right_mode
        audio.loop_clear()
        local_loop_state = "idle"
        local_base_armed = False
        local_beat_accum = 0.0
        local_open_chord = None
        loop_mirror_chords = []
        detected_key = None
        keyboard.set_scale(SCALE_NAMES[scale_idx])
        keyboard.set_active_chord_tones(set())
        right_mode = "CHORD"

    def handle_action(action: str) -> bool:
        nonlocal screen, waveform_idx, scale_idx, right_mode
        nonlocal cal_selected_cls, cal_recording, cal_rec_count, cal_status

        if action == "btn_quit":
            return False
        elif action == "btn_menu":
            release_live_outputs()
            cal_recording = False
            ensure_camera(False)
            screen = "MAIN_MENU"
        elif action == "btn_start_loop":
            guide.stop()
            right_mode = "CHORD"
            ensure_camera(True)
            screen = "STAGE"
        elif action == "btn_start_challenge":
            guide.start_song(0)
            right_mode = "CHORD"
            ensure_camera(True)
            screen = "STAGE"
        elif action == "btn_how_to":
            ensure_camera(False)
            screen = "HOW_TO_PLAY"
        elif action == "btn_calibrate":
            release_live_outputs()
            cal_recording = False
            ensure_camera(True)
            screen = "CALIBRATION"
        elif action.startswith("wf_"):
            wf_name = action[3:]
            if wf_name in WAVEFORMS:
                waveform_idx = WAVEFORMS.index(wf_name)
                audio.set_waveform(WAVEFORMS[waveform_idx])
        elif action == "btn_waveform":
            waveform_idx = (waveform_idx + 1) % len(WAVEFORMS)
            audio.set_waveform(WAVEFORMS[waveform_idx])
        elif action == "btn_scale":
            if detected_key is None or not detected_key.locked:
                scale_idx = (scale_idx + 1) % len(SCALE_NAMES)
                keyboard.set_scale(SCALE_NAMES[scale_idx])
        elif action == "btn_loop_toggle":
            trigger_loop_toggle()
        elif action == "btn_loop_clear":
            trigger_loop_clear()
        elif action == "btn_mode_toggle":
            release_live_outputs()
            right_mode = "MELODY" if right_mode == "CHORD" else "CHORD"
        elif action == "btn_toggle_guide":
            if guide.is_active:
                guide.stop()
            else:
                right_mode = "CHORD"
                guide.start_song(guide.active_song_idx)
        elif action == "btn_next_song":
            guide.next_song()
        elif action == "btn_panic":
            release_live_outputs()
            audio.all_notes_off()
        elif action.startswith("cal_cls_"):
            cal_selected_cls = int(action.split("_")[-1])
            cal_recording = False
            cal_status = f"Classe {SHAPE_CLASSES[cal_selected_cls]} selecionada."
        elif action == "btn_cal_record":
            cal_recording = True
            cal_rec_count = 0
            cal_status = f"Mostre '{SHAPE_CLASSES[cal_selected_cls]}' para a camera..."
        elif action == "btn_cal_train":
            cal_recording = False
            acc = classifier.retrain_from_user_samples()
            cal_status = f"MLP retreinada! Acuracia: {acc * 100:.1f}%"
        elif action == "btn_cal_reset":
            cal_recording = False
            acc = classifier.reset_to_default_model()
            cal_status = f"Modelo padrao restaurado ({acc * 100:.1f}%)."
        return True

    try:
        running = True
        while running:
            now = time.monotonic()
            dt = max(now - prev_frame_time, 1e-4)
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
            prev_frame_time = now

            if now - last_ping_time >= 0.25:
                audio.ping()
                last_ping_time = now
            is_online = audio.is_online()
            engine_state = audio.engine_state

            # Atualiza destaque de chord tones no teclado baseado no loop Rust
            e_root = int(engine_state.get("active_chord_root", 255))
            e_qual = int(engine_state.get("active_chord_quality", 0))
            if 0 <= e_root <= 11:
                keyboard.set_active_chord_tones(chord_tones_for(e_root, e_qual == 1))
            elif last_chord_det is not None and last_chord_det.root_pc is not None:
                keyboard.set_active_chord_tones(chord_tones_for(last_chord_det.root_pc, last_chord_det.is_minor))
            else:
                keyboard.set_active_chord_tones(set())

            if screen == "MAIN_MENU":
                display_frame = hud.draw_main_menu(
                    HUD_W, HUD_H, now, mouse, is_online, classifier.val_accuracy, WAVEFORMS[waveform_idx]
                )
            elif screen == "HOW_TO_PLAY":
                display_frame = hud.draw_how_to_play(
                    HUD_W, HUD_H, now, mouse, WAVEFORMS[waveform_idx]
                )
            else:
                # Telas com camera (CALIBRATION ou STAGE)
                ret, raw_frame, _ = camera.read_latest() if camera.is_ready() else (False, None, 0)
                if ret and raw_frame is not None:
                    flipped = cv2.flip(raw_frame, 1)
                    display_frame = cv2.resize(flipped, (HUD_W, HUD_H), interpolation=cv2.INTER_LINEAR)
                    hands, _ = tracker.process_frame(flipped, target_w=HUD_W, target_h=HUD_H)
                else:
                    display_frame = hud._get_menu_backdrop(HUD_W, HUD_H, now)
                    hands = []

                if screen == "CALIBRATION":
                    if cal_recording and hands:
                        # Usa a mao do lado direito (ou primeira visivel) para calibracao
                        target_hand = max(hands, key=lambda h_item: h_item.center[0])
                        classifier.record_user_sample(cal_selected_cls, target_hand.landmarks_norm)
                        cal_rec_count += 1
                        if cal_rec_count >= cal_rec_target:
                            cal_recording = False
                            cal_status = f"30 amostras salvas p/ {SHAPE_CLASSES[cal_selected_cls]}. Clique TREINAR."
                    display_frame = hud.draw_calibration(
                        display_frame,
                        hands,
                        mouse,
                        classifier,
                        cal_selected_cls,
                        cal_recording,
                        cal_rec_count,
                        cal_rec_target,
                        cal_status,
                        camera_error,
                    )
                else:
                    # --- STAGE: Loop Session, Sinais de Acordes, Melodia & Pulso ---
                    guide.tick(now)
                    if local_loop_state == "recording_base" and not local_base_armed:
                        local_beat_accum += (now - local_last_tick) * (rhythm_det.bpm / 60.0)
                    local_last_tick = now

                    right_hand = None
                    left_hand = None
                    for hand in hands:
                        if hand.label == "Left":
                            right_hand = hand
                        else:
                            left_hand = hand

                    if len(hands) == 1:
                        h_only = hands[0]
                        if right_mode == "MELODY" and keyboard.check_hit(h_only, HUD_W, HUD_H) is not None:
                            right_hand = h_only
                            left_hand = None
                        elif h_only.center[0] >= HUD_W * 0.42:
                            right_hand = h_only
                            left_hand = None
                        else:
                            left_hand = h_only
                            right_hand = None

                    # Mao Direita: Acordes ou Melodia
                    if right_mode == "CHORD":
                        if active_note is not None:
                            audio.note_off(active_note)
                            active_note = None

                        det = chord_interp.update(right_hand)
                        last_chord_det = det
                        if guide.is_active:
                            guide.observe_chord(det, now=now)
                        new_key = (det.root_pc, det.is_minor) if (det is not None and det.root_pc is not None) else None

                        if new_key != active_chord_key:
                            if active_chord_key is not None:
                                audio.chord_off()
                                if local_loop_state == "recording_base" and not local_base_armed and local_open_chord is not None:
                                    local_open_chord["duration_beats"] = max(0.15, local_beat_accum - local_open_chord["start_beat"])
                                    loop_mirror_chords.append(local_open_chord)
                                    local_open_chord = None
                            if new_key is not None and det is not None:
                                audio.chord_on(det.root_pc, det.is_minor, det.midi_notes, velocity=0.85)
                                if local_loop_state == "recording_base":
                                    if local_base_armed:
                                        local_base_armed = False
                                        local_beat_accum = 0.0
                                    local_open_chord = {
                                        "start_beat": local_beat_accum,
                                        "duration_beats": 0.25,
                                        "root_pc": det.root_pc,
                                        "is_minor": det.is_minor,
                                        "short_label": det.short_label,
                                    }
                            active_chord_key = new_key
                    else:
                        if active_chord_key is not None:
                            audio.chord_off()
                            active_chord_key = None
                        last_chord_det = None

                        hit_note = keyboard.check_hit(right_hand, HUD_W, HUD_H) if right_hand is not None else None
                        if hit_note != active_note:
                            if active_note is not None:
                                audio.note_off(active_note)
                            if hit_note is not None and right_hand is not None:
                                vel = keyboard.compute_strike_velocity(right_hand)
                                audio.note_on(hit_note, velocity=vel)
                            active_note = hit_note

                    # Mao Esquerda: Cutoff + Pulso Ritmico + Pitch Bend (com zona morta)
                    if left_hand is not None:
                        cutoff = float(np.clip(left_hand.openness, 0.05, 1.0))
                        norm_y = left_hand.center[1] / HUD_H
                        raw_pb = (0.5 - norm_y) * 4.0
                        if abs(raw_pb) < 0.32:
                            pitch_bend = 0.0
                        else:
                            pitch_bend = float(np.clip(raw_pb, -2.0, 2.0))

                        pulsed, pulse_vel, bpm_updated, cur_bpm = rhythm_det.update(left_hand.palm_vy, now=now)
                        if pulsed:
                            audio.pulse(velocity=pulse_vel)
                        if bpm_updated:
                            audio.modulate(bpm=cur_bpm)

                        if now - last_mod_time > 0.02:
                            audio.modulate(cutoff=cutoff, pitch_bend=pitch_bend, volume=0.85, bpm=rhythm_det.bpm)
                            last_mod_time = now
                    else:
                        rhythm_det.update(None, now=now)
                        if abs(pitch_bend) > 0.05:
                            pitch_bend *= 0.75
                            audio.modulate(pitch_bend=pitch_bend)

                    display_frame = hud.draw_stage(
                        frame=display_frame,
                        mouse=mouse,
                        hands=hands,
                        right_hand=right_hand,
                        left_hand=left_hand,
                        keyboard=keyboard,
                        chord_det=last_chord_det,
                        detected_key=detected_key,
                        guide=guide,
                        active_note=active_note,
                        right_mode=right_mode,
                        engine_online=is_online,
                        engine_state=engine_state,
                        loop_mirror_chords=loop_mirror_chords,
                        current_waveform=WAVEFORMS[waveform_idx],
                        fps=fps,
                        cutoff=cutoff,
                        pitch_bend=pitch_bend,
                        bpm=rhythm_det.bpm,
                        pulse_glow=rhythm_det.get_pulse_glow(now=now),
                        camera_error=camera_error,
                        now=now,
                    )

            cv2.imshow("AetherSynth", display_frame)

            # Processa cliques do mouse
            for act in mouse.pop_clicks():
                if not handle_action(act):
                    running = False
                    break

            if not running:
                break

            # Fechamento pelo X da janela do sistema operacional
            if cv2.getWindowProperty("AetherSynth", cv2.WND_PROP_VISIBLE) < 1:
                break

            # Atalhos de teclado opcionais (secundarios ao mouse)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                if screen != "MAIN_MENU":
                    handle_action("btn_menu")
                else:
                    break
            elif screen == "STAGE":
                if key == ord(" "):
                    handle_action("btn_loop_toggle")
                elif key in (ord("m"), ord("M")):
                    handle_action("btn_mode_toggle")
                elif key in (ord("c"), ord("C")):
                    handle_action("btn_loop_clear")
                elif key in (ord("g"), ord("G")):
                    handle_action("btn_toggle_guide")
                elif key in (ord("k"), ord("K")):
                    handle_action("btn_calibrate")
                elif key in (ord("w"), ord("W")):
                    handle_action("btn_waveform")
                elif key in (ord("e"), ord("E")):
                    handle_action("btn_scale")
                elif key in (ord("r"), ord("R")):
                    handle_action("btn_panic")

    finally:
        print("\nEncerrando AetherSynth...")
        release_live_outputs()
        audio.all_notes_off()
        audio.close()
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import math
import os
import subprocess
import sys
import tempfile
import time
import unittest

import cv2
import numpy as np

VISION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(VISION_DIR)
if VISION_DIR not in sys.path:
    sys.path.insert(0, VISION_DIR)

from gestures import (
    ChordDetection,
    ChordInterpreter,
    HolographicVirtualKeyboard,
    MLPChordClassifier,
    SHAPE_CLASSES,
    _synthesize_hand_landmarks,
    extract_geometric_features,
)
from harmony_ai import HarmonyKeyDetector
from hud import HolographicHUD, MouseController, draw_hand_sign_icon
from rhythm import RhythmPulseDetector
from tracker import HandData
from tutor import ChordProgressionGuide
from udp_client import AudioEngineClient


def _make_hand_from_landmarks(pts: np.ndarray, label: str = "Left") -> HandData:
    lms_norm = [(float(p[0]), float(p[1]), float(p[2])) for p in pts]
    lms_px = [(int((p[0] * 0.25 + 0.5) * 1280), int((p[1] * 0.25 + 0.5) * 720)) for p in pts]
    return HandData(
        label=label,
        landmarks_norm=lms_norm,
        landmarks_px=lms_px,
        fingers_up=[True, True, True, True, True],
        openness=0.8,
        center=(800, 400),
        pinch_distance=0.1,
        tilt_angle=0.0,
        tip_velocity=0.5,
        palm_vy=0.0,
    )


def _upright_hand(class_idx: int, angle_deg: float = 90.0) -> np.ndarray:
    """Constroi mao canonica e gira no plano da imagem para o angulo alvo (+90 = cima)."""
    rng = np.random.default_rng(100 + class_idx)
    # Usa geometria base sem rotacao aleatoria para testar orientacao deterministica
    pts = np.zeros((21, 3), dtype=np.float32)
    # Reutiliza sintetizador e alinha o vetor direcional ao angulo pedido no plano XY
    raw = _synthesize_hand_landmarks(class_idx, rng)
    # Alinha plano XY removendo rotacao Z primeiro usando P9-P0 (ou P4-P2 para Si)
    vec = (raw[4] - raw[2]) if class_idx == 7 else (raw[9] - raw[0])
    # Rotaciona em 3D para que vec aponte na direcao (cos(a), -sin(a), 0) no frame
    src = vec / (np.linalg.norm(vec) + 1e-6)
    rad = math.radians(angle_deg)
    dst = np.array([math.cos(rad), -math.sin(rad), 0.0], dtype=np.float32)

    v = np.cross(src, dst)
    c = float(np.dot(src, dst))
    if np.linalg.norm(v) < 1e-5:
        R = np.eye(3, dtype=np.float32) if c > 0 else np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float32)
        R = np.eye(3, dtype=np.float32) + vx + (vx @ vx) * (1.0 / (1.0 + c))
    return (raw - raw[0]) @ R.T


class TestStageAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.TemporaryDirectory()
        cls.clf = MLPChordClassifier(models_dir=cls.tmp_dir.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp_dir.cleanup()

    def test_mlp_accuracy_latency_and_3d_rotation_invariance(self):
        self.assertGreaterEqual(self.clf.val_accuracy, 0.95)

        rng = np.random.default_rng(999)
        pts = _synthesize_hand_landmarks(4, rng)  # Fa
        f_orig = extract_geometric_features(pts)

        # Rotacao 90 e 180 graus em XY nao pode alterar features nem predicao
        rot90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        rot180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)
        f_90 = extract_geometric_features(pts @ rot90.T)
        f_180 = extract_geometric_features(pts @ rot180.T)
        np.testing.assert_allclose(f_orig, f_90, atol=1e-4)
        np.testing.assert_allclose(f_orig, f_180, atol=1e-4)

        idx0, _, _ = self.clf.predict_shape(pts)
        idx90, _, _ = self.clf.predict_shape(pts @ rot90.T)
        idx180, _, _ = self.clf.predict_shape(pts @ rot180.T)
        self.assertEqual(idx0, 4)
        self.assertEqual(idx90, 4)
        self.assertEqual(idx180, 4)

        # Benchmark de latencia NumPy forward pass
        t0 = time.perf_counter()
        for _ in range(500):
            self.clf.predict_shape(pts)
        ms_per_call = ((time.perf_counter() - t0) / 500.0) * 1000.0
        self.assertLess(ms_per_call, 0.5)

    def test_live_calibration_persistence(self):
        with tempfile.TemporaryDirectory() as d:
            c1 = MLPChordClassifier(models_dir=d)
            rng = np.random.default_rng(7)
            for _ in range(10):
                c1.record_user_sample(2, _synthesize_hand_landmarks(2, rng))
            acc = c1.retrain_from_user_samples()
            self.assertGreaterEqual(acc, 0.95)
            self.assertEqual(c1.user_sample_counts()[2], 10)

            c2 = MLPChordClassifier(models_dir=d)
            self.assertEqual(c2.user_sample_counts()[2], 10)

    def test_chord_interpreter_orientations_and_si_disambiguation(self):
        interp = ChordInterpreter(self.clf)
        cases = [
            (1, 90.0, 0, False, "C"),     # Do para cima -> C Maior
            (6, -90.0, 9, True, "Am"),   # La para baixo -> A Menor
            (4, 0.0, 6, False, "F#"),    # Fa para direita -> F# Maior
            (7, 180.0, 10, False, "Bb"), # Si para esquerda -> Bb Maior
            (7, 90.0, 11, False, "B"),   # Si para cima -> B Maior
        ]
        for shape_idx, ang, exp_root, exp_min, exp_cifra in cases:
            interp.reset()
            hand = _make_hand_from_landmarks(_upright_hand(shape_idx, ang))
            det = None
            for _ in range(3):
                det = interp.update(hand)
            self.assertIsNotNone(det)
            self.assertEqual(det.root_pc, exp_root, f"Falhou para {exp_cifra}")
            self.assertEqual(det.is_minor, exp_min)
            self.assertEqual(det.short_label, exp_cifra)

        # Punho -> Silencio
        interp.reset()
        fist_hand = _make_hand_from_landmarks(_upright_hand(0, 90.0))
        det_fist = None
        for _ in range(3):
            det_fist = interp.update(fist_hand)
        self.assertIsNone(det_fist.root_pc)
        self.assertEqual(det_fist.chord_label, "Silencio")

    def test_harmony_key_detector_locks_diatonic_scales(self):
        det = HarmonyKeyDetector()

        # C - G - Am - F -> Do Maior
        k1 = det.detect_from_events([
            {"root_pc": 0, "is_minor": False, "start_beat": 0.0, "duration_beats": 2.0},
            {"root_pc": 7, "is_minor": False, "start_beat": 2.0, "duration_beats": 2.0},
            {"root_pc": 9, "is_minor": True, "start_beat": 4.0, "duration_beats": 2.0},
            {"root_pc": 5, "is_minor": False, "start_beat": 6.0, "duration_beats": 2.0},
        ])
        self.assertEqual((k1.root_pc, k1.is_minor), (0, False))
        self.assertEqual({n % 12 for n in k1.scale_notes}, {0, 2, 4, 5, 7, 9, 11})

        # D - A - Bm - G -> Re Maior (contem F# e C#)
        k2 = det.detect_from_events([
            {"root_pc": 2, "is_minor": False, "start_beat": 0.0, "duration_beats": 2.0},
            {"root_pc": 9, "is_minor": False, "start_beat": 2.0, "duration_beats": 2.0},
            {"root_pc": 11, "is_minor": True, "start_beat": 4.0, "duration_beats": 2.0},
            {"root_pc": 7, "is_minor": False, "start_beat": 6.0, "duration_beats": 2.0},
        ])
        self.assertEqual((k2.root_pc, k2.is_minor), (2, False))
        self.assertEqual({n % 12 for n in k2.scale_notes}, {2, 4, 6, 7, 9, 11, 1})

        # Am - Dm - E - Am -> La Menor
        k3 = det.detect_from_events([
            {"root_pc": 9, "is_minor": True, "start_beat": 0.0, "duration_beats": 2.0},
            {"root_pc": 2, "is_minor": True, "start_beat": 2.0, "duration_beats": 2.0},
            {"root_pc": 4, "is_minor": False, "start_beat": 4.0, "duration_beats": 2.0},
            {"root_pc": 9, "is_minor": True, "start_beat": 6.0, "duration_beats": 2.0},
        ])
        self.assertEqual((k3.root_pc, k3.is_minor), (9, True))

        # F - Bb - Gm - C -> Fa Maior (contem Bb)
        k4 = det.detect_from_events([
            {"root_pc": 5, "is_minor": False, "start_beat": 0.0, "duration_beats": 2.0},
            {"root_pc": 10, "is_minor": False, "start_beat": 2.0, "duration_beats": 2.0},
            {"root_pc": 7, "is_minor": True, "start_beat": 4.0, "duration_beats": 2.0},
            {"root_pc": 0, "is_minor": False, "start_beat": 6.0, "duration_beats": 2.0},
        ])
        self.assertEqual((k4.root_pc, k4.is_minor), (5, False))
        self.assertIn(10, {n % 12 for n in k4.scale_notes})

    def test_rhythm_pulse_detector_converges_to_bpm(self):
        # 120 BPM = 0.5s entre pulsos
        r = RhythmPulseDetector(initial_bpm=80.0)
        t = 1.0
        for _ in range(8):
            r.update(-0.3, now=t)
            t += 0.05
            pulsed, vel, _, _ = r.update(0.8, now=t)
            self.assertTrue(pulsed)
            self.assertGreater(vel, 0.45)
            t += 0.45
        self.assertAlmostEqual(r.bpm, 120.0, delta=4.0)

        # 90 BPM = 0.6667s entre pulsos
        for _ in range(8):
            r.update(-0.2, now=t)
            t += 0.05
            r.update(0.75, now=t)
            t += 0.6167
        self.assertAlmostEqual(r.bpm, 90.0, delta=4.0)

    def test_hud_screens_and_mouse_clicks(self):
        hud = HolographicHUD()
        mouse = MouseController()
        kb = HolographicVirtualKeyboard()
        guide = ChordProgressionGuide()

        menu_img = hud.draw_main_menu(1280, 720, 0.5, mouse, True, 0.99, "saw")
        self.assertEqual(menu_img.shape, (720, 1280, 3))
        start_btn = next(b for b in mouse.buttons if b.id == "btn_start_loop")
        bx, by, bw, bh = start_btn.rect
        mouse.on_mouse(cv2.EVENT_LBUTTONDOWN, bx + bw // 2, by + bh // 2, 0)
        self.assertEqual(mouse.pop_clicks(), ["btn_start_loop"])

        how_img = hud.draw_how_to_play(1280, 720, 1.0, mouse, "saw")
        self.assertEqual(how_img.shape, (720, 1280, 3))

        cal_img = hud.draw_calibration(
            np.zeros((720, 1280, 3), dtype=np.uint8),
            [], mouse, self.clf, 1, False, 0, 30, "OK", None
        )
        self.assertEqual(cal_img.shape, (720, 1280, 3))

        stage_img = hud.draw_stage(
            frame=np.zeros((720, 1280, 3), dtype=np.uint8),
            mouse=mouse,
            hands=[],
            right_hand=None,
            left_hand=None,
            keyboard=kb,
            chord_det=None,
            detected_key=None,
            guide=guide,
            active_note=None,
            right_mode="CHORD",
            engine_online=True,
            engine_state={"loop_state": "playing", "beat_pos": 1.2, "loop_len_beats": 4.0},
            loop_mirror_chords=[{"start_beat": 0.0, "duration_beats": 4.0, "short_label": "C"}],
            current_waveform="saw",
            fps=30.0,
            cutoff=0.7,
            pitch_bend=0.0,
            bpm=100.0,
            pulse_glow=0.5,
        )
        self.assertEqual(stage_img.shape, (720, 1280, 3))
        self.assertTrue(any(b.id == "btn_loop_toggle" for b in mouse.buttons))

    def test_training_mode_celebrates_before_advancing_to_next_chord(self):
        guide = ChordProgressionGuide(celebration_duration=1.2)
        guide.start_song(0)  # step 0: C (root=0, is_minor=False, shape_idx=1)
        self.assertEqual(guide.phase, "WAITING")
        self.assertEqual(guide.get_display_step().cifra, "C")

        det_c = ChordDetection(1, "Do", 0.96, "up", 0, False, "Do Maior", "C", [48, 60, 64, 67], "up")
        det_wrong = ChordDetection(2, "Re", 0.95, "up", 2, False, "Re Maior", "D", [50, 62, 66, 69], "up")

        # Acerto entra em CELEBRATING e NAO avanca imediatamente para o proximo acorde
        guide.observe_chord(det_c, now=10.0)
        self.assertEqual(guide.phase, "CELEBRATING")
        self.assertEqual(guide.current_step, 0)
        self.assertEqual(guide.get_display_step().cifra, "C")
        self.assertIn("ACERTOU", guide.feedback_message)
        self.assertEqual(guide.streak, 1)

        # Durante a janela de feedback, outro gesto nao quebra streak nem avanca antes da hora
        guide.observe_chord(det_wrong, now=10.6)
        self.assertEqual(guide.phase, "CELEBRATING")
        self.assertEqual(guide.current_step, 0)
        self.assertEqual(guide.streak, 1)

        # Apos completar o tempo de feedback, avanca para G (step 1)
        advanced = guide.tick(now=11.3)
        self.assertTrue(advanced)
        self.assertEqual(guide.phase, "WAITING")
        self.assertEqual(guide.current_step, 1)
        self.assertEqual(guide.get_display_step().cifra, "G")
        self.assertEqual(guide.get_display_step().shape_idx, 5)

    def test_hand_sign_icon_and_training_hud_render(self):
        for shape_idx in range(8):
            for orient in ("up", "down", "right", "left"):
                for ok in (False, True):
                    canvas = np.zeros((160, 160, 3), dtype=np.uint8)
                    draw_hand_sign_icon(canvas, (80, 80), 48, shape_idx, orient, success=ok)
                    self.assertGreater(int(np.sum(canvas)), 0)

        hud = HolographicHUD()
        mouse = MouseController()
        kb = HolographicVirtualKeyboard()
        guide = ChordProgressionGuide(celebration_duration=1.2)
        guide.start_song(0)

        # Renderiza fase WAITING e fase CELEBRATING
        img_wait = hud.draw_stage(
            frame=np.zeros((720, 1280, 3), dtype=np.uint8),
            mouse=mouse, hands=[], right_hand=None, left_hand=None,
            keyboard=kb, chord_det=None, detected_key=None, guide=guide,
            active_note=None, right_mode="CHORD", engine_online=True,
            engine_state={"loop_state": "idle"}, loop_mirror_chords=[],
            current_waveform="saw", fps=30.0, cutoff=0.7, pitch_bend=0.0,
            bpm=96.0, pulse_glow=0.0, now=10.0,
        )
        self.assertEqual(img_wait.shape, (720, 1280, 3))

        det_c = ChordDetection(1, "Do", 0.96, "up", 0, False, "Do Maior", "C", [48, 60, 64, 67], "up")
        guide.observe_chord(det_c, now=10.0)
        img_cel = hud.draw_stage(
            frame=np.zeros((720, 1280, 3), dtype=np.uint8),
            mouse=mouse, hands=[], right_hand=None, left_hand=None,
            keyboard=kb, chord_det=det_c, detected_key=None, guide=guide,
            active_note=None, right_mode="CHORD", engine_online=True,
            engine_state={"loop_state": "idle"}, loop_mirror_chords=[],
            current_waveform="saw", fps=30.0, cutoff=0.7, pitch_bend=0.0,
            bpm=96.0, pulse_glow=0.0, now=10.5,
        )
        self.assertEqual(img_cel.shape, (720, 1280, 3))

    def test_rust_udp_roundtrip_headless(self):
        rust_bin = os.path.join(PROJECT_DIR, "audio_engine", "target", "release", "audio_engine")
        if not os.path.exists(rust_bin):
            self.skipTest("Rust release binary nao compilado ainda")

        port = 18765
        env = os.environ.copy()
        env["HEADLESS_AUDIO"] = "1"
        env["AUDIO_BIND_ADDR"] = f"127.0.0.1:{port}"
        proc = subprocess.Popen([rust_bin], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        client = AudioEngineClient(host="127.0.0.1", port=port)
        try:
            online = False
            for _ in range(25):
                client.ping()
                time.sleep(0.05)
                if client.is_online(timeout=2.0):
                    online = True
                    break
            self.assertTrue(online, "Motor Rust headless nao respondeu ping")

            client.modulate(bpm=110.0, cutoff=0.8)
            client.loop_toggle()  # idle -> recording_base
            client.chord_on(0, False, [48, 60, 64, 67], 0.85)
            time.sleep(0.6)
            client.pulse(0.9)
            client.loop_toggle()  # recording_base -> playing
            time.sleep(0.15)
            client.ping()
            time.sleep(0.05)
            st = client.poll_status()
            self.assertEqual(st.get("loop_state"), "playing")
            self.assertGreaterEqual(float(st.get("loop_len_beats", 0.0)), 4.0)
            self.assertAlmostEqual(float(st.get("bpm", 0.0)), 110.0, delta=1.0)
        finally:
            client.close()
            proc.terminate()
            proc.wait(timeout=3)


if __name__ == "__main__":
    unittest.main()

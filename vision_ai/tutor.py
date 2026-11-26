import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from gestures import ChordDetection


@dataclass
class ProgressionStep:
    root_pc: int
    is_minor: bool
    cifra: str
    shape_idx: int  # 1=Do, 2=Re, 3=Mi, 4=Fa, 5=Sol, 6=La, 7=Si
    shape_hint: str
    arrow_dir: str  # "up" | "down" | "right" | "left"
    beats: float = 4.0


@dataclass
class ProgressionSong:
    title: str
    bpm: int
    steps: List[ProgressionStep]


PROGRESSIONS = [
    ProgressionSong(
        title="Pop I - V - vi - IV (Do Maior)",
        bpm=92,
        steps=[
            ProgressionStep(0, False, "C", 1, "1 dedo (Indicador)", "up", 4.0),
            ProgressionStep(7, False, "G", 5, "5 dedos abertos", "up", 4.0),
            ProgressionStep(9, True, "Am", 6, "Hang-loose p/ baixo", "down", 4.0),
            ProgressionStep(5, False, "F", 4, "4 dedos (sem polegar)", "up", 4.0),
        ],
    ),
    ProgressionSong(
        title="Groove vi - IV - I - V (La Menor)",
        bpm=96,
        steps=[
            ProgressionStep(9, True, "Am", 6, "Hang-loose p/ baixo", "down", 4.0),
            ProgressionStep(5, False, "F", 4, "4 dedos (sem polegar)", "up", 4.0),
            ProgressionStep(0, False, "C", 1, "1 dedo (Indicador)", "up", 4.0),
            ProgressionStep(7, False, "G", 5, "5 dedos abertos", "up", 4.0),
        ],
    ),
    ProgressionSong(
        title="Jazz ii - V - I - vi",
        bpm=88,
        steps=[
            ProgressionStep(2, True, "Dm", 2, "2 dedos p/ baixo", "down", 4.0),
            ProgressionStep(7, False, "G", 5, "5 dedos abertos", "up", 4.0),
            ProgressionStep(0, False, "C", 1, "1 dedo (Indicador)", "up", 4.0),
            ProgressionStep(9, True, "Am", 6, "Hang-loose p/ baixo", "down", 4.0),
        ],
    ),
    ProgressionSong(
        title="Acidentes F - Bb - Gm - C",
        bpm=90,
        steps=[
            ProgressionStep(5, False, "F", 4, "4 dedos (sem polegar)", "up", 4.0),
            ProgressionStep(10, False, "Bb", 7, "Polegar p/ esquerda", "left", 4.0),
            ProgressionStep(7, True, "Gm", 5, "5 dedos p/ baixo", "down", 4.0),
            ProgressionStep(0, False, "C", 1, "1 dedo (Indicador)", "up", 4.0),
        ],
    ),
]


class ChordProgressionGuide:
    """
    Modo Treino Inteligente de Sinais de Acordes.
    Mostra o icone do sinal esperado, celebra o acerto na tela e so avanca apos o feedback.
    """

    def __init__(self, celebration_duration: float = 1.25):
        self.active_song_idx = 0
        self.song = PROGRESSIONS[0]
        self.current_step = 0
        self.is_active = False

        self.phase = "WAITING"  # "WAITING" | "CELEBRATING"
        self.celebration_duration = float(celebration_duration)
        self.celebrate_started_at = 0.0
        self.celebrate_until = 0.0
        self.celebrated_step: Optional[ProgressionStep] = None

        self.correct_steps = 0
        self.total_attempts = 0
        self.streak = 0
        self.score = 0
        self.adaptive_bpm = float(self.song.bpm)
        self.transition_errors: Dict[Tuple[str, str], int] = {}
        self.last_step_time = 0.0
        self.last_cifra = "Inicio"
        self.feedback_message = "Modo Treino: faca o sinal mostrado para comecar."
        self.feedback_type = "info"

        self._wrong_key: Optional[Tuple[int, bool]] = None
        self._wrong_since = 0.0
        self._last_penalized_key: Optional[Tuple[int, bool]] = None

    def start_song(self, song_idx: int = 0):
        self.active_song_idx = song_idx % len(PROGRESSIONS)
        self.song = PROGRESSIONS[self.active_song_idx]
        self.current_step = 0
        self.phase = "WAITING"
        self.celebrated_step = None
        self.celebrate_started_at = 0.0
        self.celebrate_until = 0.0
        self.correct_steps = 0
        self.total_attempts = 0
        self.streak = 0
        self.score = 0
        self.adaptive_bpm = float(self.song.bpm)
        self.transition_errors = {}
        self.is_active = True
        self.last_step_time = time.monotonic()
        self.last_cifra = "Inicio"
        self._wrong_key = None
        self._last_penalized_key = None
        target = self.get_expected_step()
        self.feedback_message = f"Faca {target.cifra}: {target.shape_hint}"
        self.feedback_type = "info"

    def next_song(self):
        self.start_song((self.active_song_idx + 1) % len(PROGRESSIONS))

    def stop(self):
        self.is_active = False
        self.phase = "WAITING"
        self.celebrated_step = None
        self._wrong_key = None
        self._last_penalized_key = None
        self.feedback_message = "Modo Livre ativo."
        self.feedback_type = "neutral"

    def get_expected_step(self) -> Optional[ProgressionStep]:
        if not self.is_active or self.current_step >= len(self.song.steps):
            return None
        return self.song.steps[self.current_step]

    def get_display_step(self) -> Optional[ProgressionStep]:
        if self.phase == "CELEBRATING" and self.celebrated_step is not None:
            return self.celebrated_step
        return self.get_expected_step()

    def celebration_progress(self, now: Optional[float] = None) -> float:
        if self.phase != "CELEBRATING" or self.celebration_duration <= 0.0:
            return 0.0
        if now is None:
            now = time.monotonic()
        return max(0.0, min(1.0, (now - self.celebrate_started_at) / self.celebration_duration))

    def tick(self, now: Optional[float] = None) -> bool:
        """Avanca para o proximo acorde somente apos concluir o tempo de feedback do acerto."""
        if not self.is_active or self.phase != "CELEBRATING":
            return False
        if now is None:
            now = time.monotonic()
        if now >= self.celebrate_until:
            self.phase = "WAITING"
            self.celebrated_step = None
            self.current_step = (self.current_step + 1) % len(self.song.steps)
            self._wrong_key = None
            self._last_penalized_key = None
            self.last_step_time = now
            nxt = self.get_expected_step()
            if nxt is not None:
                self.feedback_message = f"Agora faca {nxt.cifra}: {nxt.shape_hint}"
            self.feedback_type = "info"
            return True
        return False

    def observe_chord(self, detection: Optional[ChordDetection], now: Optional[float] = None):
        """Monitora continuamente os gestos no palco sem penalizar transicoes rapidas."""
        if now is None:
            now = time.monotonic()
        self.tick(now)
        if not self.is_active or self.phase == "CELEBRATING":
            return

        if detection is None or detection.root_pc is None:
            self._wrong_key = None
            self._last_penalized_key = None
            return

        target = self.get_expected_step()
        if target is None:
            return

        key = (detection.root_pc, detection.is_minor)
        if key == (target.root_pc, target.is_minor):
            self._wrong_key = None
            self._last_penalized_key = None
            self.register_chord(detection, now=now)
            return

        if key != self._wrong_key:
            self._wrong_key = key
            self._wrong_since = now
            return

        if (now - self._wrong_since) >= 0.35 and key != self._last_penalized_key:
            self._last_penalized_key = key
            self.register_chord(detection, now=now)

    def register_chord(self, detection: ChordDetection, now: Optional[float] = None) -> Tuple[bool, str]:
        if not self.is_active or detection is None or detection.root_pc is None:
            return True, ""

        if self.phase == "CELEBRATING":
            return True, self.feedback_message

        target = self.get_expected_step()
        if target is None:
            return True, ""

        if now is None:
            now = time.monotonic()

        self.total_attempts += 1

        if detection.root_pc == target.root_pc and detection.is_minor == target.is_minor:
            elapsed = max(0.1, now - self.last_step_time)
            ideal_sec = (60.0 / max(40.0, self.adaptive_bpm)) * target.beats
            timing_err = abs(elapsed - ideal_sec) / ideal_sec

            self.correct_steps += 1
            self.streak += 1
            bonus = 150 if timing_err <= 0.35 else 100
            self.score += bonus + self.streak * 20

            if self.streak >= 3 and timing_err <= 0.30:
                self.adaptive_bpm = min(140.0, self.adaptive_bpm + 2.0)

            self.last_cifra = target.cifra
            self.phase = "CELEBRATING"
            self.celebrated_step = target
            self.celebrate_started_at = now
            self.celebrate_until = now + self.celebration_duration

            if self.current_step == len(self.song.steps) - 1:
                acc = self.get_accuracy()
                self.feedback_message = f"ACERTOU {target.cifra}! Progressao completa ({acc}%)!"
            else:
                self.feedback_message = f"ACERTOU {target.cifra}! Excelente sinal!"
            self.feedback_type = "success"
            return True, self.feedback_message

        # Erro harmonico
        self.streak = 0
        key = (self.last_cifra, target.cifra)
        self.transition_errors[key] = self.transition_errors.get(key, 0) + 1
        self.adaptive_bpm = max(60.0, self.adaptive_bpm - 2.5)

        if self.transition_errors[key] >= 2:
            self.feedback_message = (
                f"Dica IA: {self.last_cifra}->{target.cifra} falhou "
                f"{self.transition_errors[key]}x. Tente {target.shape_hint}."
            )
        else:
            self.feedback_message = f"Voce fez {detection.short_label}. Esperado: {target.cifra}."
        self.feedback_type = "warning"
        return False, self.feedback_message

    def get_accuracy(self) -> int:
        if self.total_attempts == 0:
            return 100
        return int((self.correct_steps / self.total_attempts) * 100)


# Alias de compatibilidade
IntelligentMusicTutor = ChordProgressionGuide
SONGS = PROGRESSIONS

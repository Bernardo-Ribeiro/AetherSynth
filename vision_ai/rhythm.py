import time
from typing import Optional, Tuple


class RhythmPulseDetector:
    """
    Detector cinematico de pulso ritmico na mao esquerda.
    Detecta batidas verticais descendentes (tap-tempo no ar) e adapta o BPM suavemente.
    """

    def __init__(self, initial_bpm: float = 96.0):
        self.bpm = float(initial_bpm)
        self.armed = True
        self.last_pulse_time: Optional[float] = None
        self.pulse_count = 0

    def reset(self):
        self.armed = True
        self.last_pulse_time = None
        self.pulse_count = 0

    def update(
        self, palm_vy: Optional[float], now: Optional[float] = None
    ) -> Tuple[bool, float, bool, float]:
        """
        Recebe palm_vy (norm/s, positivo para baixo).
        Retorna (pulsed, pulse_velocity, bpm_updated, current_bpm).
        """
        if now is None:
            now = time.monotonic()

        if palm_vy is None:
            self.armed = True
            return False, 0.0, False, self.bpm

        # Rearma quando a mao desacelera ou sobe
        if palm_vy < 0.08:
            self.armed = True

        # Dispara pulso descendente nitido respeitando tempo refratario de 220ms
        if self.armed and palm_vy > 0.42:
            if self.last_pulse_time is not None and (now - self.last_pulse_time) < 0.22:
                return False, 0.0, False, self.bpm

            self.armed = False
            self.pulse_count += 1
            pulse_vel = max(0.45, min(1.0, 0.45 + (palm_vy - 0.42) * 0.45))
            bpm_updated = False

            if self.last_pulse_time is not None:
                dt = now - self.last_pulse_time
                if 0.333 <= dt <= 1.200:
                    bpm_inst = 60.0 / dt
                    self.bpm = max(50.0, min(180.0, 0.65 * self.bpm + 0.35 * bpm_inst))
                    bpm_updated = True

            self.last_pulse_time = now
            return True, pulse_vel, bpm_updated, self.bpm

        return False, 0.0, False, self.bpm

    def get_pulse_glow(self, now: Optional[float] = None) -> float:
        if self.last_pulse_time is None:
            return 0.0
        if now is None:
            now = time.monotonic()
        elapsed = now - self.last_pulse_time
        if elapsed < 0.0 or elapsed > 0.25:
            return 0.0
        return max(0.0, 1.0 - elapsed / 0.25)

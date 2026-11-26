import json
import socket
import time
from typing import Dict, List, Optional


class AudioEngineClient:
    """Cliente UDP non-blocking para o Rust Lock-Free Audio & Loop Engine."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self._last_pong: Optional[float] = None
        self.engine_state: Dict = {
            "loop_state": "idle",
            "bpm": 96.0,
            "beat_pos": 0.0,
            "loop_len_beats": 0.0,
            "active_chord_root": 255,
            "active_chord_quality": 0,
        }

    def _send(self, payload: dict):
        try:
            data = json.dumps(payload).encode("utf-8")
            self.sock.sendto(data, self.addr)
        except Exception:
            pass

    def note_on(self, note: int, velocity: float = 0.85):
        self._send({"type": "note_on", "note": int(note), "velocity": float(velocity)})

    def note_off(self, note: int):
        self._send({"type": "note_off", "note": int(note)})

    def chord_on(self, root: int, is_minor: bool, notes: List[int], velocity: float = 0.85):
        self._send({
            "type": "chord_on",
            "root": int(root),
            "quality": "minor" if is_minor else "major",
            "notes": [int(n) for n in notes],
            "velocity": float(velocity),
        })

    def chord_off(self):
        self._send({"type": "chord_off"})

    def pulse(self, velocity: float = 0.85):
        self._send({"type": "pulse", "velocity": float(velocity)})

    def loop_toggle(self):
        self._send({"type": "loop_toggle"})

    def loop_clear(self):
        self._send({"type": "loop_clear"})

    def all_notes_off(self):
        self._send({"type": "all_notes_off"})

    def modulate(
        self,
        cutoff: Optional[float] = None,
        pitch_bend: Optional[float] = None,
        volume: Optional[float] = None,
        resonance: Optional[float] = None,
        bpm: Optional[float] = None,
    ):
        payload: Dict[str, object] = {"type": "modulate"}
        if cutoff is not None:
            payload["cutoff"] = float(cutoff)
        if pitch_bend is not None:
            payload["pitch_bend"] = float(pitch_bend)
        if volume is not None:
            payload["volume"] = float(volume)
        if resonance is not None:
            payload["resonance"] = float(resonance)
        if bpm is not None:
            payload["bpm"] = float(bpm)
        self._send(payload)

    def set_waveform(self, waveform: str):
        self._send({"type": "set_waveform", "waveform": waveform.lower()})

    def ping(self):
        self._send({"type": "ping"})

    def poll_status(self, timeout: float = 4.0) -> Dict:
        """Drena pacotes pong recebidos e atualiza telemetria local do looper."""
        try:
            while True:
                data, addr = self.sock.recvfrom(2048)
                if addr == self.addr and b"pong" in data:
                    self._last_pong = time.monotonic()
                    try:
                        parsed = json.loads(data.decode("utf-8"))
                        if isinstance(parsed, dict):
                            self.engine_state.update(parsed)
                    except Exception:
                        pass
        except BlockingIOError:
            pass
        return self.engine_state

    def is_online(self, timeout: float = 4.0) -> bool:
        self.poll_status(timeout=timeout)
        return self._last_pong is not None and (time.monotonic() - self._last_pong) < timeout

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass

import json
import socket
import time
from typing import Optional


class AudioEngineClient:
    """Cliente UDP para comunicação de ultra-baixa latência com o Rust Audio Engine."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self._last_pong = None

    def _send(self, payload: dict):
        try:
            data = json.dumps(payload).encode("utf-8")
            self.sock.sendto(data, self.addr)
        except Exception as e:
            # UDP non-blocking: não interrompe a visão caso ocorra falha momentânea
            pass

    def note_on(self, note: int, velocity: float = 0.8):
        """Dispara uma nota (note: número MIDI 0-127, ex: 60 = C4)."""
        self._send({"type": "note_on", "note": int(note), "velocity": float(velocity)})

    def note_off(self, note: int):
        """Desliga uma nota."""
        self._send({"type": "note_off", "note": int(note)})

    def all_notes_off(self):
        """Desliga todas as notas ativas."""
        self._send({"type": "all_notes_off"})

    def modulate(
        self,
        cutoff: Optional[float] = None,
        pitch_bend: Optional[float] = None,
        volume: Optional[float] = None,
        resonance: Optional[float] = None,
    ):
        """
        Modula os parâmetros do sintetizador em tempo real:
        - cutoff: 0.0 (fechado/abafado) a 1.0 (aberto/brilhante)
        - pitch_bend: semitons (-12.0 a +12.0 ou frações contínuas)
        - volume: 0.0 a 1.0
        - resonance: 0.0 a 0.95
        """
        payload = {"type": "modulate"}
        if cutoff is not None:
            payload["cutoff"] = float(cutoff)
        if pitch_bend is not None:
            payload["pitch_bend"] = float(pitch_bend)
        if volume is not None:
            payload["volume"] = float(volume)
        if resonance is not None:
            payload["resonance"] = float(resonance)

        self._send(payload)

    def set_waveform(self, waveform: str):
        """Define a forma de onda: 'sine', 'saw', 'square', 'triangle'."""
        self._send({"type": "set_waveform", "waveform": waveform.lower()})

    def ping(self):
        """Solicita uma resposta sem interromper o processamento de vídeo."""
        self._send({"type": "ping"})

    def is_online(self, timeout: float = 4.0) -> bool:
        """Recebe respostas pendentes e verifica a última resposta válida."""
        try:
            while True:
                data, addr = self.sock.recvfrom(1024)
                if addr == self.addr and b"pong" in data:
                    self._last_pong = time.monotonic()
        except BlockingIOError:
            pass
        return self._last_pong is not None and time.monotonic() - self._last_pong < timeout

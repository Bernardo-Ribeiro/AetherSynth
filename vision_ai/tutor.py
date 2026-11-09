import time
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class TutorNote:
    note: int  # MIDI pitch
    name: str  # Nome (ex: "Dó4", "Mi4")
    duration: float  # Duração ideal em batidas (ex: 1.0 = semínima)


@dataclass
class Song:
    title: str
    bpm: int
    notes: List[TutorNote]


# Repertório didático
SONGS = [
    Song(
        title="Ode a Alegria (Beethoven)",
        bpm=90,
        notes=[
            TutorNote(64, "Mi4", 1.0),
            TutorNote(64, "Mi4", 1.0),
            TutorNote(65, "Fa4", 1.0),
            TutorNote(67, "Sol4", 1.0),
            TutorNote(67, "Sol4", 1.0),
            TutorNote(65, "Fa4", 1.0),
            TutorNote(64, "Mi4", 1.0),
            TutorNote(62, "Re4", 1.0),
            TutorNote(60, "Do4", 1.0),
            TutorNote(60, "Do4", 1.0),
            TutorNote(62, "Re4", 1.0),
            TutorNote(64, "Mi4", 1.0),
            TutorNote(64, "Mi4", 1.5),
            TutorNote(62, "Re4", 0.5),
            TutorNote(62, "Re4", 2.0),
        ],
    ),
    Song(
        title="Brilha Brilha Estrelinha",
        bpm=80,
        notes=[
            TutorNote(60, "Do4", 1.0),
            TutorNote(60, "Do4", 1.0),
            TutorNote(67, "Sol4", 1.0),
            TutorNote(67, "Sol4", 1.0),
            TutorNote(69, "La4", 1.0),
            TutorNote(69, "La4", 1.0),
            TutorNote(67, "Sol4", 2.0),
            TutorNote(65, "Fa4", 1.0),
            TutorNote(65, "Fa4", 1.0),
            TutorNote(64, "Mi4", 1.0),
            TutorNote(64, "Mi4", 1.0),
            TutorNote(62, "Re4", 1.0),
            TutorNote(62, "Re4", 1.0),
            TutorNote(60, "Do4", 2.0),
        ],
    ),
    Song(
        title="Asa Branca (Introducao)",
        bpm=100,
        notes=[
            TutorNote(60, "Do4", 0.5),
            TutorNote(62, "Re4", 0.5),
            TutorNote(64, "Mi4", 1.0),
            TutorNote(67, "Sol4", 1.0),
            TutorNote(67, "Sol4", 1.0),
            TutorNote(64, "Mi4", 1.0),
            TutorNote(65, "Fa4", 1.0),
            TutorNote(65, "Fa4", 2.0),
        ],
    ),
]


class IntelligentMusicTutor:
    """
    Tutor Inteligente que monitora o aprendizado do usuário,
    calcula acurácia rítmica, diagnostica dificuldades e fornece dicas pedagógicas.
    """

    def __init__(self):
        self.active_song_idx = 0
        self.song = SONGS[0]
        self.current_step = 0
        self.is_active = False

        # Estatísticas do aluno
        self.correct_notes = 0
        self.total_attempts = 0
        self.streak = 0
        self.error_history = {}  # { note_name: error_count }
        self.last_note_time = 0.0
        self.feedback_message = "Bem-vindo ao Modo Tutor! Toque a primeira nota para comecar."
        self.feedback_type = "neutral"  # "success", "warning", "info", "neutral"
        self.adaptive_bpm = self.song.bpm

    def start_song(self, song_idx: int = 0):
        self.active_song_idx = song_idx % len(SONGS)
        self.song = SONGS[self.active_song_idx]
        self.current_step = 0
        self.correct_notes = 0
        self.total_attempts = 0
        self.streak = 0
        self.error_history = {}
        self.adaptive_bpm = self.song.bpm
        self.is_active = True
        self.last_note_time = time.time()
        self.feedback_message = f"Musica selecionada: {self.song.title}. Toque {self.get_expected_note().name}!"
        self.feedback_type = "info"

    def stop(self):
        self.is_active = False
        self.feedback_message = "Modo Livre (Tutor desativado)."
        self.feedback_type = "neutral"

    def get_expected_note(self) -> Optional[TutorNote]:
        if not self.is_active or self.current_step >= len(self.song.notes):
            return None
        return self.song.notes[self.current_step]

    def register_played_note(self, note: int) -> Tuple[bool, str]:
        """Avalia a nota tocada e avança na partitura."""
        if not self.is_active or self.current_step >= len(self.song.notes):
            return True, ""

        now = time.time()
        expected = self.song.notes[self.current_step]
        self.total_attempts += 1

        if note == expected.note:
            self.correct_notes += 1
            self.streak += 1
            self.current_step += 1

            # Análise rítmica e de consistência
            if self.streak >= 5:
                self.feedback_message = f"Excelente! {self.streak} notas seguidas sem errar!"
                self.feedback_type = "success"
            elif self.streak >= 3:
                self.feedback_message = "Muito bom! Continue no mesmo ritmo."
                self.feedback_type = "success"
            else:
                self.feedback_message = f"Correto ({expected.name})! Proxima: {self._next_note_name()}"
                self.feedback_type = "success"

            # Conclusão da música
            if self.current_step >= len(self.song.notes):
                acc = int((self.correct_notes / self.total_attempts) * 100)
                self.feedback_message = f"Parabens! Musica concluida com {acc}% de precisao!"
                self.feedback_type = "success"
                self.is_active = False
                return True, self.feedback_message

            self.last_note_time = now
            return True, self.feedback_message
        else:
            # Errou a nota
            self.streak = 0
            note_err = expected.name
            self.error_history[note_err] = self.error_history.get(note_err, 0) + 1

            # Dica adaptativa da IA baseada no padrão de erros
            if self.error_history[note_err] >= 2:
                self.feedback_message = (
                    f"Atencao: voce errou '{note_err}' {self.error_history[note_err]}x. "
                    "Posicione o dedo com calma sobre a esfera correta."
                )
                self.feedback_type = "warning"
            else:
                self.feedback_message = f"Nota incorreta! A nota esperada era {expected.name}."
                self.feedback_type = "warning"

            return False, self.feedback_message

    def _next_note_name(self) -> str:
        if self.current_step < len(self.song.notes):
            return self.song.notes[self.current_step].name
        return "Fim"

    def get_accuracy(self) -> int:
        if self.total_attempts == 0:
            return 100
        return int((self.correct_notes / self.total_attempts) * 100)

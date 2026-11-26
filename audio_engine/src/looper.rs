use crate::synth::{PolySynth, VoiceRole};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum LoopState {
    Idle = 0,
    RecordingBase = 1,
    Playing = 2,
    Overdubbing = 3,
}

#[derive(Debug, Clone)]
pub struct ChordLoopEvent {
    pub start_beat: f32,
    pub duration_beats: f32,
    pub root: u8,
    pub quality: u8,
    pub notes: Vec<u8>,
    pub velocity: f32,
}

#[derive(Debug, Clone)]
pub struct MelodyLoopEvent {
    pub start_beat: f32,
    pub duration_beats: f32,
    pub note: u8,
    pub velocity: f32,
}

pub struct LoopEngine {
    pub state: LoopState,
    pub bpm: f32,
    pub beat_pos: f32,
    pub loop_len_beats: f32,
    pub base_armed: bool,
    pub chord_events: Vec<ChordLoopEvent>,
    pub melody_events: Vec<MelodyLoopEvent>,
    open_chord: Option<ChordLoopEvent>,
    open_melody: Option<(u8, f32, f32)>,
    playing_chord_idx: Option<usize>,
    playing_melody_indices: Vec<usize>,
    live_chord_info: Option<(u8, u8, Vec<u8>, f32)>,
}

impl LoopEngine {
    pub fn new() -> Self {
        Self {
            state: LoopState::Idle,
            bpm: 96.0,
            beat_pos: 0.0,
            loop_len_beats: 0.0,
            base_armed: false,
            chord_events: Vec::new(),
            melody_events: Vec::new(),
            open_chord: None,
            open_melody: None,
            playing_chord_idx: None,
            playing_melody_indices: Vec::new(),
            live_chord_info: None,
        }
    }

    pub fn on_live_chord_on(&mut self, root: u8, quality: u8, notes: &[u8], velocity: f32) {
        self.live_chord_info = Some((root, quality, notes.to_vec(), velocity));
        if self.state == LoopState::RecordingBase {
            if self.base_armed {
                self.base_armed = false;
                self.beat_pos = 0.0;
            }
            if let Some(mut open) = self.open_chord.take() {
                open.duration_beats = (self.beat_pos - open.start_beat).max(0.12);
                self.chord_events.push(open);
            }
            self.open_chord = Some(ChordLoopEvent {
                start_beat: self.beat_pos,
                duration_beats: 0.25,
                root,
                quality,
                notes: notes.to_vec(),
                velocity,
            });
        }
    }

    pub fn on_live_chord_off(&mut self) {
        self.live_chord_info = None;
        if self.state == LoopState::RecordingBase && !self.base_armed {
            if let Some(mut open) = self.open_chord.take() {
                open.duration_beats = (self.beat_pos - open.start_beat).max(0.12);
                self.chord_events.push(open);
            }
        }
    }

    pub fn on_live_note_on(&mut self, note: u8, velocity: f32) {
        if self.state == LoopState::Overdubbing && self.loop_len_beats >= 4.0 {
            if let Some((prev_note, start_beat, vel)) = self.open_melody.take() {
                self.push_melody_event(prev_note, start_beat, self.beat_pos, vel);
            }
            self.open_melody = Some((note, self.beat_pos, velocity));
        }
    }

    pub fn on_live_note_off(&mut self, note: u8) {
        if self.state == LoopState::Overdubbing && self.loop_len_beats >= 4.0 {
            if let Some((open_note, start_beat, vel)) = self.open_melody {
                if open_note == note {
                    self.open_melody = None;
                    self.push_melody_event(open_note, start_beat, self.beat_pos, vel);
                }
            }
        }
    }

    fn push_melody_event(&mut self, note: u8, start_beat: f32, end_beat: f32, velocity: f32) {
        let len = self.loop_len_beats.max(4.0);
        let dur = if end_beat >= start_beat {
            (end_beat - start_beat).max(0.15)
        } else {
            (len - start_beat).max(0.15)
        };
        let clamped_dur = dur.min((len - start_beat).max(0.15));
        self.melody_events.push(MelodyLoopEvent {
            start_beat,
            duration_beats: clamped_dur,
            note,
            velocity,
        });
    }

    pub fn toggle(&mut self, synth: &mut PolySynth) {
        match self.state {
            LoopState::Idle => {
                self.chord_events.clear();
                self.melody_events.clear();
                self.open_chord = None;
                self.open_melody = None;
                self.playing_chord_idx = None;
                self.playing_melody_indices.clear();
                self.beat_pos = 0.0;
                self.loop_len_beats = 0.0;
                self.state = LoopState::RecordingBase;

                if let Some((root, quality, ref notes, vel)) = self.live_chord_info {
                    self.base_armed = false;
                    self.open_chord = Some(ChordLoopEvent {
                        start_beat: 0.0,
                        duration_beats: 0.25,
                        root,
                        quality,
                        notes: notes.clone(),
                        velocity: vel,
                    });
                } else {
                    self.base_armed = true;
                }
            }
            LoopState::RecordingBase => {
                if self.base_armed && self.chord_events.is_empty() && self.open_chord.is_none() {
                    self.state = LoopState::Idle;
                    self.base_armed = false;
                    self.beat_pos = 0.0;
                    return;
                }

                if let Some(mut open) = self.open_chord.take() {
                    open.duration_beats = (self.beat_pos - open.start_beat).max(0.15);
                    self.chord_events.push(open);
                }

                if self.chord_events.is_empty() {
                    self.state = LoopState::Idle;
                    self.base_armed = false;
                    self.beat_pos = 0.0;
                    return;
                }

                let raw_len = self.beat_pos.max(1.0);
                let q_len = ((raw_len / 4.0).round().max(1.0)) * 4.0;
                let scale = q_len / raw_len;

                for ev in &mut self.chord_events {
                    ev.start_beat = (ev.start_beat * scale).clamp(0.0, q_len - 0.1);
                    let scaled_dur = (ev.duration_beats * scale).max(0.25);
                    ev.duration_beats = scaled_dur.min((q_len - ev.start_beat).max(0.1));
                }

                synth.release_role(VoiceRole::Live);
                self.loop_len_beats = q_len;
                self.beat_pos = 0.0;
                self.playing_chord_idx = None;
                self.playing_melody_indices.clear();
                self.state = LoopState::Playing;
                self.sync_voices_at_current_beat(synth);
            }
            LoopState::Playing => {
                self.state = LoopState::Overdubbing;
            }
            LoopState::Overdubbing => {
                if let Some((note, start_beat, vel)) = self.open_melody.take() {
                    self.push_melody_event(note, start_beat, self.beat_pos, vel);
                }
                self.state = LoopState::Playing;
            }
        }
    }

    pub fn clear(&mut self, synth: &mut PolySynth) {
        synth.release_role(VoiceRole::ChordLoop);
        synth.release_role(VoiceRole::MelodyLoop);
        self.state = LoopState::Idle;
        self.beat_pos = 0.0;
        self.loop_len_beats = 0.0;
        self.base_armed = false;
        self.chord_events.clear();
        self.melody_events.clear();
        self.open_chord = None;
        self.open_melody = None;
        self.playing_chord_idx = None;
        self.playing_melody_indices.clear();
    }

    pub fn advance(&mut self, frames: usize, sample_rate: f32, synth: &mut PolySynth) {
        if frames == 0 || sample_rate <= 0.0 {
            return;
        }
        let d_beats = (frames as f32 / sample_rate) * (self.bpm.clamp(40.0, 220.0) / 60.0);

        match self.state {
            LoopState::Idle => {}
            LoopState::RecordingBase => {
                if !self.base_armed {
                    self.beat_pos += d_beats;
                }
            }
            LoopState::Playing | LoopState::Overdubbing => {
                if self.loop_len_beats < 4.0 {
                    return;
                }
                let prev_beat = self.beat_pos;
                self.beat_pos += d_beats;
                if self.beat_pos >= self.loop_len_beats {
                    self.beat_pos %= self.loop_len_beats;
                    if self.beat_pos < prev_beat {
                        // Force clean re-articulation on wrap-around
                        if let Some((_, start_beat, _)) = self.open_melody {
                            let note_and_vel = self.open_melody.take();
                            if let Some((n, _, v)) = note_and_vel {
                                self.push_melody_event(n, start_beat, self.loop_len_beats, v);
                                self.open_melody = Some((n, 0.0, v));
                            }
                        }
                        self.playing_chord_idx = None;
                        for &idx in &self.playing_melody_indices {
                            if let Some(ev) = self.melody_events.get(idx) {
                                synth.note_off_role(ev.note, VoiceRole::MelodyLoop);
                            }
                        }
                        self.playing_melody_indices.clear();
                    }
                }
                self.sync_voices_at_current_beat(synth);
            }
        }
    }

    fn sync_voices_at_current_beat(&mut self, synth: &mut PolySynth) {
        let active_chord = self.chord_events.iter().position(|ev| {
            self.beat_pos >= ev.start_beat && self.beat_pos < ev.start_beat + ev.duration_beats
        });

        if active_chord != self.playing_chord_idx {
            match active_chord {
                Some(idx) => {
                    let ev = &self.chord_events[idx];
                    synth.chord_on(&ev.notes, ev.velocity, VoiceRole::ChordLoop);
                }
                None => {
                    synth.release_role(VoiceRole::ChordLoop);
                }
            }
            self.playing_chord_idx = active_chord;
        }

        let mut new_melody_active = Vec::new();
        for (idx, ev) in self.melody_events.iter().enumerate() {
            if self.beat_pos >= ev.start_beat && self.beat_pos < ev.start_beat + ev.duration_beats {
                new_melody_active.push(idx);
                if !self.playing_melody_indices.contains(&idx) {
                    synth.note_on_role(ev.note, ev.velocity, VoiceRole::MelodyLoop);
                }
            }
        }
        for &old_idx in &self.playing_melody_indices {
            if !new_melody_active.contains(&old_idx) {
                if let Some(ev) = self.melody_events.get(old_idx) {
                    synth.note_off_role(ev.note, VoiceRole::MelodyLoop);
                }
            }
        }
        self.playing_melody_indices = new_melody_active;
    }

    pub fn active_chord(&self) -> (u8, u8) {
        if matches!(self.state, LoopState::Playing | LoopState::Overdubbing) {
            if let Some(idx) = self.playing_chord_idx {
                if let Some(ev) = self.chord_events.get(idx) {
                    return (ev.root, ev.quality);
                }
            }
        }
        if let Some((root, quality, _, _)) = self.live_chord_info {
            return (root, quality);
        }
        (255, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recording_base_waits_for_first_chord_and_quantizes_proportionally() {
        let mut synth = PolySynth::new(48_000.0);
        let mut looper = LoopEngine::new();
        looper.bpm = 120.0; // 2 beats/sec

        looper.toggle(&mut synth);
        assert_eq!(looper.state, LoopState::RecordingBase);
        assert!(looper.base_armed);

        // Advancing before first chord arrives keeps beat_pos at 0.0
        looper.advance(48_000, 48_000.0, &mut synth);
        assert_eq!(looper.beat_pos, 0.0);

        // First chord at beat 0.0 -> 2.5 beats, second chord 2.5 -> 5.0 beats (raw 5.0 -> quantizes to 4.0)
        looper.on_live_chord_on(0, 0, &[48, 60, 64, 67], 0.85);
        assert!(!looper.base_armed);
        looper.advance(60_000, 48_000.0, &mut synth); // 1.25s = 2.5 beats
        looper.on_live_chord_on(7, 0, &[55, 67, 71, 74], 0.85);
        looper.advance(60_000, 48_000.0, &mut synth); // total 5.0 beats
        looper.toggle(&mut synth);

        assert_eq!(looper.state, LoopState::Playing);
        assert_eq!(looper.loop_len_beats, 4.0);
        assert_eq!(looper.chord_events.len(), 2);
        assert!((looper.chord_events[0].start_beat - 0.0).abs() < 1e-3);
        assert!((looper.chord_events[1].start_beat - 2.0).abs() < 1e-3);
        assert!((looper.chord_events[1].duration_beats - 2.0).abs() < 1e-3);
    }

    #[test]
    fn bpm_time_stretch_changes_loop_speed_without_changing_note_pitch() {
        let mut synth = PolySynth::new(48_000.0);
        let mut looper = LoopEngine::new();
        looper.bpm = 60.0;
        looper.toggle(&mut synth);
        looper.on_live_chord_on(0, 0, &[48, 60, 64, 67], 0.85);
        looper.advance(48_000 * 4, 48_000.0, &mut synth); // 4 beats at 60 BPM
        looper.toggle(&mut synth);

        let freq_before = synth
            .voices
            .iter()
            .find(|v| v.active && v.note == 60)
            .unwrap()
            .frequency;

        looper.bpm = 120.0;
        looper.advance(24_000, 48_000.0, &mut synth); // 0.5s at 120 BPM = 1 beat
        assert!((looper.beat_pos - 1.0).abs() < 1e-3);

        let freq_after = synth
            .voices
            .iter()
            .find(|v| v.active && v.note == 60)
            .unwrap()
            .frequency;
        assert!((freq_before - freq_after).abs() < 1e-4);
    }
}

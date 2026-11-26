use std::f32::consts::PI;

pub const MAX_VOICES: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceRole {
    Live,
    ChordLoop,
    MelodyLoop,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Waveform {
    Sine,
    Saw,
    Square,
    Triangle,
}

impl Default for Waveform {
    fn default() -> Self {
        Waveform::Saw
    }
}

impl Waveform {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "sine" => Waveform::Sine,
            "saw" | "sawtooth" => Waveform::Saw,
            "square" => Waveform::Square,
            "triangle" | "tri" => Waveform::Triangle,
            _ => Waveform::Saw,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Adsr {
    pub attack: f32,  // seconds
    pub decay: f32,   // seconds
    pub sustain: f32, // level (0.0 - 1.0)
    pub release: f32, // seconds
}

impl Default for Adsr {
    fn default() -> Self {
        Self {
            attack: 0.012,
            decay: 0.10,
            sustain: 0.72,
            release: 0.15,
        }
    }
}

impl Adsr {
    pub fn for_role(role: VoiceRole) -> Self {
        match role {
            VoiceRole::ChordLoop => Self {
                attack: 0.018,
                decay: 0.12,
                sustain: 0.68,
                release: 0.18,
            },
            VoiceRole::Live | VoiceRole::MelodyLoop => Self {
                attack: 0.008,
                decay: 0.10,
                sustain: 0.72,
                release: 0.14,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EnvelopeState {
    Idle,
    Attack,
    Decay,
    Sustain,
    Release,
}

#[derive(Debug, Clone)]
pub struct Voice {
    pub note: u8,
    pub role: VoiceRole,
    pub frequency: f32,
    pub base_frequency: f32,
    pub velocity: f32,
    pub phase: f32,
    pub active: bool,
    pub env_state: EnvelopeState,
    pub env_level: f32,
    pub env_time: f32,
    release_level: f32,
    pub adsr: Adsr,
}

impl Voice {
    pub fn new() -> Self {
        Self {
            note: 0,
            role: VoiceRole::Live,
            frequency: 440.0,
            base_frequency: 440.0,
            velocity: 0.0,
            phase: 0.0,
            active: false,
            env_state: EnvelopeState::Idle,
            env_level: 0.0,
            env_time: 0.0,
            release_level: 0.0,
            adsr: Adsr::default(),
        }
    }

    pub fn trigger(
        &mut self,
        note: u8,
        velocity: f32,
        pitch_bend_semitones: f32,
        role: VoiceRole,
    ) {
        self.note = note;
        self.role = role;
        self.adsr = Adsr::for_role(role);
        self.velocity = velocity.clamp(0.05, 1.0);
        self.base_frequency = 440.0 * 2.0_f32.powf((note as f32 - 69.0) / 12.0);
        let applied_bend = if role == VoiceRole::ChordLoop {
            0.0
        } else {
            pitch_bend_semitones
        };
        self.update_pitch_bend(applied_bend);
        self.active = true;
        self.env_state = EnvelopeState::Attack;
        self.env_time = 0.0;
    }

    pub fn release(&mut self) {
        if self.active
            && self.env_state != EnvelopeState::Idle
            && self.env_state != EnvelopeState::Release
        {
            self.release_level = self.env_level;
            self.env_state = EnvelopeState::Release;
            self.env_time = 0.0;
        }
    }

    pub fn update_pitch_bend(&mut self, semitones: f32) {
        self.frequency = self.base_frequency * 2.0_f32.powf(semitones / 12.0);
    }

    pub fn next_sample(&mut self, sample_rate: f32, waveform: Waveform) -> f32 {
        if !self.active || self.env_state == EnvelopeState::Idle {
            return 0.0;
        }

        let dt = 1.0 / sample_rate;
        self.env_time += dt;

        match self.env_state {
            EnvelopeState::Idle => {
                self.env_level = 0.0;
                self.active = false;
            }
            EnvelopeState::Attack => {
                if self.adsr.attack <= 0.0001 {
                    self.env_level = 1.0;
                    self.env_state = EnvelopeState::Decay;
                    self.env_time = 0.0;
                } else {
                    self.env_level = (self.env_time / self.adsr.attack).min(1.0);
                    if self.env_time >= self.adsr.attack {
                        self.env_state = EnvelopeState::Decay;
                        self.env_time = 0.0;
                    }
                }
            }
            EnvelopeState::Decay => {
                if self.adsr.decay <= 0.0001 {
                    self.env_level = self.adsr.sustain;
                    self.env_state = EnvelopeState::Sustain;
                } else {
                    let progress = (self.env_time / self.adsr.decay).min(1.0);
                    self.env_level = 1.0 - (1.0 - self.adsr.sustain) * progress;
                    if self.env_time >= self.adsr.decay {
                        self.env_state = EnvelopeState::Sustain;
                        self.env_level = self.adsr.sustain;
                    }
                }
            }
            EnvelopeState::Sustain => {
                self.env_level = self.adsr.sustain;
            }
            EnvelopeState::Release => {
                if self.adsr.release <= 0.0001 {
                    self.env_level = 0.0;
                    self.env_state = EnvelopeState::Idle;
                    self.active = false;
                } else {
                    let progress = (self.env_time / self.adsr.release).min(1.0);
                    self.env_level = self.release_level * (1.0 - progress);
                    if self.env_time >= self.adsr.release {
                        self.env_state = EnvelopeState::Idle;
                        self.env_level = 0.0;
                        self.active = false;
                    }
                }
            }
        }

        if !self.active {
            return 0.0;
        }

        let raw = match waveform {
            Waveform::Sine => (self.phase * 2.0 * PI).sin(),
            Waveform::Saw => 2.0 * (self.phase - (self.phase + 0.5).floor()),
            Waveform::Square => {
                if self.phase < 0.5 {
                    0.8
                } else {
                    -0.8
                }
            }
            Waveform::Triangle => {
                let p = self.phase - 0.25;
                2.0 * (2.0 * (p - (p + 0.5).floor()).abs() - 0.5)
            }
        };

        self.phase += self.frequency * dt;
        while self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        raw * self.env_level * self.velocity
    }
}

// Low Pass State Variable Filter
#[derive(Debug, Clone)]
pub struct LowPassFilter {
    pub cutoff: f32,
    pub resonance: f32,
    s1: f32,
    s2: f32,
    g: f32,
    target_g: f32,
    smoothing: f32,
    sample_rate: f32,
}

impl LowPassFilter {
    pub fn new(sample_rate: f32) -> Self {
        let cutoff = 5000.0_f32.min(sample_rate * 0.45);
        let g = (PI * cutoff / sample_rate).tan();
        Self {
            cutoff,
            resonance: 0.1,
            s1: 0.0,
            s2: 0.0,
            g,
            target_g: g,
            smoothing: 1.0 - (-1.0 / (0.02 * sample_rate)).exp(),
            sample_rate,
        }
    }

    pub fn set_cutoff(&mut self, cutoff_hz: f32) {
        self.cutoff = cutoff_hz.clamp(40.0, 18000.0_f32.min(self.sample_rate * 0.45));
        self.target_g = (PI * self.cutoff / self.sample_rate).tan();
    }

    pub fn set_resonance(&mut self, res: f32) {
        self.resonance = res.clamp(0.0, 0.95);
    }

    pub fn process(&mut self, input: f32) -> f32 {
        self.g += self.smoothing * (self.target_g - self.g);
        let k = 2.0 - 1.8 * self.resonance;
        let a1 = 1.0 / (1.0 + self.g * (self.g + k));
        let v1 = a1 * (self.s1 + self.g * (input - self.s2));
        let v2 = self.s2 + self.g * v1;
        self.s1 = 2.0 * v1 - self.s1;
        self.s2 = 2.0 * v2 - self.s2;
        v2
    }
}

// Simple Delay effect for spacious ambient synth
#[derive(Debug, Clone)]
pub struct DelayEffect {
    buffer: Vec<f32>,
    write_idx: usize,
    feedback: f32,
    mix: f32,
}

impl DelayEffect {
    pub fn new(sample_rate: usize, delay_seconds: f32) -> Self {
        let len = (sample_rate as f32 * delay_seconds) as usize;
        Self {
            buffer: vec![0.0; len.max(10)],
            write_idx: 0,
            feedback: 0.30,
            mix: 0.22,
        }
    }

    pub fn process(&mut self, input: f32) -> f32 {
        if self.buffer.is_empty() {
            return input;
        }
        let read_sample = self.buffer[self.write_idx];
        self.buffer[self.write_idx] = input + read_sample * self.feedback;
        self.write_idx = (self.write_idx + 1) % self.buffer.len();

        input * (1.0 - self.mix) + read_sample * self.mix
    }
}

pub struct PolySynth {
    pub voices: [Voice; MAX_VOICES],
    pub waveform: Waveform,
    pub filter: LowPassFilter,
    pub delay: DelayEffect,
    pub pitch_bend: f32,
    pub master_volume: f32,
    pub sample_rate: f32,
    pub norm_cutoff: f32,
    pub pulse_env: f32,
    pulse_decay: f32,
}

impl PolySynth {
    pub fn new(sample_rate: f32) -> Self {
        let voices = std::array::from_fn(|_| Voice::new());
        let mut synth = Self {
            voices,
            waveform: Waveform::Saw,
            filter: LowPassFilter::new(sample_rate),
            delay: DelayEffect::new(sample_rate as usize, 0.28),
            pitch_bend: 0.0,
            master_volume: 0.75,
            sample_rate,
            norm_cutoff: 0.7,
            pulse_env: 0.0,
            pulse_decay: (-1.0 / (0.12 * sample_rate.max(1.0))).exp(),
        };
        synth.set_cutoff(0.7);
        synth
    }

    fn allocate_and_trigger(&mut self, note: u8, velocity: f32, role: VoiceRole) {
        if let Some(v) = self
            .voices
            .iter_mut()
            .find(|v| v.active && v.note == note && v.role == role)
        {
            v.trigger(note, velocity, self.pitch_bend, role);
            return;
        }
        if let Some(v) = self.voices.iter_mut().find(|v| !v.active) {
            v.trigger(note, velocity, self.pitch_bend, role);
            return;
        }
        if let Some(v) = self
            .voices
            .iter_mut()
            .filter(|v| v.env_state == EnvelopeState::Release)
            .min_by(|a, b| {
                a.env_level
                    .partial_cmp(&b.env_level)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        {
            v.trigger(note, velocity, self.pitch_bend, role);
            return;
        }
        if let Some(v) = self.voices.iter_mut().min_by(|a, b| {
            a.env_level
                .partial_cmp(&b.env_level)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            v.trigger(note, velocity, self.pitch_bend, role);
        }
    }

    pub fn note_on_role(&mut self, note: u8, velocity: f32, role: VoiceRole) {
        self.allocate_and_trigger(note, velocity, role);
    }

    pub fn note_off_role(&mut self, note: u8, role: VoiceRole) {
        for v in self.voices.iter_mut() {
            if v.active && v.note == note && v.role == role {
                v.release();
            }
        }
    }

    pub fn release_role(&mut self, role: VoiceRole) {
        for v in self.voices.iter_mut() {
            if v.active && v.role == role {
                v.release();
            }
        }
    }

    pub fn chord_on(&mut self, notes: &[u8], velocity: f32, role: VoiceRole) {
        for v in self.voices.iter_mut() {
            if v.active && v.role == role && !notes.contains(&v.note) {
                v.release();
            }
        }
        let chord_vel = (velocity * 0.68).clamp(0.1, 1.0);
        for &n in notes {
            self.allocate_and_trigger(n, chord_vel, role);
        }
    }

    pub fn trigger_pulse(&mut self, velocity: f32) {
        let v = velocity.clamp(0.2, 1.0);
        self.pulse_env = self.pulse_env.max(v);
        for voice in self.voices.iter_mut() {
            if voice.active
                && (voice.role == VoiceRole::ChordLoop || voice.role == VoiceRole::Live)
                && voice.env_state != EnvelopeState::Release
            {
                voice.env_level = (voice.env_level + 0.25 * v).min(1.0);
            }
        }
        self.apply_effective_cutoff();
    }

    pub fn all_notes_off(&mut self) {
        for v in self.voices.iter_mut() {
            v.release();
        }
    }

    pub fn set_pitch_bend(&mut self, semitones: f32) {
        self.pitch_bend = semitones;
        for v in self.voices.iter_mut() {
            if v.active && v.role != VoiceRole::ChordLoop {
                v.update_pitch_bend(semitones);
            }
        }
    }

    pub fn set_cutoff(&mut self, norm_cutoff: f32) {
        self.norm_cutoff = norm_cutoff.clamp(0.0, 1.0);
        self.apply_effective_cutoff();
    }

    fn apply_effective_cutoff(&mut self) {
        let eff = (self.norm_cutoff + 0.28 * self.pulse_env).clamp(0.0, 1.0);
        let hz = 80.0 * (14000.0 / 80.0_f32).powf(eff);
        self.filter.set_cutoff(hz);
    }

    pub fn set_resonance(&mut self, res: f32) {
        self.filter.set_resonance(res);
    }

    pub fn next_sample(&mut self) -> f32 {
        if self.pulse_env > 0.001 {
            self.pulse_env *= self.pulse_decay;
            self.apply_effective_cutoff();
        } else if self.pulse_env > 0.0 {
            self.pulse_env = 0.0;
            self.apply_effective_cutoff();
        }

        let mut mixed = 0.0;
        let mut active_count = 0;

        for v in self.voices.iter_mut() {
            if v.active {
                mixed += v.next_sample(self.sample_rate, self.waveform);
                active_count += 1;
            }
        }

        if active_count > 1 {
            mixed /= (active_count as f32).sqrt();
        }

        let filtered = self.filter.process(mixed);
        let delayed = self.delay.process(filtered);
        let accent = 1.0 + 0.22 * self.pulse_env;
        let out = delayed * self.master_volume * accent;
        out.tanh()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filtered_rms(sample_rate: f32, cutoff: f32, frequency: f32) -> f32 {
        let mut filter = LowPassFilter::new(sample_rate);
        filter.set_cutoff(cutoff);
        let mut energy = 0.0;
        let count = sample_rate as usize;
        for i in 0..count {
            let input = (2.0 * PI * frequency * i as f32 / sample_rate).sin();
            let output = filter.process(input);
            assert!(output.is_finite());
            if i >= count / 2 {
                energy += output * output;
            }
        }
        (energy / (count / 2) as f32).sqrt()
    }

    #[test]
    fn cutoff_audibly_changes_high_frequency_content() {
        for sample_rate in [44_100.0, 48_000.0] {
            let closed = filtered_rms(sample_rate, 200.0, 3_000.0);
            let open = filtered_rms(sample_rate, 10_000.0, 3_000.0);
            assert!(open > closed * 20.0, "closed={closed}, open={open}");
        }
    }

    #[test]
    fn early_release_does_not_jump_to_sustain() {
        let mut voice = Voice::new();
        voice.trigger(60, 1.0, 0.0, VoiceRole::Live);
        voice.next_sample(48_000.0, Waveform::Saw);
        let before = voice.env_level;
        voice.release();
        voice.next_sample(48_000.0, Waveform::Saw);
        assert!(voice.env_level <= before);
    }

    #[test]
    fn chord_loop_ignores_pitch_bend_while_live_bends() {
        let mut synth = PolySynth::new(48_000.0);
        synth.chord_on(&[48, 60, 64, 67], 0.85, VoiceRole::ChordLoop);
        synth.note_on_role(72, 0.9, VoiceRole::Live);
        synth.set_pitch_bend(2.0);

        let loop_c = synth
            .voices
            .iter()
            .find(|v| v.active && v.note == 60 && v.role == VoiceRole::ChordLoop)
            .expect("chord loop voice present");
        assert!((loop_c.frequency - loop_c.base_frequency).abs() < 1e-3);

        let live_c = synth
            .voices
            .iter()
            .find(|v| v.active && v.note == 72 && v.role == VoiceRole::Live)
            .expect("live voice present");
        assert!(live_c.frequency > live_c.base_frequency * 1.1);
    }

    #[test]
    fn sixteen_voices_hold_chord_and_melody_without_stealing() {
        let mut synth = PolySynth::new(48_000.0);
        synth.chord_on(&[48, 60, 64, 67], 0.85, VoiceRole::ChordLoop);
        for n in [72, 74, 76, 77] {
            synth.note_on_role(n, 0.9, VoiceRole::MelodyLoop);
        }
        synth.note_on_role(79, 0.9, VoiceRole::Live);
        let active = synth.voices.iter().filter(|v| v.active).count();
        assert_eq!(active, 9);
    }
}

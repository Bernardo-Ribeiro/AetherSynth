mod looper;
mod synth;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use looper::LoopEngine;
use serde::Deserialize;
use std::net::UdpSocket;
use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;
use synth::{PolySynth, VoiceRole, Waveform};

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AudioCommand {
    NoteOn {
        note: u8,
        #[serde(default = "default_velocity")]
        velocity: f32,
    },
    NoteOff {
        note: u8,
    },
    ChordOn {
        root: u8,
        #[serde(default)]
        quality: String,
        notes: Vec<u8>,
        #[serde(default = "default_velocity")]
        velocity: f32,
    },
    ChordOff,
    Pulse {
        #[serde(default = "default_velocity")]
        velocity: f32,
    },
    LoopToggle,
    LoopClear,
    AllNotesOff,
    Modulate {
        #[serde(default)]
        cutoff: Option<f32>,
        #[serde(default)]
        pitch_bend: Option<f32>,
        #[serde(default)]
        volume: Option<f32>,
        #[serde(default)]
        resonance: Option<f32>,
        #[serde(default)]
        bpm: Option<f32>,
    },
    SetWaveform {
        waveform: String,
    },
    Ping,
}

fn default_velocity() -> f32 {
    0.85
}

enum EngineEvent {
    NoteOn { note: u8, velocity: f32 },
    NoteOff { note: u8 },
    ChordOn {
        root: u8,
        quality: u8,
        notes: Vec<u8>,
        velocity: f32,
    },
    ChordOff,
    Pulse { velocity: f32 },
    LoopToggle,
    LoopClear,
    AllNotesOff,
    SetWaveform { waveform: Waveform },
}

struct SharedTelemetry {
    cutoff_bits: AtomicU32,
    pitch_bend_bits: AtomicU32,
    volume_bits: AtomicU32,
    resonance_bits: AtomicU32,
    bpm_bits: AtomicU32,
    beat_pos_bits: AtomicU32,
    loop_len_bits: AtomicU32,
    loop_state: AtomicU8,
    active_chord_root: AtomicU8,
    active_chord_quality: AtomicU8,
}

impl SharedTelemetry {
    fn new() -> Self {
        Self {
            cutoff_bits: AtomicU32::new(0.7_f32.to_bits()),
            pitch_bend_bits: AtomicU32::new(0.0_f32.to_bits()),
            volume_bits: AtomicU32::new(0.8_f32.to_bits()),
            resonance_bits: AtomicU32::new(0.12_f32.to_bits()),
            bpm_bits: AtomicU32::new(96.0_f32.to_bits()),
            beat_pos_bits: AtomicU32::new(0.0_f32.to_bits()),
            loop_len_bits: AtomicU32::new(0.0_f32.to_bits()),
            loop_state: AtomicU8::new(0),
            active_chord_root: AtomicU8::new(255),
            active_chord_quality: AtomicU8::new(0),
        }
    }

    #[inline]
    fn load_f32(atom: &AtomicU32) -> f32 {
        f32::from_bits(atom.load(Ordering::Relaxed))
    }

    #[inline]
    fn store_f32(atom: &AtomicU32, val: f32) {
        atom.store(val.to_bits(), Ordering::Relaxed);
    }
}

struct AudioProcessor {
    synth: PolySynth,
    looper: LoopEngine,
    rx: mpsc::Receiver<EngineEvent>,
    telemetry: Arc<SharedTelemetry>,
    sample_rate: f32,
    last_cutoff: f32,
    last_pitch_bend: f32,
    last_resonance: f32,
}

impl AudioProcessor {
    fn new(sample_rate: f32, rx: mpsc::Receiver<EngineEvent>, telemetry: Arc<SharedTelemetry>) -> Self {
        Self {
            synth: PolySynth::new(sample_rate),
            looper: LoopEngine::new(),
            rx,
            telemetry,
            sample_rate,
            last_cutoff: -1.0,
            last_pitch_bend: f32::NAN,
            last_resonance: -1.0,
        }
    }

    fn drain_events_and_sync_params(&mut self) {
        while let Ok(ev) = self.rx.try_recv() {
            match ev {
                EngineEvent::NoteOn { note, velocity } => {
                    self.synth.note_on_role(note, velocity, VoiceRole::Live);
                    self.looper.on_live_note_on(note, velocity);
                }
                EngineEvent::NoteOff { note } => {
                    self.synth.note_off_role(note, VoiceRole::Live);
                    self.looper.on_live_note_off(note);
                }
                EngineEvent::ChordOn {
                    root,
                    quality,
                    notes,
                    velocity,
                } => {
                    self.synth.chord_on(&notes, velocity, VoiceRole::Live);
                    self.looper.on_live_chord_on(root, quality, &notes, velocity);
                }
                EngineEvent::ChordOff => {
                    self.synth.release_role(VoiceRole::Live);
                    self.looper.on_live_chord_off();
                }
                EngineEvent::Pulse { velocity } => {
                    self.synth.trigger_pulse(velocity);
                }
                EngineEvent::LoopToggle => {
                    self.looper.toggle(&mut self.synth);
                }
                EngineEvent::LoopClear => {
                    self.looper.clear(&mut self.synth);
                }
                EngineEvent::AllNotesOff => {
                    self.synth.all_notes_off();
                }
                EngineEvent::SetWaveform { waveform } => {
                    self.synth.waveform = waveform;
                }
            }
        }

        let cutoff = SharedTelemetry::load_f32(&self.telemetry.cutoff_bits);
        if (cutoff - self.last_cutoff).abs() > 1e-4 {
            self.synth.set_cutoff(cutoff);
            self.last_cutoff = cutoff;
        }

        let pitch_bend = SharedTelemetry::load_f32(&self.telemetry.pitch_bend_bits);
        if (pitch_bend - self.last_pitch_bend).abs() > 1e-4 || self.last_pitch_bend.is_nan() {
            self.synth.set_pitch_bend(pitch_bend);
            self.last_pitch_bend = pitch_bend;
        }

        let resonance = SharedTelemetry::load_f32(&self.telemetry.resonance_bits);
        if (resonance - self.last_resonance).abs() > 1e-4 {
            self.synth.set_resonance(resonance);
            self.last_resonance = resonance;
        }

        self.synth.master_volume = SharedTelemetry::load_f32(&self.telemetry.volume_bits).clamp(0.0, 1.2);
        self.looper.bpm = SharedTelemetry::load_f32(&self.telemetry.bpm_bits).clamp(40.0, 220.0);
    }

    fn publish_telemetry(&self) {
        SharedTelemetry::store_f32(&self.telemetry.beat_pos_bits, self.looper.beat_pos);
        SharedTelemetry::store_f32(&self.telemetry.loop_len_bits, self.looper.loop_len_beats);
        self.telemetry
            .loop_state
            .store(self.looper.state as u8, Ordering::Relaxed);
        let (root, quality) = self.looper.active_chord();
        self.telemetry
            .active_chord_root
            .store(root, Ordering::Relaxed);
        self.telemetry
            .active_chord_quality
            .store(quality, Ordering::Relaxed);
    }

    fn render_f32(&mut self, data: &mut [f32], channels: usize) {
        self.drain_events_and_sync_params();
        let ch = channels.max(1);
        for chunk in data.chunks_mut(32 * ch) {
            let frames = chunk.len() / ch;
            self.looper.advance(frames, self.sample_rate, &mut self.synth);
            for frame in chunk.chunks_mut(ch) {
                let sample = self.synth.next_sample();
                for s in frame.iter_mut() {
                    *s = sample;
                }
            }
        }
        self.publish_telemetry();
    }

    fn render_i16(&mut self, data: &mut [i16], channels: usize) {
        self.drain_events_and_sync_params();
        let ch = channels.max(1);
        for chunk in data.chunks_mut(32 * ch) {
            let frames = chunk.len() / ch;
            self.looper.advance(frames, self.sample_rate, &mut self.synth);
            for frame in chunk.chunks_mut(ch) {
                let sample = (self.synth.next_sample().clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                for s in frame.iter_mut() {
                    *s = sample;
                }
            }
        }
        self.publish_telemetry();
    }

    fn render_u16(&mut self, data: &mut [u16], channels: usize) {
        self.drain_events_and_sync_params();
        let ch = channels.max(1);
        for chunk in data.chunks_mut(32 * ch) {
            let frames = chunk.len() / ch;
            self.looper.advance(frames, self.sample_rate, &mut self.synth);
            for frame in chunk.chunks_mut(ch) {
                let sample = ((self.synth.next_sample().clamp(-1.0, 1.0) * 0.5 + 0.5)
                    * u16::MAX as f32) as u16;
                for s in frame.iter_mut() {
                    *s = sample;
                }
            }
        }
        self.publish_telemetry();
    }
}

fn build_low_latency_stream(
    device: &cpal::Device,
    default_config: &cpal::SupportedStreamConfig,
    processor: AudioProcessor,
) -> Result<(cpal::Stream, String), Box<dyn std::error::Error>> {
    let err_fn = |err| eprintln!("Erro no stream de áudio: {}", err);
    let sample_format = default_config.sample_format();
    let channels = default_config.channels() as usize;
    let base_config = default_config.config();

    let candidate_buffers = [
        ("Fixed(256)", cpal::BufferSize::Fixed(256)),
        ("Fixed(512)", cpal::BufferSize::Fixed(512)),
        ("Default", cpal::BufferSize::Default),
    ];

    let mut proc_opt = Some(processor);

    for (label, buf_size) in candidate_buffers {
        let mut cfg = base_config.clone();
        cfg.buffer_size = buf_size;

        // Probe support by building stream on the last attempt or trying with proc
        if label == "Default" {
            let mut proc = proc_opt.take().unwrap();
            let stream = match sample_format {
                cpal::SampleFormat::F32 => device.build_output_stream(
                    &cfg,
                    move |data: &mut [f32], _| proc.render_f32(data, channels),
                    err_fn,
                    None,
                )?,
                cpal::SampleFormat::I16 => device.build_output_stream(
                    &cfg,
                    move |data: &mut [i16], _| proc.render_i16(data, channels),
                    err_fn,
                    None,
                )?,
                cpal::SampleFormat::U16 => device.build_output_stream(
                    &cfg,
                    move |data: &mut [u16], _| proc.render_u16(data, channels),
                    err_fn,
                    None,
                )?,
                other => return Err(format!("Formato de áudio não suportado: {:?}", other).into()),
            };
            return Ok((stream, label.to_string()));
        } else {
            // Check if this fixed buffer size builds cleanly using a slot
            let proc = proc_opt.take().unwrap();
            let shared_fallback = std::sync::Arc::new(std::sync::Mutex::new(Some(proc)));
            let slot_cb = Arc::clone(&shared_fallback);

            let res = match sample_format {
                cpal::SampleFormat::F32 => device.build_output_stream(
                    &cfg,
                    move |data: &mut [f32], _| {
                        if let Ok(mut guard) = slot_cb.try_lock() {
                            if let Some(ref mut p) = *guard {
                                p.render_f32(data, channels);
                            }
                        }
                    },
                    err_fn,
                    None,
                ),
                cpal::SampleFormat::I16 => device.build_output_stream(
                    &cfg,
                    move |data: &mut [i16], _| {
                        if let Ok(mut guard) = slot_cb.try_lock() {
                            if let Some(ref mut p) = *guard {
                                p.render_i16(data, channels);
                            }
                        }
                    },
                    err_fn,
                    None,
                ),
                cpal::SampleFormat::U16 => device.build_output_stream(
                    &cfg,
                    move |data: &mut [u16], _| {
                        if let Ok(mut guard) = slot_cb.try_lock() {
                            if let Some(ref mut p) = *guard {
                                p.render_u16(data, channels);
                            }
                        }
                    },
                    err_fn,
                    None,
                ),
                other => return Err(format!("Formato de áudio não suportado: {:?}", other).into()),
            };

            match res {
                Ok(stream) => return Ok((stream, label.to_string())),
                Err(_) => {
                    proc_opt = shared_fallback.lock().unwrap().take();
                }
            }
        }
    }

    Err("Falha ao criar stream de áudio".into())
}

fn spawn_headless_tick(mut processor: AudioProcessor) {
    std::thread::spawn(move || {
        let mut buf = [0.0_f32; 256 * 2];
        loop {
            processor.render_f32(&mut buf, 2);
            std::thread::sleep(Duration::from_micros(5333));
        }
    });
}

fn state_str(code: u8) -> &'static str {
    match code {
        1 => "recording_base",
        2 => "playing",
        3 => "overdubbing",
        _ => "idle",
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bind_addr = std::env::var("AUDIO_BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:8765".to_string());
    let force_headless = std::env::var("HEADLESS_AUDIO").ok().as_deref() == Some("1");

    println!("=====================================================");
    println!("  AetherSynth - Rust Lock-Free Audio & Loop Engine");
    println!("  16-Voice Polyphonic Synth + Sample-Accurate Looper");
    println!("=====================================================");

    let (tx, rx) = mpsc::channel::<EngineEvent>();
    let telemetry = Arc::new(SharedTelemetry::new());

    let mut _active_stream: Option<cpal::Stream> = None;

    if !force_headless {
        let host = cpal::default_host();
        if let Some(device) = host.default_output_device() {
            if let Ok(default_config) = device.default_output_config() {
                let sample_rate = default_config.sample_rate().0 as f32;
                let channels = default_config.channels() as usize;
                println!(
                    "Dispositivo: {} ({} Hz, {} canais, {:?})",
                    device.name().unwrap_or_else(|_| "Audio".into()),
                    sample_rate,
                    channels,
                    default_config.sample_format()
                );

                let processor = AudioProcessor::new(sample_rate, rx, Arc::clone(&telemetry));
                let (stream, buf_mode) = build_low_latency_stream(&device, &default_config, processor)?;
                stream.play()?;
                println!("Stream iniciado (buffer: {}, lock-free channel)!", buf_mode);
                _active_stream = Some(stream);
            } else {
                println!("[AVISO] Configuracao de saida indisponivel; executando em modo headless.");
                spawn_headless_tick(AudioProcessor::new(48_000.0, rx, Arc::clone(&telemetry)));
            }
        } else {
            println!("[AVISO] Nenhum dispositivo de audio detectado; executando em modo headless.");
            spawn_headless_tick(AudioProcessor::new(48_000.0, rx, Arc::clone(&telemetry)));
        }
    } else {
        println!("[INFO] HEADLESS_AUDIO=1 ativo (48000 Hz software tick).");
        spawn_headless_tick(AudioProcessor::new(48_000.0, rx, Arc::clone(&telemetry)));
    }

    let socket = UdpSocket::bind(&bind_addr)?;
    println!("Ouvindo comandos UDP em {}", bind_addr);

    let mut buf = [0u8; 2048];
    loop {
        match socket.recv_from(&mut buf) {
            Ok((size, src)) => {
                let msg = match std::str::from_utf8(&buf[..size]) {
                    Ok(m) => m,
                    Err(_) => continue,
                };

                match serde_json::from_str::<AudioCommand>(msg) {
                    Ok(cmd) => match cmd {
                        AudioCommand::NoteOn { note, velocity } => {
                            let _ = tx.send(EngineEvent::NoteOn { note, velocity });
                        }
                        AudioCommand::NoteOff { note } => {
                            let _ = tx.send(EngineEvent::NoteOff { note });
                        }
                        AudioCommand::ChordOn {
                            root,
                            quality,
                            notes,
                            velocity,
                        } => {
                            let q = match quality.to_lowercase().as_str() {
                                "minor" | "m" | "1" => 1,
                                _ => 0,
                            };
                            let _ = tx.send(EngineEvent::ChordOn {
                                root,
                                quality: q,
                                notes,
                                velocity,
                            });
                        }
                        AudioCommand::ChordOff => {
                            let _ = tx.send(EngineEvent::ChordOff);
                        }
                        AudioCommand::Pulse { velocity } => {
                            let _ = tx.send(EngineEvent::Pulse { velocity });
                        }
                        AudioCommand::LoopToggle => {
                            let _ = tx.send(EngineEvent::LoopToggle);
                        }
                        AudioCommand::LoopClear => {
                            let _ = tx.send(EngineEvent::LoopClear);
                        }
                        AudioCommand::AllNotesOff => {
                            let _ = tx.send(EngineEvent::AllNotesOff);
                        }
                        AudioCommand::Modulate {
                            cutoff,
                            pitch_bend,
                            volume,
                            resonance,
                            bpm,
                        } => {
                            if let Some(c) = cutoff {
                                SharedTelemetry::store_f32(&telemetry.cutoff_bits, c.clamp(0.0, 1.0));
                            }
                            if let Some(pb) = pitch_bend {
                                SharedTelemetry::store_f32(&telemetry.pitch_bend_bits, pb.clamp(-12.0, 12.0));
                            }
                            if let Some(v) = volume {
                                SharedTelemetry::store_f32(&telemetry.volume_bits, v.clamp(0.0, 1.2));
                            }
                            if let Some(r) = resonance {
                                SharedTelemetry::store_f32(&telemetry.resonance_bits, r.clamp(0.0, 0.95));
                            }
                            if let Some(b) = bpm {
                                SharedTelemetry::store_f32(&telemetry.bpm_bits, b.clamp(40.0, 220.0));
                            }
                        }
                        AudioCommand::SetWaveform { waveform } => {
                            let wf = Waveform::from_str(&waveform);
                            let _ = tx.send(EngineEvent::SetWaveform { waveform: wf });
                            println!("Forma de onda alterada para: {:?}", wf);
                        }
                        AudioCommand::Ping => {
                            let lstate = state_str(telemetry.loop_state.load(Ordering::Relaxed));
                            let bpm = SharedTelemetry::load_f32(&telemetry.bpm_bits);
                            let beat_pos = SharedTelemetry::load_f32(&telemetry.beat_pos_bits);
                            let loop_len = SharedTelemetry::load_f32(&telemetry.loop_len_bits);
                            let root = telemetry.active_chord_root.load(Ordering::Relaxed);
                            let qual = telemetry.active_chord_quality.load(Ordering::Relaxed);
                            let resp = format!(
                                "{{\"status\":\"pong\",\"loop_state\":\"{}\",\"bpm\":{:.2},\"beat_pos\":{:.3},\"loop_len_beats\":{:.2},\"active_chord_root\":{},\"active_chord_quality\":{}}}",
                                lstate, bpm, beat_pos, loop_len, root, qual
                            );
                            let _ = socket.send_to(resp.as_bytes(), src);
                        }
                    },
                    Err(e) => {
                        eprintln!("Erro ao decodificar JSON UDP: {} (msg: {})", e, msg);
                    }
                }
            }
            Err(e) => {
                eprintln!("Erro no socket UDP: {}", e);
            }
        }
    }
}

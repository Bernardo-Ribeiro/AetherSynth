mod synth;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use serde::Deserialize;
use std::net::UdpSocket;
use std::sync::{Arc, Mutex};
use synth::{PolySynth, Waveform};

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
    },
    SetWaveform {
        waveform: String,
    },
    Ping,
}

fn default_velocity() -> f32 {
    0.8
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=====================================================");
    println!("  Gestures Music AI - Rust Audio Engine v0.1");
    println!("  Low-latency Polyphonic Synthesizer + DSP Filter");
    println!("=====================================================");

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("Nenhum dispositivo de saída de áudio encontrado!");

    println!("Dispositivo de áudio detectado: {}", device.name()?);

    let default_config = device.default_output_config()?;
    let sample_rate = default_config.sample_rate().0 as f32;
    let channels = default_config.channels() as usize;

    println!(
        "Configuração de áudio: {} Hz, {} canais, formato: {:?}",
        sample_rate,
        channels,
        default_config.sample_format()
    );

    let synth = Arc::new(Mutex::new(PolySynth::new(sample_rate)));
    let synth_audio = Arc::clone(&synth);

    // Audio stream callback
    let err_fn = |err| eprintln!("Erro no stream de áudio: {}", err);

    let stream = match default_config.sample_format() {
        cpal::SampleFormat::F32 => device.build_output_stream(
            &default_config.into(),
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                if let Ok(mut s) = synth_audio.try_lock() {
                    for frame in data.chunks_mut(channels) {
                        let sample = s.next_sample();
                        for channel_sample in frame.iter_mut() {
                            *channel_sample = sample;
                        }
                    }
                } else {
                    for frame in data.chunks_mut(channels) {
                        for channel_sample in frame.iter_mut() {
                            *channel_sample = 0.0;
                        }
                    }
                }
            },
            err_fn,
            None,
        )?,
        _ => panic!("Formato de áudio não suportado (apenas F32 suportado atualmente)"),
    };

    stream.play()?;
    println!("Stream de áudio iniciado com sucesso!");

    // UDP Listener
    let bind_addr = "127.0.0.1:8765";
    let socket = UdpSocket::bind(bind_addr)?;
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
                    Ok(cmd) => {
                        let mut s = synth.lock().unwrap();
                        match cmd {
                            AudioCommand::NoteOn { note, velocity } => {
                                s.note_on(note, velocity);
                            }
                            AudioCommand::NoteOff { note } => {
                                s.note_off(note);
                            }
                            AudioCommand::AllNotesOff => {
                                s.all_notes_off();
                            }
                            AudioCommand::Modulate {
                                cutoff,
                                pitch_bend,
                                volume,
                                resonance,
                            } => {
                                if let Some(c) = cutoff {
                                    s.set_cutoff(c);
                                }
                                if let Some(pb) = pitch_bend {
                                    s.set_pitch_bend(pb);
                                }
                                if let Some(v) = volume {
                                    s.master_volume = v.clamp(0.0, 1.2);
                                }
                                if let Some(r) = resonance {
                                    s.set_resonance(r);
                                }
                            }
                            AudioCommand::SetWaveform { waveform } => {
                                s.waveform = Waveform::from_str(&waveform);
                                println!("Forma de onda alterada para: {:?}", s.waveform);
                            }
                            AudioCommand::Ping => {
                                let _ = socket.send_to(b"{\"status\":\"pong\"}", src);
                            }
                        }
                    }
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

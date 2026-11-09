# Gestures Music AI 🎵🤖

> **Instrumento Musical Holográfico e Tutor Inteligente Controlado por Gestos em Tempo Real**
> Projeto para a disciplina de **Tópicos Avançados (IA ou Robótica)** — Curso TADS.

---

## 🎯 Visão Geral do Projeto

Nem todos têm acesso a instrumentos musicais físicos ou conhecimentos formais de teoria musical. O **Gestures Music AI** transforma qualquer webcam em um instrumento musical holográfico de baixa latência e em um **tutor musical inteligente**, permitindo que o usuário toque melodias no ar, module timbres com movimentos expressivos e aprenda músicas com feedback pedagógico da IA em tempo real.

---

## 🧠 Onde entra a Inteligência Artificial?

Respondendo diretamente ao desafio de ir além de um simples rastreador de mãos:

1. **Classificação de Posturas e Acordes por Machine Learning (Scikit-Learn / KNN)**:
   - A partir dos 21 landmarks de cada mão, um vetor de características invariante a escala e rotação é extraído.
   - O classificador de ML identifica gestos dinâmicos (`Fist`, `Open Palm`, `Point`, `Peace`, `Three`, `Four`, `Pinch`, `Rock`) com probabilidades de confiança, permitindo alternar modos sonoros e acionar notas.
2. **Mapeamento Contínuo e Não-Linear de Expressividade**:
   - A **Mão Esquerda** funciona como controle acústico analógico:
     - **Abertura da mão (Openness)**: Modula dinamicamente a frequência de corte (*Cutoff*) de um filtro ressonante passa-baixas (abafado quando fechada, brilhante quando aberta).
     - **Altura vertical da mão**: Controla o *Pitch Bend* contínuo (dobra a nota em frações de semitom para solos expressivos).
3. **Modo Tutor / Avaliador Pedagógico Inteligente**:
   - Acompanha o usuário compasso a compasso através de um repertório (*Ode à Alegria*, *Brilha Brilha Estrelinha*, *Asa Branca*).
   - Analisa desvios rítmicos e erros de notas, calculando a acurácia em tempo real.
   - Oferece dicas personalizadas da IA na tela (ex: *"Atenção na transição para a nota Mi4: você costuma hesitar"*) e ajusta dinamicamente a velocidade (BPM adaptativo).

---

## ⚡ Arquitetura de Baixa Latência (Híbrida: Rust + Python)

```
                    ┌────────────────────────┐
                    │  Webcam (30 / 60 FPS)  │
                    └───────────┬────────────┘
                                │ Frames BGR
                                ▼
         ┌─────────────────────────────────────────────────┐
         │             PYTHON VISION & IA LAYER            │
         │  - MediaPipe Tasks (HandLandmarker XNNPACK)     │
         │  - Extração de Cinemática e Landmarks           │
         │  - Classificador de Gestos ML (KNN)             │
         │  - Modo Tutor & Avaliação em Tempo Real         │
         │  - HUD Holográfico de Realidade Aumentada (OpenCV)│
         └──────────────────────┬──────────────────────────┘
                                │ UDP Local (127.0.0.1:8765)
                                │ Latência < 0.05 ms
                                ▼
         ┌─────────────────────────────────────────────────┐
         │             RUST AUDIO ENGINE (cpal)            │
         │  - Síntese Polifônica (8 vozes paralelas)       │
         │  - Formas de Onda: Saw, Sine, Square, Triangle  │
         │  - Envelopes ADSR sem estalos (anti-clicking)   │
         │  - Filtro Passa-Baixas Ressonante (LowPass SVF) │
         │  - Efeito Delay Analógico Estéreo               │
         │  - Buffer Estrito em Hardware (Zero GC)         │
         └──────────────────────┬──────────────────────────┘
                                │ Áudio PCM F32
                                ▼
                    ┌────────────────────────┐
                    │   Alto-Falantes / Fone │
                    └────────────────────────┘
```

---

## 🚀 Como Executar

Basta executar o script unificado na raiz do projeto:

```bash
cd /home/bernardo/Projetos/gestures-music-ai
./run.sh
```

O script inicia automaticamente:
1. O executável de áudio de alta performance em Rust em background.
2. O ambiente virtual Python com a visão computacional e o HUD holográfico.
3. Ao fechar a janela do programa (`q`), o processo de áudio em Rust é encerrado de forma limpa.

---

## 🎮 Controles e Atalhos

| Tecla | Função |
| :---: | :--- |
| **`ESPAÇO`** | Liga / Desliga o **Modo Tutor Inteligente** |
| **`T`** | Alterna entre as músicas do repertório do Tutor |
| **`W`** | Alterna a forma de onda do sintetizador (*Sawtooth*, *Sine*, *Square*, *Triangle*) |
| **`E`** | Alterna a escala musical (*Dó Maior*, *Pentatônica A Menor*, *Blues*, *Akebono*) |
| **`R`** | *All Notes Off* (silencia notas ativas) |
| **`Q` ou `ESC`** | Encerra o programa |

### Interação com as Mãos:
- **Mão Direita (Melodia)**: Aponte com o dedo indicador para tocar nas teclas flutuantes holográficas no ar.
- **Mão Esquerda (Expressão)**:
  - Abra e feche os dedos para modular o filtro de corte (*Cutoff*).
  - Suba e desça a mão para modular o *Pitch Bend* contínuo.

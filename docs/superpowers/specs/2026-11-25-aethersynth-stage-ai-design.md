# AetherSynth Stage AI — Especificação de Design

**Data:** 2026-11-25  
**Repositório:** `/home/bernardo/Projetos/gestures-music-ai`  
**Status:** Aprovado  

---

## 1. Intenção e Motivação

Transformar o **AetherSynth** de um protótipo de teclado aéreo de notas soltas em um **Instrumento de Sinais de Acordes de Palco e Loop Session Inteligente**, atendendo simultaneamente a dois requisitos centrais:

1. **Requisito Acadêmico (Tópicos Avançados — TADS):** Consolidar o uso de **Inteligência Artificial autoral, real e demonstrável ao vivo** (sem heurísticas `if/elif` interceptando ou substituindo o classificador, e sem modelos prontos fechados para gestos).
2. **Requisito Prático de Desempenho:** Eliminar os gargalos atuais de latência (~100–150 ms somados entre fila padrão da webcam V4L2, buffer de áudio `cpal` e contenção de `Mutex` em Rust) e otimizar o pipeline gráfico em CPU Linux de 4 núcleos para manter 30+ FPS estáveis com resposta musical imediata.

---

## 2. Objetivos, Não-Objetivos e Anti-Objetivos

### Objetivos
- **Redução drástica de latência ponta a ponta:**
  - Captura assíncrona da webcam em thread dedicada (`CAP_V4L2`, `MJPG`, `BUFFERSIZE=1`) entregando sempre o frame mais recente.
  - Suavização cinemática com **Filtro 1€ (One-Euro Filter)** para zerar tremedeira sem atrasar movimentos rápidos.
  - Motor de áudio em Rust **100% lock-free** (sem `Arc<Mutex<PolySynth>>`), com `BufferSize::Fixed(256)` (~5,3 ms a 48 kHz) e *fallback* automático.
- **Consolidação de 3 módulos reais de IA / processamento inteligente:**
  1. **Rede Neural MLP (`MLPClassifier`, arquitetura `30 -> 64 -> 32 -> 8`)** para reconhecer os 7 sinais musicais de acordes de palco + punho fechado (`Fist`), invariante à rotação, escala e translação, executada em matrizes NumPy puras (~0,01 ms por inferência) e equipada com **Modo Calibração Ao Vivo (`[K]`)** (*few-shot online retraining*).
  2. **Detector Probabilístico de Tom e Campo Harmônico** (correlação de perfis tonais Krumhansl-Kessler ponderada por duração e posição métrica + verossimilhança diatônica Bayesiana nas 24 tonalidades), travando automaticamente os pads melódicos na escala do tom detectado e destacando as notas do acorde atual (*chord tones*).
  3. **Rastreador Cinemático de Pulso Rítmico** na mão esquerda via picos de velocidade vertical filtrados pelo Filtro 1€, permitindo *tap-tempo* vivo (BPM) e acento rítmico nos acordes.
- **Loop Session *sample-accurate* em Rust:**
  - Gravação da base harmônica (acordes) e overdub de melodia diretamente no callback DSP, com 16 vozes polifônicas e *time-stretch* contínuo de BPM sem alteração de afinação.
- **Otimização da HUD Liquid Glass (`vision_ai/hud.py`):**
  - Eliminar loops pesados de desenho por frame e caches desnecessários para reduzir carga de CPU na renderização 720p.

### Não-Objetivos
- Exportação de projetos para DAWs, gravação de áudio em disco (`.wav`/`.mid`) ou carregamento de *soundfonts*/samples externos.
- Reconhecimento de gestos dinâmicos longos baseados em janelas temporais de vários segundos (ex.: desenhar letras no ar).
- Suporte a múltiplas câmeras ou clientes remotos fora do loopback `127.0.0.1:8765`.

### Anti-Objetivos (Critérios de Falha)
- Manter desvios `if/elif` na frente do classificador de ML (invalida o uso real de IA).
- Introduzir janelas de atraso perceptíveis (> 30 ms) entre gesto e ataque sonoro.
- Causar *dropouts*, cliques ou travamentos no stream de áudio por bloqueio de `Mutex` ou alocações no callback `cpal`.
- Adicionar dependências externas pesadas fora das bibliotecas já instaladas no ambiente.

---

## 3. Restrições Técnicas

- **Hardware alvo:** Máquina Linux com CPU de 4 núcleos (sem dependência obrigatória de GPU dedicada). O pipeline de visão + IA + HUD deve operar de forma fluida a 30+ FPS; a inferência do classificador de gestos deve consumir `< 0,2 ms` por mão.
- **Zero novas dependências externas:**
  - **Python (`vision_ai/requirements.txt`):** usar estritamente `opencv-python`, `mediapipe`, `numpy`, `scikit-learn` (e biblioteca padrão Python).
  - **Rust (`audio_engine/Cargo.toml`):** usar estritamente `cpal`, `serde`, `serde_json` e biblioteca padrão (`std::sync::atomic`, `std::sync::mpsc`).

---

## 4. Abordagens Consideradas

| Abordagem | Descrição Resumida | Trade-off | Status |
| :--- | :--- | :--- | :--- |
| **1. Pipeline Desacoplado + MLP Invariante (NumPy) + Looper Rust + IA Harmônica** | Rede Neural própria sobre 30 atributos 3D invariantes à rotação (~0,01 ms em NumPy) + orientação angular para qualidade/acidente + calibração ao vivo `[K]` + Krumhansl-Kessler Bayesiano + motor Rust *lock-free*. | Foca em posturas + cinemática instantânea em vez de trajetórias longas. | **Escolhida** |
| **2. Classificação Temporal por Janela Deslizante** | Rede/classificador analisando sequências de 10–15 frames para capturar gestos no tempo. | Adiciona 150–250 ms intrínsecos de espera para fechar a janela, prejudicando a resposta como instrumento. | **Descartada** |
| **3. `GestureRecognizer` Fechado do MediaPipe + IA apenas Harmônica** | Uso do modelo TFLite pronto do Google para gestos. | Preso a 7 gestos genéricos que não cobrem os sinais de músicos (`Dó`..`Si`) e enfraquece a autoria de IA na avaliação. | **Descartada** |

---

## 5. Decisões Definidas vs. Escolhas de Engenharia

### 5.1 Decidido pelo Usuário
- **Linguagem de Sinais de Acordes (Palco):**
  - **Dó:** 1 dedo estendido (indicador).
  - **Ré:** 2 dedos estendidos (indicador + médio).
  - **Mi:** 3 dedos estendidos (indicador + médio + anelar).
  - **Fá:** 4 dedos estendidos (indicador, médio, anelar e mínimo; polegar recolhido).
  - **Sol:** 5 dedos abertos (mão espalmada).
  - **Lá:** sinal de *hang-loose* (polegar + mínimo estendidos).
  - **Si:** apenas polegar estendido (demais dedos fechados).
  - **Qualidade / Acidente por Orientação da Mão:**
    - Apontando **para cima (`↑`)** = **Maior**.
    - Apontando **para baixo (`↓`)** = **Menor**.
    - Posição **horizontal** = **Sustenido / Bemol** (ex.: `F#`, `Bb`).
- **Abordagem 1** adotada para IA rápida e redução de latência.
- **Modo Loop Session** permitindo gravar bases harmônicas, criar melodias por cima (*overdub*) e usar **gestos pulsantes** para alterar o ritmo.
- **Detecção automática do tom da música** a partir da harmonia gravada no loop, travando a escala para evitar notas fora do tom durante a criação de melodias/harmonias.
- **Menu Principal estilo jogo + Interface 100% clicável por mouse:** Iniciar em um menu inicial interativo antes de abrir a câmera, com botões clicáveis (`cv2.setMouseCallback`) em todas as telas (`MAIN_MENU`, `HOW_TO_PLAY`, `CALIBRATION`, `STAGE`), mantendo atalhos de teclado apenas como conveniência secundária.

### 5.2 Escolhas de Engenharia (Propostas pelo Projetista)

#### A. IA de Sinais de Acordes em 2 Estágios (`vision_ai/gestures.py`)
1. **Estágio A — Classificação de Forma Intrínseca (MLP `30 -> 64 -> 32 -> 8`):**
   - Extração de **30 features geométricas 3D invariantes à translação, escala e rotação no plano** a partir dos 21 landmarks do MediaPipe (`(x, y, z)` normalizados pelo tamanho da palma $L = \|P_9 - P_0\|_3$, com piso `1e-5`):
     - **15 cossenos de ângulos articulares 3D:** para cada um dos 5 dedos, ângulos em `MCP`, `PIP` e `DIP` (ou `CMC/MCP/IP` no polegar), calculados via produto escalar normalizado dos vetores ósseos adjacentes.
     - **5 distâncias ponta–pulso normalizadas:** $\|P_{\text{tip}} - P_0\|_3 / L$ para `[4, 8, 12, 16, 20]`.
     - **5 distâncias ponta–base do dedo (MCP) normalizadas:** $\|P_{\text{tip}} - P_{\text{mcp}}\|_3 / L$, incluindo abdução do polegar (`4` até `5`) e fechamento de cada dedo (`[8-5, 12-9, 16-13, 20-17]`).
     - **5 relações interdigitais normalizadas:** distâncias entre pontas adjacentes (`4-8, 8-12, 12-16, 16-20`) e polegar–centro da palma (`4-9`) — essenciais para separar com folga **Fá** (polegar cruzado/recolhido) de **Sol** (polegar abduzido), e **Si** (polegar estendido) de **Fist** (punho fechado).
   - **Treinamento & Inferência:**
     - `MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=400, early_stopping=True, random_state=42)` com `StandardScaler` embutido (média e desvio salvos junto aos pesos).
     - Dataset base sintético-cinemático realista com **300 amostras por classe (2.400 amostras totais)** geradas com variação anatômica, leve flexão, perspectiva 3D e ruído gaussiano, garantindo que o modelo reconheça as 8 classes (`0: Fist/Mute`, `1: Dó`, `2: Ré`, `3: Mi`, `4: Fá`, `5: Sol`, `6: Lá`, `7: Si`) desde a primeira execução.
     - Pesos exportados para `vision_ai/models/chord_mlp_weights.npz` (`mean`, `scale`, `W0..W2`, `b0..b2`).
     - **Forward pass em runtime** escrito diretamente em NumPy (`x_norm = (x - mean) / scale`, ReLU nas duas camadas ocultas, Softmax na saída) — custo medido < 0,02 ms por mão, **zero `if/elif` antes ou depois para sobrescrever a classe da forma**.
   - **Calibração Ao Vivo (`[K]`):**
     - Ao pressionar `K`, entra em fluxo guiado na própria HUD pedindo ~25–30 frames por classe (pode pular classes ou cancelar com `ESC`). As amostras reais da mão do usuário são concatenadas ao conjunto base (com peso maior/oversampling local `3x`), a MLP é retreinada em ~0,2–0,4 s em background/pausa breve, e os pesos + amostras do usuário são persistidos em `vision_ai/models/user_samples.npz` e `chord_mlp_weights.npz`.

2. **Estágio B — Orientação Espacial e Estabilização (`ChordInterpreter`):**
   - **Vetor direcional por classe (Decisão de Engenharia):**
     - Para as classes `1..6` (**Dó, Ré, Mi, Fá, Sol, Lá**), usa-se o vetor axial da palma no frame espelhado: $\vec{d} = (P_{9,x} - P_{0,x},\; P_{9,y} - P_{0,y})$.
     - Exclusivamente para a classe `7` (**Si** — apenas polegar estendido, como na imagem de referência), usa-se o vetor do polegar $\vec{d}_{\text{Si}} = (P_{4,x} - P_{2,x},\; P_{4,y} - P_{2,y})$ para evitar ambiguidade com o eixo da palma ao rotacionar a mão para `Bb`.
   - Ângulo em tela $\alpha = \text{atan2}(-d_y, d_x) \in (-180^\circ, 180^\circ]$:
     - **Cima (`↑`):** $\alpha \in [45^\circ, 135^\circ]$ $\rightarrow$ **Maior** (tônica + intervalos `[0, 4, 7]`, baixo em `-12`).
     - **Baixo (`↓`):** $\alpha \in [-135^\circ, -45^\circ]$ $\rightarrow$ **Menor** (tônica + intervalos `[0, 3, 7]`, baixo em `-12`).
     - **Direita (`→`):** $\alpha \in (-45^\circ, 45^\circ)$ $\rightarrow$ **Sustenido Maior (`#`)** (tônica `+1` semitom, tríade maior — ex.: forma **Fá** apontando para a direita = `F#`).
     - **Esquerda (`←`):** $\alpha \in (135^\circ, 180^\circ] \cup (-180^\circ, -135^\circ)$ $\rightarrow$ **Bemol Maior (`b`)** (tônica `-1` semitom, tríade maior — ex.: forma **Si** apontando para a esquerda = `Bb`).
   - **Estabilização temporal anti-*flicker*:**
     - Histerese angular de `±10°` ao sair do setor de orientação ativo.
     - Voto de estabilidade: exige **3 frames consecutivos** com a mesma classe (ou confiança Softmax $\ge 0.78$ mantida por 2 frames) antes de disparar `chord_on`; se a confiança cair abaixo de `0.60`, mantém o último estado válido por até 2 frames antes de considerar transição.
     - Classe `0 (Fist)` envia `chord_off` imediatamente (após 2 frames para evitar corte acidental).

#### B. Detector Probabilístico de Tom e Travamento de Escala (`vision_ai/harmony_ai.py`)
- Mantém um histograma de energia das **12 classes de altura (pitch classes `0..11`)** construído a partir dos acordes gravados no Loop Session (ou, enquanto o loop estiver vazio, a partir de uma janela decrescente dos últimos acordes tocados ao vivo).
- Para cada acorde na base:
  - Contribui para sua tríade (`root`, `third`, `fifth`) com pesos `[1.0, 0.7, 0.8]` multiplicados pela duração do evento em batidas e por um fator de peso métrico (`1.25x` se iniciar próximo ao tempo 1 do compasso).
- Pontua as **24 tonalidades candidatas** (12 Maiores + 12 Menores Naturais) combinando:
  1. Correlação de Pearson normalizada com os perfis cognitivos de **Krumhansl-Kessler** (`MAJOR_PROFILE`, `MINOR_PROFILE`).
  2. Verossimilhança diatônica (fração das notas do acorde e raízes que pertencem ao campo harmônico daquela tonalidade + bônus de cadência/tônica quando o primeiro ou último acorde do loop coincide com o grau I).
- Normaliza via Softmax com temperatura para exibir na HUD o **tom vencedor e a confiança percentual** (ex.: `TOM: Dó Maior (94%)`).
- **Travamento automático da escala:** Quando o loop possui ao menos 2 eventos de acorde (ou >= 1 acorde confirmado após fechar a base), a escala dos **8 pads superiores** é atualizada automaticamente para o modo diatônico do tom detectado (`[0, 2, 4, 5, 7, 9, 11, 12]` para Maior ou `[0, 2, 3, 5, 7, 8, 10, 12]` para Menor Natural, na oitava 4–5), garantindo que **todas as notas tocáveis estejam dentro do tom**. Os pads cujas notas pertencem à tríade do acorde que está soando na batida atual do loop recebem destaque visual extra (`chord tone` com borda/pulso diferenciado).

#### C. Rastreamento 1€ + Gesto Pulsante Rítmico (`vision_ai/tracker.py` e `vision_ai/rhythm.py`)
- **One-Euro Filter (`OneEuroFilter`):** Aplicado em Python sobre os landmarks normalizados `(x, y)` de cada mão detectada (parâmetros iniciais: `min_cutoff=1.4`, `beta=0.04`, `d_cutoff=1.0`, usando `time.monotonic()`), removendo tremor de baixa velocidade sem lag em gestos rápidos.
- **Papel da Mão Esquerda (Expressão + Pulso Rítmico):**
  - **Abertura da mão (`openness`):** modula continuamente o `cutoff` do filtro passa-baixas (`0.05..1.0`).
  - **Altura vertical com trava neutra (Decisão de Engenharia — isolamento de `pitch_bend`):** modula `pitch_bend` suavemente com zona morta central de `±0.08` em torno de `y = 0.5`. No motor Rust, cada `Voice` recebe um rótulo `VoiceRole` (`ChordLoop`, `MelodyLoop`, `Live`), de modo que `pitch_bend` é aplicado **apenas às vozes `Live` e `MelodyLoop`**, nunca desafinando a base de acordes em loop (`ChordLoop`) durante a execução de melodia ou pulsos rítmicos.
  - **Detector de Gesto Pulsante (`RhythmPulseDetector`):** Monitora a velocidade vertical descendente normalizada $v_y = \Delta y_{\text{palma}} / \Delta t$ (positiva para baixo no frame).
    - Dispara um evento de **pulso** quando $v_y$ cruza um limiar adaptativo superior (`v_thresh`, piso mínimo para ignorar micro-movimentos) após um vale/retorno ascendente, com tempo refratário mínimo de **220 ms** (teto ~270 BPM; janela válida de *tap-tempo* entre **50 e 180 BPM**, isto é, intervalos entre `0.333s` e `1.200s`).
    - Cada pulso envia `{"type": "pulse", "velocity": v_norm}` ao motor Rust para reatacar/acentuar ritmicamente o acorde ativo (ou o loop).
    - Se houver pelo menos 2 pulsos dentro de `[0.333s, 1.200s]`, calcula `bpm_inst = 60.0 / dt_pulso`, aplica atualização exponencial suave `bpm = 0.65 * bpm + 0.35 * bpm_inst` (clamp `[50.0, 180.0]`) e envia `{"type": "modulate", "bpm": bpm}` ao motor Rust.

#### D. Motor de Áudio Rust Lock-Free + Loop Session (`audio_engine/src/`)
1. **Arquitetura Lock-Free (Zero `Mutex` no callback de áudio):**
   - A thread principal cria um par `mpsc::channel::<EngineEvent>()` e um `Arc<SharedEngineState>` contendo apenas tipos atômicos (`AtomicU32` para `cutoff`, `pitch_bend`, `volume`, `resonance`, `bpm`, `beat_pos_bits`, `loop_len_bits`; `AtomicU8` para `loop_state`, `active_chord_root`, `active_chord_quality`; `AtomicBool` para `is_playing`).
   - A thread do `cpal` recebe a posse exclusiva (`move`) de `PolySynth` (expandido de **8 para 16 vozes**) + `LoopEngine` + `Receiver<EngineEvent>`.
   - No início de cada buffer do callback de áudio:
     - Drena todos os eventos pendentes via `while let Ok(ev) = rx.try_recv()` (sem bloquear nem alocar).
     - Lê os parâmetros contínuos atômicos com `Ordering::Relaxed`.
     - Avança o `LoopEngine` amostra a amostra em unidades de **batidas** ($\Delta \text{beat} = \frac{\text{bpm}}{60.0 \cdot \text{sample\_rate}}$), mantendo a afinação intacta quando o BPM varia (*time-stretch* rítmico puro).
     - Atualiza os atômicos de telemetria (`beat_pos`, `loop_len_beats`, `loop_state`, `active_chord_root`, `active_chord_quality`) ao final do buffer.
   - A thread do socket UDP (`127.0.0.1:8765`) decodifica JSON com `serde_json`, atualiza atômicos para parâmetros contínuos ou envia eventos discretos no canal `tx.send(...)`, e responde imediatamente a `ping` com o estado completo do looper:
     `{"status":"pong","loop_state":"playing","bpm":104.0,"beat_pos":2.35,"loop_len_beats":8.0,"active_chord_root":0,"active_chord_quality":1}`
2. **Baixa Latência de Hardware:**
   - Configura `StreamConfig` tentando `cpal::BufferSize::Fixed(256)` (~5,3 ms @ 48 kHz); se o backend ALSA/PipeWire recusar, faz *fallback* automático para `Fixed(512)` e depois `BufferSize::Default`.
   - Suporta `SampleFormat::F32`, `I16` e `U16` (convertendo internamente do mesmo pipeline `f32`).
   - Remove o método morto `Voice::stop` ou aproveita-o nos testes/voice-stealing sem warnings.
3. **Máquina de Estados e Quantização do `LoopEngine` (Decisões de Engenharia):**
   - Estados:
     - `Idle`: loop vazio; apenas toca entradas ao vivo.
     - `RecordingBase`: entra em estado **armado** ao pressionar `[ESPAÇO]` e **inicia o relógio de batidas (`beat = 0.0`) apenas quando o primeiro `chord_on` chega** (evitando silêncio morto inicial que deslocaria o tempo 1 e o peso métrico de `1.25x` na `HarmonyAI`), acumulando eventos `{ start_beat, duration_beats, root, quality, notes: [u8; 4], velocity }` na faixa de base. Se `[ESPAÇO]` for pressionado novamente sem nenhum acorde gravado, cancela e volta para `Idle`.
     - `Playing`: ao fechar `RecordingBase`, seja $T_{\text{raw}}$ a duração bruta em batidas desde o primeiro `chord_on`; quantiza-se o ciclo para o **múltiplo de 4 batidas mais próximo** ($T_q = \max(4.0,\; 4.0 \cdot \text{round}(T_{\text{raw}} / 4.0))$), aplicando **reescala temporal proporcional** (`scale = T_q / T_raw`) em todos os `start_beat` e `duration_beats` gravados na base (preservando o espaçamento rítmico da progressão sem cortar nem esticar apenas o último acorde). O Python aplica a mesma reescala proporcional ao espelho local enviado para `HarmonyAI` e para a timeline da HUD. Reproduz os eventos da base + eventos gravados na faixa de melodia (`overdub`) em ciclo contínuo.
     - `Overdubbing`: mantém o loop tocando e grava eventos melódicos (`note_on`/`note_off`) com sua posição em batidas relativas ao ciclo atual.
   - Ação de `Pulse { velocity }`: se estiver tocando acordes ao vivo (ou no loop), dispara um envelope curto de acento/reatacada no acorde corrente sem alterar a harmonia gravada.
   - Limpeza (`LoopClear`): envia `all_notes_off` apenas nas vozes do looper, zera eventos e retorna para `Idle`.

#### E. Protocolo UDP Atualizado (`vision_ai/udp_client.py` <-> `audio_engine/src/main.rs`)
Comandos JSON suportados (mantendo retrocompatibilidade com `note_on`, `note_off`, `all_notes_off`, `modulate`, `set_waveform`, `ping`):
- `{"type": "chord_on", "root": 0, "quality": "major", "notes": [48, 60, 64, 67], "velocity": 0.85}`
- `{"type": "chord_off"}`
- `{"type": "pulse", "velocity": 0.9}`
- `{"type": "loop_toggle"}` — cicla `Idle -> RecordingBase -> Playing -> Overdubbing <-> Playing`
- `{"type": "loop_clear"}`
- `{"type": "modulate", "cutoff": ..., "pitch_bend": ..., "volume": ..., "resonance": ..., "bpm": ...}`

#### F. Menu Estilo Jogo, Controles por Mouse e Guia de Progressões (`vision_ai/main.py`, `tutor.py`, `hud.py`)
- **Fluxo de Telas Estilo Jogo (Inicialização Preguiçosa da Câmera):**
  - `MAIN_MENU`: Tela inicial animada sem abrir a webcam ainda, exibindo o título **AetherSynth**, status do motor Rust DSP / MLP e 5 botões grandes clicáveis: `INICIAR LOOP SESSION`, `MODO DESAFIO (ACORDES)`, `CALIBRAR IA (MINHA MAO)`, `COMO JOGAR & TIMBRE`, `SAIR DO JOGO`.
  - `HOW_TO_PLAY`: Referência visual dos 7 sinais de acordes + `Punho`, 4 setas de orientação (`Maior`, `Menor`, `Sustenido #`, `Bemol b`), seletores de forma de onda (`SAW`, `SINE`, `SQUARE`, `TRIANGLE`) e botão `[ < MENU PRINCIPAL ]`.
  - `CALIBRATION`: Abre a câmera sob demanda, exibe 8 botões de classe (`Punho, Do, Re, Mi, Fa, Sol, La, Si`), `[ GRAVAR AMOSTRAS ]`, `[ TREINAR REDE MLP ]`, `[ PADRAO ]` e `[ < VOLTAR ]`.
  - `STAGE`: Abre a câmera sob demanda com barra superior clicável (`[ < MENU ]`, `TOM`, `BPM`, `[ ONDA ]`, `[ ESCALA ]`, `RUST DSP`), pads de escala no topo, timeline do Loop Session e dock inferior com botões clicáveis (`[ GRAVAR BASE / FECHAR LOOP / + OVERDUB / LOOP ATIVO ]`, `[ LIMPAR ]`, `[ MODO: ACORDES / MELODIA ]`, `[ DESAFIO ]`, `[ CALIBRAR IA ]`, `[ SILENCIAR ]`). Atalhos de teclado (`ESPAÇO`, `M`, `C`, `G`, `K`, `W`, `E`, `R`, `Q`/`ESC`) seguem funcionando como alternativa opcional.
- **Otimização da HUD (`vision_ai/hud.py`):**
  - Substituir o loop de ~45 `cv2.circle` do dial de Cutoff por uma única chamada `cv2.ellipse`.
  - Reduzir cópias repetidas de sub-regiões na composição de vidro translúcido, usar texto 7-bit ASCII nas fontes Hershey do OpenCV (evitando `?` em acentos) com ícones vetoriais desenhados por primitivas OpenCV, e cachear os retângulos dos 8 pads quando `(w, h, scale_notes)` não mudam.

### 5.3 Escopo Deixado ao Implementador
- Ajuste fino de ganhos relativos entre as vozes de acorde (baixo + tríade) e as vozes de melodia para evitar saturação no `tanh()` final.
- Constantes exatas de posicionamento, fontes e margens visuais na HUD 1280×720.
- Detalhes de texto pedagógico das progressões no Guia (`[G]`).

---

## 6. Critérios de Verificação

1. **Testes Unitários e Build Rust (`audio_engine`):**
   - `cargo test` passando cobrindo: filtro passa-baixas, envelopes ADSR, operação *lock-free* com drenagem de eventos `mpsc`, alocação/roubo em 16 vozes, quantização de loop em múltiplos de 4 batidas e variação de BPM sem alterar frequência das notas.
   - `cargo build --release` sem warnings de código morto.
2. **Testes Automatizados Python (`vision_ai`):**
   - Teste de acurácia e latência da MLP em `vision_ai/gestures.py` (inferência NumPy `< 0.5 ms`, acurácia de validação $\ge 95\%$ nas 8 classes e invariância sob rotação sintética de 90°/180°).
   - Teste de calibração ao vivo (*few-shot*) verificando persistência e recarga de `chord_mlp_weights.npz`.
   - Teste do detector de tom (`harmony_ai.py`) identificando corretamente progressões padrão (ex.: `C → Am → F → G` $\rightarrow$ `Dó Maior`; `Am → Dm → E` / `Am → F → C → G` $\rightarrow$ `Lá Menor`).
   - Teste do detector de pulso rítmico (`rhythm.py`) convergindo para o BPM correto dado sinal sintético periódico a 90 BPM e 120 BPM.
3. **Verificação Integrada / Smoke Test:**
   - Execução conjunta do binário Rust e cliente Python validando troca de mensagens UDP `chord_on`, `pulse`, `loop_toggle`, `modulate` e leitura de status do looper em `ping`.

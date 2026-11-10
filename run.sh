#!/usr/bin/env bash

# Diretório base do projeto
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUST_BIN="$PROJECT_DIR/audio_engine/target/release/audio_engine"
PYTHON_VENV="$PROJECT_DIR/vision_ai/venv/bin/python"
PYTHON_MAIN="$PROJECT_DIR/vision_ai/main.py"

echo "=========================================================="
echo "  Iniciando AetherSynth (Rust DSP + Python Vision HUD)"
echo "=========================================================="

# 1. Cargo recompila apenas quando o código do motor mudou.
echo "[INFO] Verificando motor de audio Rust..."
if ! (cd "$PROJECT_DIR/audio_engine" && "$HOME/.cargo/bin/cargo" build --release); then
    echo "[ERRO] Nao foi possivel compilar o motor de audio."
    exit 1
fi

# 2. Inicia o motor de áudio Rust em background
echo "[INFO] Iniciando Rust Audio Engine..."
"$RUST_BIN" &
RUST_PID=$!

# Função para garantir que o processo Rust seja encerrado ao sair
cleanup() {
    echo ""
    echo "[INFO] Encerrando Rust Audio Engine (PID: $RUST_PID)..."
    kill $RUST_PID 2>/dev/null
    wait $RUST_PID 2>/dev/null
    echo "[OK] Sistema finalizado com sucesso."
}
trap cleanup EXIT INT TERM

# Aguarda 0.5s para o socket UDP inicializar
sleep 0.5

# Configuração do backend gráfico para compatibilidade e silenciamento de avisos Qt
export QT_QPA_PLATFORM="xcb;wayland"

# 3. Inicia a interface Python com visão computacional e HUD
echo "[INFO] Iniciando Vision AI & Holographic HUD..."
"$PYTHON_VENV" "$PYTHON_MAIN"

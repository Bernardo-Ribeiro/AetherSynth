#!/usr/bin/env bash

# Diretório base do projeto
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUST_BIN="$PROJECT_DIR/audio_engine/target/release/audio_engine"
PYTHON_VENV="$PROJECT_DIR/vision_ai/venv/bin/python"
PYTHON_MAIN="$PROJECT_DIR/vision_ai/main.py"

echo "=========================================================="
echo "  Iniciando Gestures Music AI (Rust DSP + Python Vision)"
echo "=========================================================="

# 1. Verifica se o binário Rust existe, se não, compila
if [ ! -f "$RUST_BIN" ]; then
    echo "[INFO] Compilando o motor de audio Rust em release..."
    cd "$PROJECT_DIR/audio_engine" && $HOME/.cargo/bin/cargo build --release
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

#!/usr/bin/env bash
# One-shot setup for cytools-agent.
# Idempotent: safe to re-run.

set -euo pipefail

cd "$(dirname "$0")"

echo "==> 1. Conda environment"
# use conda from PATH; otherwise source it from a common install location
if ! command -v conda >/dev/null 2>&1; then
    for base in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
        if [ -f "$base/etc/profile.d/conda.sh" ]; then
            source "$base/etc/profile.d/conda.sh"
            break
        fi
    done
fi
command -v conda >/dev/null 2>&1 || { echo "    conda not found" >&2; exit 1; }

if conda env list | grep -qE "^cytools-agent\s"; then
    echo "    'cytools-agent' env already exists"
else
    conda env create -f environment.yml
fi

echo "==> 2. Ollama"
if ! command -v ollama >/dev/null 2>&1; then
    if [[ "$(uname)" == "Darwin" ]]; then
        echo "    installing via Homebrew (macOS)"
        brew install ollama
    else
        echo "    installing via ollama.com script (Linux; may prompt for sudo)"
        curl -fsSL https://ollama.com/install.sh | sh
    fi
else
    echo "    already installed: $(ollama --version 2>&1 | head -1)"
fi

# The agent needs a >=16k context window: Ollama's VRAM-based default (often
# 4096) silently FRONT-truncates long sessions -- the system prompt is lost
# first and the model appears to go stupid. The service config below bakes
# OLLAMA_CONTEXT_LENGTH in so every start (including after reboot) is correct.
echo "==> 3. Ollama server (as an always-on service, 16k context)"
if [[ "$(uname)" == "Darwin" ]]; then
    # launchd reads env via launchctl; brew services manages the daemon
    launchctl setenv OLLAMA_CONTEXT_LENGTH 16384
    brew services restart ollama >/dev/null
    echo "    brew service (re)started with OLLAMA_CONTEXT_LENGTH=16384"
elif command -v systemctl >/dev/null 2>&1 && [ -d /run/systemd/system ]; then
    # the official installer creates ollama.service; add our env as a drop-in
    # (survives ollama upgrades) and make sure it starts now and on boot.
    # Stop any hand-started `ollama serve` first -- it holds port 11434.
    pkill -u "$USER" -x ollama 2>/dev/null || true
    sudo mkdir -p /etc/systemd/system/ollama.service.d
    printf '[Service]\nEnvironment="OLLAMA_CONTEXT_LENGTH=16384"\n' | \
        sudo tee /etc/systemd/system/ollama.service.d/cytools-agent.conf >/dev/null
    sudo systemctl daemon-reload
    sudo systemctl enable --now ollama
    echo "    systemd service enabled (drop-in: OLLAMA_CONTEXT_LENGTH=16384)"
else
    # no service manager (e.g. a container): plain background process
    if ! curl -sf http://localhost:11434/api/version >/dev/null 2>&1; then
        echo "    no systemd; starting in background (log: /tmp/ollama.log)"
        OLLAMA_CONTEXT_LENGTH=16384 nohup ollama serve >/tmp/ollama.log 2>&1 &
    fi
fi
for i in {1..15}; do
    curl -sf http://localhost:11434/api/version >/dev/null 2>&1 && break
    sleep 1
done
curl -sf http://localhost:11434/api/version >/dev/null 2>&1 \
    || { echo "    Ollama did not come up -- check 'systemctl status ollama'" >&2; exit 1; }
echo "    server is up on localhost:11434"

echo "==> 4. Pull qwen3:8b (~5.2 GB; idempotent -- fast if cached)"
ollama pull qwen3:8b

echo
echo "Done. Launch the notebook with:"
echo "    conda activate cytools-agent && jupyter lab"
echo "then open notebooks/demo.ipynb (it runs in the env's Python 3 kernel)."

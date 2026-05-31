#!/usr/bin/env bash
# One-shot setup for cytools-agent.
# Idempotent: safe to re-run.
# Assumes the CYTools repo is a sibling: ~/cytools next to ~/cytools-agent.

set -euo pipefail

cd "$(dirname "$0")"

echo "==> 1. Conda environment"
source ~/miniforge3/etc/profile.d/conda.sh
if conda env list | grep -qE "^cytools-agent\s"; then
    echo "    'cytools-agent' env already exists"
else
    conda env create -f environment.yml
fi

echo "==> 2. Jupyter kernel"
conda run -n cytools-agent python -m ipykernel install --user --name cytools-agent --display-name "Python (cytools-agent)" >/dev/null
echo "    kernel 'Python (cytools-agent)' registered"

echo "==> 3. Ollama"
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

echo "==> 4. Ollama server"
if curl -sf http://localhost:11434/api/version >/dev/null 2>&1; then
    echo "    already running on localhost:11434"
else
    echo "    starting in background (log: /tmp/ollama.log)"
    nohup ollama serve >/tmp/ollama.log 2>&1 &
    for i in {1..10}; do
        sleep 1
        if curl -sf http://localhost:11434/api/version >/dev/null 2>&1; then
            break
        fi
    done
fi

echo "==> 5. Pull qwen2.5-coder:7b-instruct (~5 GB; idempotent — fast if cached)"
ollama pull qwen2.5-coder:7b-instruct

echo
echo "Done. Launch the notebook with:"
echo "    conda activate cytools-agent && jupyter lab"
echo "and select the 'Python (cytools-agent)' kernel (top right)."

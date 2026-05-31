# CYTools-Agent

A tool harness and agent loop for driving [CYTools](https://github.com/LiamMcAllisterGroup/cytools) with LLMs (local via Ollama, or any OpenAI-compatible API).

## Layout

- `cytools_agent/tools/` — the model-facing tools (`polytope`, `triangulation`, `code`) plus call logging (`history`)
- `cytools_agent/schema.py` — generates OpenAI tool schemas from a function's signature + docstring
- `cytools_agent/agent.py` — the agent loop (`Agent`) and the tool-call parser
- `notebooks/` — the demo notebook and any other interfaces
- `environment.yml` — the conda environment, used by *both* run paths below

## Running

Two ways. Both build the same environment from `environment.yml`.

### Docker (sandboxed, reproducible)

```sh
docker compose up --build
docker compose exec ollama ollama pull qwen2.5-coder:7b-instruct   # once, ~5 GB
```

Then open `http://localhost:8899/lab?token=cytools` and run `notebooks/demo.ipynb`.

The model's code (`run_python`) executes *inside* the container; the source is mounted read-only, so model code can't modify it.

### Non-Docker alternative (`setup.sh`)

`setup.sh` runs everything directly on the host instead of in containers — useful for faster iteration, using a host GPU, or when you don't want Docker. There is **no sandbox**: `run_python` executes on your machine.

```sh
./setup.sh                                  # builds the env, installs/starts Ollama, pulls the model
conda activate cytools-agent && jupyter lab
```

Select the **Python (cytools-agent)** kernel, then run `notebooks/demo.ipynb`.

## Notes

- The notebook reads `OLLAMA_HOST` (set in `docker-compose.yml`) and falls back to `http://localhost:11434`, so the same notebook works in both run paths.

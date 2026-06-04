# cytools-agent
*[Nate MacFadden](https://github.com/natemacfadden), Liam McAllister Group, Cornell*

An agent loop and tool harness that lets a local LLM (via [Ollama](https://ollama.com)) drive [CYTools](https://github.com/LiamMcAllisterGroup/cytools) — fetching polytopes, computing triangulations and Calabi-Yau invariants, running arbitrary CYTools code, and exporting the session as a standalone script.

## Installation

```sh
./setup.sh   # creates the conda env, installs Ollama, pulls the default model
conda activate cytools-agent
jupyter lab
```

Open `notebooks/demo.ipynb` with the **Python (cytools-agent)** kernel.

## Usage

```python
from openai import OpenAI
from cytools_agent.tools import (polytope, triangulation, cy, code, history)
from cytools_agent.schema import function_to_schema
from cytools_agent.agent import Agent
from cytools_agent.prompt import DEFAULT_SYSTEM_PROMPT

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
TOOL_FNS = [polytope.fetch_polytopes, polytope.get_polytope_info,
            polytope.ks_stats,
            triangulation.get_heights, triangulation.get_triangulation_info,
            cy.get_cy_info, cy.get_cy_cones,
            code.run_python, code.cytools_help,
            history.save_history]
tools = [function_to_schema(fn) for fn in TOOL_FNS]
tool_impls = {fn.__name__: fn for fn in TOOL_FNS}

agent = Agent(client, "qwen3:4b", DEFAULT_SYSTEM_PROMPT, tools, tool_impls,
              max_steps=20, verbosity=2)
agent.tool_impls["save_history"] = agent.save_script  # bind to this instance

print(agent.chat("Fetch 3 polytopes at h11=5 and compute CY volumes at the Kähler cone tip."))
agent.save_script("session.py")  # export as a runnable script
```

`.chat()` is stateful — history accumulates across calls.

## Tools

| Tool | Description |
|---|---|
| `fetch_polytopes` | Fetch polytopes from the Kreuzer-Skarke database by Hodge numbers |
| `get_polytope_info` | Hodge numbers, favorability, face counts, automorphism order, etc. |
| `ks_stats` | Count polytopes at (h11, h21) — check existence before fetching |
| `get_heights` | Height vectors for triangulations (NTFE or FRST), returns shape metadata |
| `get_triangulation_info` | Validate a triangulation and get simplex count |
| `get_cy_info` | CY invariants (intersection numbers, Chern classes); optionally volumes at a Kähler point |
| `get_cy_cones` | Mori cone rays (= Kähler cone hyperplane normals) |
| `run_python` | Execute arbitrary Python in a persistent CYTools namespace |
| `cytools_help` | Look up CYTools signatures/docstrings without running code |
| `save_history` | Write the session as a standalone runnable Python script |

## Evaluation

```sh
python eval/eval.py qwen3:8b 30                       # sample 30 corpus questions
python eval/eval.py qwen3:8b --ids 54,57,58 --reps 3  # targeted re-run
python eval/agent_tests.py qwen3:8b 3                 # behavioral test suite
python eval/corpus.py verify                          # check corpus integrity
```

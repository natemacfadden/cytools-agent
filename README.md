# cytools-agent
*[Nate MacFadden](https://github.com/natemacfadden), Liam McAllister Group, Cornell*

An agent loop and tool harness that lets a local LLM (via [Ollama](https://ollama.com)) drive [CYTools](https://github.com/LiamMcAllisterGroup/cytools) -- fetching polytopes, computing triangulations and Calabi-Yau invariants, running arbitrary CYTools code, and exporting the session as a standalone script.

> **WARNING -- no sandbox.** The `run_python` tool executes model-generated code directly on your machine, with no isolation. Run only models and prompts you trust, on a machine where that is acceptable. There is currently no containment around what the model can read, write, or execute.

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
from cytools_agent.tools import (polytope, triangulation, cy, code)
from cytools_agent.schema import function_to_schema
from cytools_agent.agent import Agent
from cytools_agent.prompt import DEFAULT_SYSTEM_PROMPT

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
TOOL_FNS = [polytope.fetch_polytopes, polytope.get_polytope_info,
            polytope.ks_stats,
            triangulation.get_heights, triangulation.get_triangulation_info,
            cy.get_cy_info, cy.get_cy_cones,
            code.run_python, code.cytools_help]
tools = [function_to_schema(fn) for fn in TOOL_FNS]
tool_impls = {fn.__name__: fn for fn in TOOL_FNS}

agent = Agent(client, "qwen3:8b", DEFAULT_SYSTEM_PROMPT, tools, tool_impls,
              max_steps=20, verbosity=2)
# save_history is auto-registered on Agent and available to the model

print(agent.chat("Fetch 3 polytopes at h11=5 and compute CY volumes at the Kahler cone tip."))
agent.save_history("session.py")  # or let the model call it
```

`.chat()` is stateful -- history accumulates across calls.

## Tools

Polytopes are referenced by canonical string id (`h11-X_h21-Y_ind-Z`);
triangulations are passed as height vectors. Tools take ids/heights, not live
objects.

| Tool | Description |
|---|---|
| `fetch_polytopes(limit, h11, h21=None, favorable=None)` | Up to `limit` Kreuzer-Skarke polytope ids at the given Hodge numbers. `favorable=True/False` scans deeper until `limit` matches found. Cached. |
| `get_polytope_info(ks_ind)` | One polytope's geometry: Hodge numbers, Euler char, N/M favorability, trilayer flag, automorphism order, point/vertex counts, `facedim_to_nfaces` (faces per dimension). |
| `ks_stats(h11, h21=None)` | Exact polytope count at `(h11[, h21])` from a precomputed table over the full KS database (473M+). Check existence/counts before fetching. |
| `get_heights(ks_ind, n=None, kind="NTFE", effort=0.5)` | Triangulations as `{"shape": [n_tri, n_pts], "heights": [...]}` (count = `shape[0]`). `n` set -> random sample of `n`; `n` omitted -> all inequivalent (`kind="NTFE"`) or all fine-regular-star (`kind="FRST"`). `effort` refuses too-large polytopes. |
| `get_triangulation_info(ks_ind, heights)` | The triangulation from `heights`: fine/regular/star/valid flags, hash, simplex count. |
| `get_cy_info(ks_ind, heights, t=None, cone="Kcup")` | CY invariants from a triangulation: Hodge numbers, Euler char, second Chern class, nonzero in-basis triple intersections, prime-toric-divisor count. `t` (a Kahler-cone point, or `"tip"`) adds divisor + CY volumes there (after membership check). |
| `get_cy_cones(ks_ind, heights, cone="Kcup")` | Mori cone rays = dual Kahler cone hyperplane normals. `cone="Kcup"` accurate, `"toric"` cheaper at large h11. |
| `run_python(code)` | Arbitrary Python in a persistent namespace (preloaded `cytools`, `numpy`, the tools, `get_polytope`/`get_cy`). Escape hatch; captures stdout, auto-saves figures. |
| `cytools_help(name)` | Resolve a dotted name string (e.g. `"Polytope.triangulate"`) in the `run_python` namespace -> signature + docstring. API discovery without running code. |
| `save_history(path)` | Session as a standalone runnable script: tool calls as code, agent text as comments. Auto-registered on every `Agent`. |

## Evaluation

Run from the repo root (module form, since `eval/` is a package):

```sh
python -m eval.eval qwen3:8b 30                       # sample 30 corpus questions
python -m eval.eval qwen3:8b --ids 54,57,58 --reps 3  # targeted re-run
python -m eval.agent_tests qwen3:8b 3                 # behavioral test suite
python -m eval.corpus verify                          # check corpus integrity
```

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

Polytopes are identified by canonical string ids of the form
`h11-X_h21-Y_ind-Z`; triangulations are passed around as height vectors. Most
tools take an id (and, for CY-level tools, a height vector) so the agent never
holds a live object across turns.

| Tool | Description |
|---|---|
| `fetch_polytopes(limit, h11, h21=None, favorable=None)` | Fetch up to `limit` 4d reflexive polytopes from the Kreuzer-Skarke database at the given Hodge numbers; returns their ids. `favorable=True/False` scans deeper into the list until `limit` matches are found. Results are cached, so repeat fetches are free. |
| `get_polytope_info(ks_ind)` | Geometry of one polytope by id: Hodge numbers, Euler characteristic, N- and M-favorability, trilayer flag, automorphism-group order, point/vertex counts, and `facedim_to_nfaces` (how many faces of each dimension). |
| `ks_stats(h11, h21=None)` | Look up the exact number of polytopes at `(h11[, h21])` in the full Kreuzer-Skarke database (473M+ polytopes) from a precomputed count table. Use it to check existence/counts before fetching instead of guessing. |
| `get_heights(ks_ind, n=None, kind="NTFE", effort=0.5)` | Generate triangulations as height vectors, returned as `{"shape": [n_tri, n_pts], "heights": [...]}` (read the count from `shape[0]`). `n` given -> fast random sample of `n`; `n` omitted -> ALL inequivalent triangulations (`kind="NTFE"`, default) or all fine regular star ones (`kind="FRST"`). Exhaustive modes are guarded by `effort`, which refuses polytopes too large to enumerate. |
| `get_triangulation_info(ks_ind, heights)` | Build the triangulation selected by `heights` and report its fine / regular / star / valid flags, a hash identity, and simplex count. |
| `get_cy_info(ks_ind, heights, t=None, cone="Kcup")` | Build the Calabi-Yau from a triangulation and return its invariants: Hodge numbers, Euler characteristic, second Chern class, nonzero in-basis triple intersection numbers, and prime-toric-divisor count. If `t` is given (a point in the Kahler cone, or `"tip"` for the stretched-cone tip) it also checks membership and returns divisor and CY volumes there. |
| `get_cy_cones(ks_ind, heights, cone="Kcup")` | Return the Mori cone rays of the CY, which are exactly the hyperplane normals of the dual Kahler cone. `cone="Kcup"` is accurate; `"toric"` is cheaper at large h11. |
| `run_python(code)` | Execute arbitrary Python in a persistent namespace preloaded with `cytools`, `numpy`, the other tools, and `get_polytope` / `get_cy` for raw objects. The escape hatch for anything without a dedicated tool; captures stdout and auto-saves any matplotlib figures. |
| `cytools_help(name)` | API discovery without running code: resolves a dotted name string (e.g. `"Polytope.triangulate"`) against the `run_python` namespace and returns its call signature and docstring. |
| `save_history(path)` | Write the session so far as a standalone, runnable Python script -- tool calls replayed as code, the agent's text as comments. Auto-registered on every `Agent`. |

## Evaluation

Run from the repo root (module form, since `eval/` is a package):

```sh
python -m eval.eval qwen3:8b 30                       # sample 30 corpus questions
python -m eval.eval qwen3:8b --ids 54,57,58 --reps 3  # targeted re-run
python -m eval.agent_tests qwen3:8b 3                 # behavioral test suite
python -m eval.corpus verify                          # check corpus integrity
```

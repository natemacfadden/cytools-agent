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

## Demo

About the most complex thing `qwen3:8b` will carry to completion -- a single
prompt chaining a database fetch, a 10-polytope loop, exhaustive triangulation,
simplex counting, and a plot:

```python
agent.chat("For the first 10 polytopes at h11=6, how many NTFEs do each have? "
           "Plot me a distribution of simplex counts.")
```

On a good run the model fetches the ids, then writes one `run_python` block that
loops over them, enumerates each polytope's NTFE triangulations, counts
simplices, and saves a histogram:

```
The first 10 polytopes at h11=6 have NTFE counts [1, 3, 6, 1, 2, 2, 4, 4, 2, 4].
A histogram of the simplex counts is saved as fig_1.png.
```

(`[1, 3, 6, 1, 2, 2, 4, 4, 2, 4]` is exactly correct.) This sits near the
model's ceiling, but lands a clean run most of the time it doesn't time out
(~3 in 5 measured). Earlier it was far worse: the model would write correct
`run_python` code, forget to `print()` the result, get `(no output)`, and then
fabricate counts -- so `run_python` now echoes a trailing bare expression and,
when nothing is printed, names the variables the code assigned, and the system
prompt forbids reporting unprinted values. Smaller, focused prompts (single
fetch, one CY invariant, one aggregation) are reliable. See
`eval/agent_tests.py` for the full set of end-to-end cases.

## Tools

Polytopes are referenced by canonical string id (`h11-X_h21-Y_ind-Z`);
triangulations are passed as height vectors. Tools take ids/heights, not live
objects.

> **Note.** These tools are *model-read*: their docstrings are written for the
> LLM and are sent verbatim as its tool schema (so editing a docstring changes
> what the model sees). Functions are tagged `# model-read` or `# human-read`
> in the source to make this boundary explicit; everything under `eval/` is
> human-read.

| Tool | Description |
|---|---|
| `save_history(path)` | Session as a standalone runnable script: tool calls as code, agent text as comments. Auto-registered on every `Agent`. |
| `cytools_help(name)` | Resolve a dotted name string (e.g. `"Polytope.triangulate"`) in the `run_python` namespace -> signature + docstring. API discovery without running code. |
| `ks_stats(h11, h21=None)` | Exact polytope count at `(h11[, h21])` from a precomputed table over the full KS database (473M+). Check existence/counts before fetching. |
| `fetch_polytopes(limit, h11, h21=None, favorable=None)` | Up to `limit` Kreuzer-Skarke polytope ids at the given Hodge numbers. `favorable=True/False` scans deeper until `limit` matches found. Cached. |
| `get_polytope_info(ks_ind)` | One polytope's geometry: Hodge numbers, Euler char, N/M favorability, trilayer flag, automorphism order, point/vertex counts, `facedim_to_nfaces` (faces per dimension). |
| `get_heights(ks_ind, n=None, kind="NTFE", effort=0.5)` | Triangulations as `{"shape": [n_tri, n_pts], "heights": [...]}` (count = `shape[0]`). `n` set -> random sample of `n`; `n` omitted -> all inequivalent (`kind="NTFE"`) or all fine-regular-star (`kind="FRST"`). `effort` refuses too-large polytopes. |
| `get_triangulation_info(ks_ind, heights)` | The triangulation from `heights`: fine/regular/star/valid flags, hash, simplex count. |
| `get_cy_info(ks_ind, heights, t=None, cone="Kcup")` | CY invariants from a triangulation: Hodge numbers, Euler char, second Chern class, nonzero in-basis triple intersections, prime-toric-divisor count. `t` (a Kahler-cone point, or `"tip"`) adds divisor volumes, CY volume, and curve volumes (+ `min_curve_volume`) there (after a cone-membership check). Pass a list of heights -> a list of results. |
| `get_cy_cones(ks_ind, heights, cone="Kcup")` | Mori cone rays = dual Kahler cone hyperplane normals. `cone="Kcup"` accurate, `"toric"` cheaper at large h11. |
| `run_python(code)` | Arbitrary Python in a persistent namespace (preloaded `cytools`, `numpy`, the tools, `get_polytope`/`get_cy`). Escape hatch; captures stdout, auto-saves figures. |

## Use from Claude Code (MCP)

`mcp_server.py` exposes the same tools over MCP, so any MCP client (e.g. Claude
Code) can drive CYTools directly. It registers the same `MODEL_TOOLS` the
in-house agent uses, so the names, docstrings, and parameter schemas are
identical -- the tools behave the same whichever loop calls them. (`run_python`
is included; `save_history` is not -- the MCP client manages its own session.)

**Recommended: register once, use everywhere.** After `./setup.sh`, run this a
single time (the `cd` is only so `"$(pwd)"` resolves to the absolute path):

```sh
cd /path/to/cytools-agent
claude mcp add --scope user cytools -- \
  conda run --no-capture-output -n cytools-agent python "$(pwd)/mcp_server.py"
```

That writes a user-scope entry to `~/.claude.json`, so `cytools` and its 9 tools
are available in **every** Claude Code session, from any directory -- no `cd`,
no per-repo file. Then just ask in plain language ("fetch 3 favorable polytopes
at h11=5 and give their CY volumes") and Claude calls the tools. Verify with
`/mcp`.

`conda run -n cytools-agent` runs the server inside the env without activating
it (needs only `conda` on PATH + the env to exist); `--no-capture-output` is
required so `conda` doesn't buffer stdout, which is the stdio protocol channel.

**Alternative: per-repo.** The committed `.mcp.json` registers the same server
at project scope -- launch Claude Code from inside the repo and approve it on
first open. Shareable via git and survives repo moves (relative path), but only
active when the repo is your working directory.

**Updating:** the package is an editable install, so `git pull` needs no
reinstall -- just restart/reconnect the server (`/mcp`) to load new code. Run
`conda env update -f environment.yml` only when dependencies change; re-run the
`claude mcp add` above only if you move the repo.

## Evaluation

`eval/corpus.jsonl` is the **corpus**: ~100 questions about specific polytopes
/ CYs, each with a known answer and the standalone code that produces it (e.g.
"how many NTFE triangulations does `h11-5_h21-20_ind-0` have?" -> `142`). It
measures whether the agent, given only the question, reaches that answer with
its tools. Each entry has an integer **id**.

Run from the repo root (module form, since `eval/` is a package):

```sh
# run the agent on a random sample of 30 corpus questions; grade vs known answers
python -m eval.eval qwen3:8b 30

# re-run specific corpus ids 3x each (to gauge a fix, since a local model is
# nondeterministic); reports pass/fail/timeout per id
python -m eval.eval qwen3:8b --ids 54,57,58 --reps 3

# behavioral suite: hand-written cases checking not just the answer but HOW the
# agent worked -- did it call the right tools, avoid hallucinating, etc.
python -m eval.agent_tests qwen3:8b 3

# integrity check: re-execute each corpus entry's stored code and confirm it
# still reproduces the stored answer
python -m eval.corpus verify
```

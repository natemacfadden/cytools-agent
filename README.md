# cytools-agent
*[Nate MacFadden](https://github.com/natemacfadden), Liam McAllister Group, Cornell*

An agent loop and tool harness that lets a local LLM (via [Ollama](https://ollama.com)) drive [CYTools](https://github.com/LiamMcAllisterGroup/cytools) -- fetching polytopes, computing triangulations and Calabi-Yau invariants, running arbitrary CYTools code, and exporting the session as a standalone script.

> **WARNING -- no sandbox.** The `run_python` tool executes model-generated code directly on your machine, with no isolation. Run only models and prompts you trust, on a machine where that is acceptable.

## Installation

```sh
./setup.sh   # conda env + Ollama as an always-on service + default model
conda activate cytools-agent
jupyter lab
```

Open `notebooks/demo.ipynb` with the **Python (cytools-agent)** kernel.

`setup.sh` is idempotent and sets Ollama up as a system service (starts on boot, restarts on crashes, configured with the context window the agent needs; one `sudo` prompt on Linux). After setup there is nothing to start or remember.

## Quick start

Ask research questions through the **orchestrator** -- it plans, executes with the tools, and backs every number with a harness-written evidence log:

```python
from cytools_agent.orchestrator import run_session
print(run_session("Among the first 100 polytopes at h11=3, plot the distribution of NTFE triangulation counts and report the mean."))
```

Questions can sweep Hodge numbers ("at each h11 in [2,10]"), ask for several figures at once, color a scatter by a third quantity, and search ("the largest h11 such that..."). For follow-ups that build on earlier answers, use the stateful chat:

```python
from cytools_agent.orchestrator import OrchestratorChat
chat = OrchestratorChat(model="qwen3:8b")
chat.chat("Fetch the first 25 polytopes at h11=3 and their NTFE counts.")
chat.chat("Now scatter those counts against the polytopes' h21 values.")
```

**Watch it live:** `python -m cytools_agent.viewer`, then open http://127.0.0.1:8765 -- the plan, each step's code and real output, and figures render as the session runs. Archived sessions are browsable there too; `python -m cytools_agent.viewer export` bakes one into a shareable standalone HTML.

**Extra confidence:** `run_session_voted(question, votes=3)` runs independent sessions and accepts an answer only when the final numbers agree; disagreement is flagged rather than silently picked.

There is also a lighter single-agent chat loop (`cytools_agent.agent.Agent`) -- see the notebook for setup. It handles focused questions (one fetch, one invariant, one aggregation) reliably; the orchestrator exists for everything bigger.

## Tools

Polytopes are referenced by canonical string id (`h11-X_h21-Y_ind-Z`); triangulations are passed as height vectors. Tools take ids/heights, not live objects.

> **Note.** These tools are *model-read*: their docstrings are sent verbatim to the LLM as its tool schema, so editing a docstring changes what the model sees. Functions are tagged `# model-read` or `# human-read` in the source; everything under `eval/` is human-read.

| Tool | Description |
|---|---|
| `ks_stats(h11, h21=None)` | Exact polytope count at `(h11[, h21])` from a precomputed table over the full KS database (473M+). Check existence/counts before fetching. |
| `fetch_polytopes(limit, h11, h21=None, favorable=None)` | Up to `limit` Kreuzer-Skarke polytope ids at the given Hodge numbers. Rate-limited and budgeted per session (the database is a shared academic resource). |
| `get_polytope_info(ks_ind)` | One polytope's geometry: Hodge numbers, Euler char, favorability, automorphism order, point/vertex/face counts, 2-face genera. |
| `get_heights(ks_ind, n=None, kind="NTFE", effort=0.5)` | Triangulations as `{"shape": [n_tri, n_pts], "heights": [...]}`. `n` set: a random sample; omitted: all inequivalent (`NTFE`) or all fine-regular-star (`FRST`). `effort` refuses too-large enumerations. |
| `get_triangulation_info(ks_ind, heights)` | Fine/regular/star/valid flags, hash, simplex count for the triangulation defined by `heights`. |
| `get_cy_info(ks_ind, heights, t=None, cone="Kcup")` | CY invariants: Hodge numbers, Euler char, second Chern class, triple intersections, divisor count. `t` (a Kahler-cone point or `"tip"`) adds divisor/CY/curve volumes there. |
| `get_cy_cones(ks_ind, heights, cone="Kcup")` | Mori cone rays = dual Kahler cone hyperplane normals. `"Kcup"` accurate, `"toric"` cheaper at large h11. |
| `compute_for_each(ks_inds, exprs)` | Harness-side iteration: evaluates each expression once per id and stores aligned lists plus exact stats (n/mean/min/max/sum). The model writes a one-item expression; the harness does the loop. |
| `make_plot(kind, x, y, ...)` | Builds and saves a scatter/histogram/line/bar figure from stored list names, with computed data facts (ranges, correlation, outliers) returned alongside. |
| `search_polytopes(condition, objective)` | Budget-aware search over h11 levels for polytopes satisfying a condition ("largest h11 such that..."), reporting exactly what was and was not checked. |
| `run_python(code)` | Arbitrary Python in a persistent namespace (preloaded `cytools`, `numpy`, the tools). Escape hatch; captures stdout, auto-saves figures, wall-clock capped. |
| `cytools_help(name)` | Signature + docstring for a dotted name (e.g. `"Polytope.triangulate"`). API discovery without running code. |
| `cy_glossary(term)` | Domain term -> definition + the exact recipe to compute it with these tools. |

## Use from Claude Code (MCP)

`mcp_server.py` exposes the same tools over MCP, so any MCP client (e.g. Claude Code) can drive CYTools directly, with identical names, docstrings, and schemas. Register once, use everywhere:

```sh
cd /path/to/cytools-agent
claude mcp add --scope user cytools -- \
  conda run --no-capture-output -n cytools-agent python "$(pwd)/mcp_server.py"
```

This makes the tools available in every Claude Code session from any directory; verify with `/mcp`. (`--no-capture-output` is required: stdout is the stdio protocol channel.) Alternatively, the committed `.mcp.json` registers the server at project scope when Claude Code is launched from inside the repo.

The package is an editable install: after `git pull`, just reconnect the server (`/mcp`). Run `conda env update -f environment.yml` only when dependencies change.

## Evaluation

`eval/corpus.jsonl` holds ~100 single-fact questions with known answers and the standalone code that produces them; `eval/pm_corpus.jsonl` holds 10 hard multi-step research problems; `eval/ladder.jsonl` is a 6-rung difficulty ladder. Run from the repo root:

```sh
python -m eval.eval qwen3:8b 30                                # corpus sample, graded
python -m eval.eval qwen3:8b --ids 54,57,58 --reps 3           # targeted re-runs
python -m eval.corpus verify                                   # every stored answer still reproduces
python -m eval.eval_orch --ids 3,4,6,9 --reps 3 --model qwen3:8b   # orchestrator on the hard corpus
python -m eval.eval_orch --corpus eval/ladder.jsonl --reps 5   # capability profile by rung
python -m eval.eval_single_pm qwen3:8b --ids 3,4,6,9 --reps 3  # single-agent comparison
python -m eval.verify_glossary                                 # invariants + recipes admission gate
```

## Under the hood

Normal use needs none of this; `setup.sh` configures everything.

The orchestrator wraps the tools in scaffolding that measurably lifts small local models (qwen3:8b: 0% to ~40% single-run, ~80% voted, on the hard plot corpus). Each piece is a flag, on by default, `=0` to disable:

| Flag | What it does |
|---|---|
| `CYTOOLS_SCHEMA_ACT` | Model replies decoded under a JSON Schema, so malformed or empty replies cannot be sampled at all. |
| `CYTOOLS_PIPELINE` | Questions fitting fetch -> map -> reduce -> plot (or search) compile to a typed spec the harness executes deterministically; misfits fall back to the plan-and-walk path. |
| `CYTOOLS_MAP_TOOLS` | `compute_for_each` / `make_plot` / `search_polytopes`. |
| `CYTOOLS_FINISH_FORGIVE` | Accept `answer = ...` scratchpad assignment as the step finish signal (grounding still enforced). |
| `CYTOOLS_NUM_CTX` | Per-request context size (default 16384; `0` = server default). |
| `CYTOOLS_KS_BUDGET` | Real database queries allowed per session (default 40; also `CYTOOLS_KS_MIN_INTERVAL`, `CYTOOLS_KS_MAX_LIMIT`). |
| `CYTOOLS_RUN_TIMEOUT` | Wall-clock cap on one `run_python` call (default 150 s). |
| `CYTOOLS_AGENT_KS_CACHE` | Opt-in (default off): persist fetched polytopes across runs. Dev feature; grows large. |

Every tool call is recorded in a harness-written ledger (exact arguments and structured results -- models can read rows but never write them), answers cite the rows backing their numbers, and computed data is audited against machine-checked identities (`cytools_agent/tools/invariants.py`). The protocol between the PM, the engineer, and the check layers is documented in `cytools_agent/orchestrator/PROTOCOL.md`; the A/B record behind each design choice is in `scratch/AB_RESULTS.md`.

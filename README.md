# cytools-agent
*[Nate MacFadden](https://github.com/natemacfadden), Liam McAllister Group, Cornell*

An agent loop and tool harness that lets a local LLM (via [Ollama](https://ollama.com)) drive [CYTools](https://github.com/LiamMcAllisterGroup/cytools) -- fetching polytopes, computing triangulations and Calabi-Yau invariants, and running arbitrary CYTools code.

We don't trust the model's CYTools knowledge, so it comes from a curated glossary via RAG. ([How it works](#how-it-works))

> **WARNING -- no sandbox.** `run_python` executes model-generated code directly on your machine, with no isolation, and `compute_for_each` / `search_polytopes` likewise `eval()` model-written expressions. Run only models and prompts you trust, on a machine where that is acceptable. `eval/eval_claude.py` grants these tools to headless runs by default -- pass `--no-code` to withhold them.

## Installation

```sh
./setup.sh   # conda env + Ollama as an always-on service + default model
conda activate cytools-agent
jupyter lab
```

Open `notebooks/demo.ipynb` -- launched from the activated env, it runs in the default **Python 3** kernel.

`setup.sh` is idempotent and sets Ollama up as a system service (starts on boot, restarts on crashes, configured with the context window the agent needs; one `sudo` prompt on Linux). After setup there is nothing to start or remember.

## Quick start

The zero-setup way to use it is over MCP: register the tools once and drive them from Claude Code or any MCP client (see [Use from Claude Code](#use-from-claude-code-mcp) below).

To drive a *local* model yourself, use the agent loop. `notebooks/demo.ipynb` has the full setup -- assemble the tool list, point an OpenAI-compatible client at Ollama, then:

```python
from cytools_agent import Agent
agent = Agent(client, model, system_prompt, tools, tool_impls)
print(agent.chat("Is the first polytope at h11=3 favorable in the N lattice?"))
```

It handles focused questions -- a fetch, an invariant, an aggregation -- and can iterate over many polytopes and build plots through the `compute_for_each` / `make_plot` / `search_polytopes` tools.

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
| `reference(query="")` | Searchable, indexed reference book -- the glossary's long-form companion for background a one-line definition can't carry. |
| `find_kahler_for_divisor_volumes(ks_ind, target)` | Solve for Kahler parameters at which the CY's divisor volumes match `target` (the inverse of reading volumes off a given point). |

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

Three question corpora live under `eval/`, differing by who wrote them and whether the answers are known:

| Corpus | Questions | Answers | Written by |
|---|---|---|---|
| `eval/corpus.jsonl` | ~100 single-fact | known, with the code that reproduces each | agent-extracted from CYTools example notebooks |
| `eval/pm_corpus.jsonl` | 10 hard multi-step plot/research | known | agent-authored during development |
| `eval/heldout.jsonl` | 25 authentic research questions | held out (computed later) | **human-written** (frozen; source `n8`) |

(`eval/ladder.jsonl` is a 6-rung difficulty ladder, not a research corpus.) Run from the repo root -- pick the line for the corpus you want:

```sh
# agent-written, known answers -> auto-graded against the stored truths
python -m eval.eval qwen3:8b 30                       # stratified sample of corpus.jsonl
python -m eval.eval qwen3:8b --ids 54,57,58 --reps 3  # targeted re-runs
python -m eval.corpus verify                          # confirm every stored answer still reproduces

# agent-written hard multi-step problems -> plain agent loop, auto-graded
python -m eval.eval_single_pm qwen3:8b --corpus eval/pm_corpus.jsonl --reps 3

# human-written questions: truths are held out, so each run is RECORDED (status RUBRIC,
# graded by hand later) rather than auto-scored. The system ladder is the runner that
# handles null answers; it works on ANY corpus, holding the model + questions fixed and
# varying one stack layer per rung L0-L2 (see diagnostics/README.md).
python -m eval.system_ladder --rung L2 --corpus eval/heldout.jsonl --model qwen3:8b

python -m eval.verify_glossary                        # invariants + recipes admission gate
```

The system ladder writes self-describing result files (rung, model, corpus, commit, seed, date + per-question results) to `diagnostics/system_ladder/`, never overwriting a prior run. Result files are local and not committed; headline numbers will be committed once the eval corpus matures.

## How it works

### Design principles

Everything here is aimed at getting a *weak* model to do *correct* work:

- The system prompt stays minimal. Generic, always-on text acts as noise as much as signal, so behavior is shaped through the tools and harness instead.
- Guidance is directed and stateful: a detailed error message delivered mid-computation, while the model is attempting one concrete thing, lands far better than a general instruction read long ago. Most error messages here are written for the model, not the developer.
- The tool boundary is forgiving. If a call is unambiguous to a human, support it rather than reject it -- accept the synonym, the stray kwarg, the slightly-off form, and steer from there.
- The model is never trusted to recall or compute; domain facts come from retrieval and values from the real tools, not its memory.

### Flow of a query

The model runs a plain tool-use loop: it calls the curated tools (or writes code with `run_python`), reads each real result, and keeps going until it has the answer. Mistakes come back as tool errors written for the model, so it corrects on the next step instead of failing silently.

### Knowledge from source (RAG)

The glossary maps each CYTools term to a definition and the recipe to compute it. For each request the harness retrieves the relevant entries and adds them to the prompt (`glossary_context`), so the model answers from the glossary and real CYTools docstrings, not its own memory. Retrieval is hybrid: keyword matching plus embeddings (`BAAI/bge-small-en-v1.5`), which catches paraphrases the keywords miss; without `sentence-transformers` it falls back to keyword-only.

An offline gate (`eval/verify_glossary.py`) runs every recipe against the live library, so the glossary can't drift out of date.

### Flags

Normal use needs none of the knobs below; `setup.sh` configures everything. They are flags, on by default, `=0` to disable:

| Flag | What it does |
|---|---|
| `CYTOOLS_MAP_TOOLS` | `compute_for_each` / `make_plot` / `search_polytopes` -- harness-side iteration, plotting, and search. |
| `CYTOOLS_KS_BUDGET` | Real database queries allowed per session (default 40; also `CYTOOLS_KS_MIN_INTERVAL`, `CYTOOLS_KS_MAX_LIMIT`). |
| `CYTOOLS_RUN_TIMEOUT` | Wall-clock cap on one `run_python` call (default 150 s). |
| `CYTOOLS_AGENT_KS_CACHE` / `CYTOOLS_AGENT_KS_BASE` | Opt-in (default off): the writable overlay and read-only trusted base of the persisted polytope cache. Dev feature; grows large. |

## License

[GPLv3](LICENSE). Copyright (c) 2026 Nate MacFadden.

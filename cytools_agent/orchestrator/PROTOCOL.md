# Orchestrator protocol

How a question becomes an answer: the enforced message formats first, then the
check layers. File references are to this package unless noted.

## Actors and channels

There is one model (the `model` argument, e.g. qwen3:8b). "PM" and
"engineer" are two roles it plays, distinguished only by system prompt and
conversation shape:

- Each **PM** action (translate, compile, plan, the judge calls, summarize)
  is a separate stateless call: fresh system prompt + one user message, no
  memory between calls. State lives in the harness, not the PM.
- The **engineer** is the only stateful conversation: within one dispatched
  step it accumulates messages (its replies + the outputs of the code it
  ran) until the step finishes or the budget ends. It is also the only role
  whose output causes execution.

All calls go through Ollama's native `/api/chat` endpoint (not the
OpenAI-compatible `/v1`) because only the native API supports the three
things the harness depends on: per-request `num_ctx` (defeats silent context
truncation), `think=false` (suppresses chain-of-thought, ~40x faster), and
`format=<JSON Schema>` -- structured outputs, where the schema compiles to a
decoding grammar that masks logits, so a malformed reply cannot be sampled
at all, rather than merely being discouraged. Every message format below is
enforced this way. (The env var `CYTOOLS_SCHEMA_ACT` controls this
enforcement: on by default, set it to 0 to fall back to ordinary
tool-calling.)

## The flow

Sequential, with one decision point. The question is always translated
first; then the pipeline path is attempted; the plan-and-walk path runs only
if that attempt is abandoned.

First, the two templates ("shapes") a question can be pressed into. The
"spec" is a fill-in-the-blanks form (full field listing in section 2) that
describes the work as one of:

- Shape A (map): fetch polytopes by Hodge numbers -> compute 1-3 named
  quantities once per polytope -> aggregate them -> optionally plot them.
- Shape B (search): sweep h11 levels looking for any polytope satisfying a
  per-polytope condition (largest h11 / smallest h11 / any).

The compile step asks the model to fill that form -- or to say the question
fits neither template, in which case it sets `fits=false` in the form.
"Recompile" below always means: issue the same compile call again, with the
failure reason appended to the request text, so the model fills the form a
second time knowing exactly what was wrong with its first attempt. Every
recompile happens at most once per run.

The normal path, which most fitted questions take end to end:

```
translate (1 LLM call): restate the question plainly, unpack jargon
    |
pipeline compile (1 LLM call): the model fills the spec form (shape A or B)
    |
did the pipeline attempt succeed? ---- no ----> walk (defined below)
    |
   yes
    |
harness executes the spec deterministically
(fetch / compute_for_each / reduce / make_plot / search_polytopes; no LLM)
    |
answer composed in code from the computed values
    |
final verification (defined below)
```

"Did the pipeline attempt succeed?" is no in exactly three cases, checked
in this order; everything else is yes:

1. The model answered `fits=false` (the question matches neither shape A
   nor shape B). Straight to the walk, no retry.
2. The spec fails validation (section "check layers", layer 2). One
   recompile with the validation failure appended; if the new spec is also
   invalid, the walk.
3. Execution of a valid spec raises an exception partway. One recompile
   with the error appended; if the new spec is invalid or also raises, the
   walk.

The walk -- the fallback path, entered only through those three cases:

```
plan (1 LLM call): 2-5 {do, produce} steps
    |
for each step, in order:
    dispatch -> engineer loop (run code, observe, repeat)
    per-step checks (check layer 5 below: did the step really make its
    declared `produce`, and address this step?); on failure, one
    corrective re-dispatch naming what was wrong;
    if the retry also fails, stop the walk (remaining steps abandoned)
    |
summarize (1 LLM call): answer written from the evidence log,
with an honesty note if the walk stopped early
    |
final verification
```

Final verification -- both paths end here:

```
verify_answer (1 LLM call): is every asked deliverable present?
missing something -> one redraft naming it; still missing -> append a
visible caveat
    |
return the answer
```

The two paths compose answers differently -- pipeline: programmatically from
computed values; walk: an LLM summary of the evidence log -- and both funnel
through the same final verification.

### 1. translate (PM, 1 LLM call)

Restates the question in plain words, unpacking jargon via the glossary and --
in chat -- resolving references ("those polytopes") against prior turns.
Measured over 411 archived sessions, the restatement rewrites heavily
(median character-similarity to the question: 0.44). The rewriting cuts both
ways: typically it appends useful glossary definitions, but it has also
paraphrased a deliverable away ("scatter" -> "pair", blinding a shape guard)
and occasionally invents function-call instructions it was told not to give.
Defenses: a runaway restatement is length-capped and replaced by the raw
question, and all shape guards check the raw question too, never only the
restatement. Note: the stage's net value has never been A/B-tested (no arm
has run with translate disabled); it is kept on convention, not evidence.

### 2. The pipeline fast path (PM, 1 compile call; harness executes)

One schema-constrained call fills the spec form:

```
{ fits:  bool,
  fetch: {h11: int|[int], h21, limit, favorable, use_stored},
  map:   {name: "<one-item Python expr over ks_ind>", ...},   # 1-3 columns
  reduce:[{name, op: mean|min|max|sum|count|argmax|argmin|ids_where_positive, of}],
  plot:  [{kind, x, y, color, logx, logy}, ...],              # <=4 figures
  search:{map: {name: expr}, condition: "<bool expr over names>",
          objective: largest_h11|smallest_h11|any} }          # shape B
```

If the spec validates (next section), the harness executes it
deterministically -- fetch, `compute_for_each`, Python reductions,
`make_plot`, or `search_polytopes` -- and the answer string is composed
programmatically from computed values (no LLM sits between the numbers and
the prose). Any misfit or runtime failure falls back to the walk after one
recompile.

### 3. plan (PM, 1 call, only if the pipeline declined)

Schema: `{todo: [{do: str, produce: str}]}`, 2-5 items enforced by the
decoding grammar; code-level back-stops remain (`_coerce_steps`, retry x3,
`_force_two_steps`, `_ensure_deliverable` appends a dropped plot step).
`produce` is load-bearing: it is what the per-step checks verify against.

### 4. dispatch (PM -> engineer, plain text, fixed form)

Each step is sent as one user message:

```
<tool cheatsheet (signatures + sampled worked example)>
<evidence log so far (rendered, truncated)>
[variables already in the scratchpad: ...]
STEP i/n -- DO: <step.do>
PRODUCE: <step.produce>
<glossary recipes for terms detected in the step>
```

(The capitalized labels here are literal text in the prompt, not emphasis.)

### 5. act (engineer's reply -- the one enforced format)

The engineer's whole reply is decoded under:

```
{ reflection: str,        # interpretation of previous output (pressure valve)
  intent:     str,        # required, what this step does
  code:       str,        # required, Python for run_python
  done:       bool,       # required every turn
  answer:     str }       # the result, when done
```

All three required fields must exist every turn, including the final one;
`code` may be the empty string when the model is only finishing. The
combinations and their meanings:

- `done=false`, code non-empty: an ordinary work turn.
- `done=true`, code non-empty: run the code first, then finish if the
  answer is grounded in its output -- the usual ending ("print the final
  value and stop" in one turn).
- `done=true`, code empty: finish on the strength of earlier observations.
- Requiring `done` every turn is deliberate: the model must re-decide "am I
  finished?" each turn, which is what closed the old never-signals-done
  failure mode.

The message intentionally spans two timesteps -- `reflection` looks
backward, `intent`/`code` look forward. That is because the evidence log
wants each observation to carry an interpretation, and output N can only be
interpreted on turn N+1: the harness writes turn N+1's `reflection` into
observation N's `interpretation` field. One act message therefore closes the
previous observation and opens the next.

The harness runs `code` (timeout-capped), appends an observation
`{intent, ran_code, received_output, interpretation}` to `evidence.jsonl` --
`ran_code`/`received_output` are harness-captured ground truth, the rest are
the model's claims -- and returns the output as the next user message. Loop
until `done` with a grounded answer, or the budget ends the round
(14 steps; traceback steps refunded; hard cap +8; repeated/no-op code breaks
the loop). Finish-forgiveness: an `answer` variable assigned in the
scratchpad counts as the done-signal (still subject to the grounding check).

### 6. compose + return (PM)

`summarize()` writes the user answer from the evidence log only (with an
honesty clause if the walk stopped early); pipeline-path answers skip this
stage entirely. `run_session_voted(q, votes=N)` wraps the whole protocol in
numeric self-consistency.

## The check layers (outermost to innermost)

1. **Decode-time** -- the three schemas above. Protocol violations
   (prose-instead-of-call, empty reply, missing done) cannot be emitted.
2. **Spec validation (deterministic, pre-execution)** -- expressions must
   parse; plot/reduce columns must exist (this turn's map, or stored
   columns in chat); fetch limits bounded; search conditions may not
   reference undefined names. Plus shape guards against silent narrowing:
   question sweeps h11 but spec holds a single value; question says plot but
   spec has none. Plus the quantity lint: glossary terms detected in the
   question map to recipe markers; a spec using a *different* quantity's
   markers is rejected with the correct recipe quoted verbatim. One
   recompile with the failure reason, then fall back.
3. **Run-time resource guards** -- KS database budget (40 real queries/
   session), 1.5 s spacing, 5000/query size cap, merge-safe shared cache;
   `run_python` wall-clock cap (150 s); `compute_for_each` skips erroring
   items without breaking column alignment.
4. **Evidence-time (anti-fabrication)** -- `grounded()`: a finishing answer
   must appear in a `received_output` whose code actually computes (an AST
   check rejects `print(42)`-style typed literals). Plot deliverables are
   grounded by the `[saved N figure(s)]` marker, which only the harness can
   write.
5. **Post-step (PM judges the engineer)** -- `_produce_met` (programmatic:
   plot steps need the figure marker), `verify_produce` (LLM: right
   type/shape for `produce`), `addresses` (LLM: drift gate, lenient by
   design). One corrective re-dispatch, then the walk stops honestly.
6. **Post-answer** -- `verify_answer` (LLM: every asked deliverable present?)
   triggers one redraft naming what was missing; if still incomplete, the
   caveat is appended visibly. Voting flags disagreeing runs as low
   confidence.

## The static-analysis machinery under the layers

Several checks above say "programmatic" or "deterministic"; concretely, most
of them parse the model's code into a Python syntax tree (the `ast` module)
and inspect its structure -- which makes them immune to wording variation,
unlike string matching or an LLM judge. Where each is used:

- **Expression well-formedness** (layer 2, `pipeline._valid`): every map
  expression, search helper, and search condition must parse as a single
  Python expression before anything executes.
- **Undefined-name detection** (layer 2, `pipeline._valid`): walk the
  condition's tree, collect every variable name, subtract the defined
  helpers, preloaded tools, builtins, and `ks_ind`; anything left is a
  dangling shorthand the model used without defining. Rejected with a
  fill-this-slot instruction naming the loose name.
- **Quantity lint** (layer 2; opt-in engineer variant): identifiers are
  extracted from the spec's expressions and compared against the marker sets
  derived from glossary recipes (dict fields like `['genera_2face']` and
  method names, regex-extracted from each recipe). A spec whose identifiers
  belong to a *different* glossary quantity than the question names is
  rejected, with the right recipe quoted.
- **Typed-literal rejection** (layer 4, `evidence._prints_only_literals`):
  walk the code that produced a candidate answer; if everything it prints is
  a constant (`print(42)`) rather than a name or call, the "answer" was
  typed, not computed, and grounding rejects it.
- **No-op narration detection** (engineer loop-breaker,
  `engineer._is_noop_print`): code consisting only of `print(<literal>)`
  statements is bucketed as a no-op regardless of the printed text, so
  varying the narration cannot dodge the repeated-code detector.
- **Assignment extraction** (`code._assigned_names`): when a `run_python`
  call prints nothing, the tree's top-level assignment targets are listed
  back to the model ("You assigned: counts -- print the value you need"),
  turning a silent step into a pointed hint.
- **Syntax validity per observation** (`evidence.valid_python`): each
  observation records whether its `ran_code` parses at all, so diagnostics
  can separate "wrote broken code" from "wrote working code that failed".

(The remaining non-LLM checks are regex- or marker-based rather than tree-
based: the figure-saved marker, the h11-range and plot-words shape guards,
and the answer-number extraction used by voting.)

## Known weak point

When the pipeline declines and the walk improvises on a question it cannot
structure (notably search-shaped questions), layers 4-6 catch most but not
all confabulation: a wrong number that genuinely appeared in printed output
passes `grounded()`. Voting is the current mitigation; refusing the walk for
search-shaped questions is the proposed stricter policy.

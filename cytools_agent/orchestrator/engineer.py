# =============================================================================
#    Copyright (C) 2026  Nate MacFadden for the Liam McAllister Group
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  The engineer: it implements one dispatched task by repeatedly
#               calling a single `act` tool (intent + code, plus a reflection
#               on the previous output). `act` is a protocol, not a function --
#               run_engineer is its interpreter: it runs the code, records the
#               observation (ran_code/received_output are harness-captured),
#               and feeds the output back. This module also hosts the shared
#               native-Ollama transport used by the PM.
# -----------------------------------------------------------------------------

# external imports
import ast
import json
import os
import re
import time
import urllib.request

# local imports
from cytools_agent.agent import _strip_template_tags, extract_tool_call
from cytools_agent.tools import code as _code
from cytools_agent.orchestrator.evidence import (emit, grounded, valid_python,
                                                 write_evidence)

OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


# model-read: the engineer's instructions, the API it may call, and its tool
# ---------------------------------------------------------------------------
ENGINEER_SYSTEM = (
    "You are the engineer for a project manager. Solve the dispatched task by "
    "calling the `act` tool one step at a time (its fields are described on "
    "the tool); never answer in plain text and never finish without having "
    "run code. The run_python namespace is a persistent SCRATCHPAD -- assign "
    "intermediate results to named variables and build on them. Compute the "
    "requested quantity EXPLICITLY in code and print exactly it: read the field "
    "THIS task needs from the right call and reduce it, e.g. info = "
    "get_polytope_info(ks_ind); print(REDUCE(info[FIELD])) -- where you replace "
    "FIELD and REDUCE with the dict key and reduction (max/min/sum/len/mean/...) "
    "your task actually requires. Do not print a big dump and read it off by "
    "eye, and do NOT read answers off a polytope id. Do ONLY the dispatched "
    "task -- do not wander into related questions "
    "or compute quantities it did not ask for. If a step only fetches or "
    "builds intermediate objects for later steps (polytopes, triangulations, "
    "CYs), finish by reporting a brief confirmation -- their count via "
    "print(len(...)) -- not a raw dump of the objects, and never an invented "
    "number. Once the step is solved, FINISH it: "
    "call act with done=true and the result in `answer` -- do not keep "
    "re-printing the result to 'confirm' it. Never report a number you did "
    "not see in an output. Write only in English."
)

# the API the engineer calls -- given so it does not GUESS signatures
TOOL_CHEATSHEET = (
    "Callable signatures (do not guess these):\n"
    "  fetch_polytopes(limit, h11, h21=None, favorable=None) -> list of ids "
    "(do not guess limit -- use ks_stats(h11) for the count; for FAVORABLE "
    "ones pass favorable=True; do not guess h21)\n"
    "  ks_stats(h11, h21=None) -> {count, exists, h21_values}; ks_stats() "
    "with NO args -> {total, h11_min, h11_max, count_by_h11} (whole-database "
    "census, local, no query)\n"
    "  reference(query) -> {glossary:[{term,definition,recipe}], api:[{name,"
    "signature,doc}]}; look up what a term MEANS or how a function works "
    "BEFORE guessing. reference() with no argument -> the table of contents "
    "(topic sections + terms); reference('<section title>') -> a whole "
    "section\n"
    "  get_polytope_info(ks_ind) -> {h11, h21, n_rigid_divisors, genera_2face, "
    "facedim_to_nfaces, ...}\n"
    "  get_heights(ks_ind, n=None, kind='NTFE', sampler='auto') -> {shape, "
    "heights}; the triangulations are get_heights(ks_ind)['heights'] (a list "
    "of height vectors); for big polytopes pass n=<count> to sample "
    "(sampler='gnn' = near-uniform, a sample NOT the census)\n"
    "  get_cy(ks_ind, heights=None) -> one CY (or a list for many heights)\n"
    "  get_cy_info(ks_ind, heights=None, t=None, cone='Kcup') -> dict (or a list "
    "for many heights); t='tip' adds cy_volume and curve_volumes (a list -- "
    "reduce it yourself, e.g. max(info['curve_volumes']))\n"
    "  get_cy_cones(ks_ind, heights=None, cone='Kcup') -> {mori_rays, "
    "kahler_cone_hyperplanes}\n"
    "For all inequivalent CYs of one polytope: get_cy_info(ks_ind, "
    "get_heights(ks_ind), t='tip') -> a list (aggregate it, e.g. "
    "min(min(r['curve_volumes']) for r in result))."
)


# A/B (CYTOOLS_MAP_TOOLS): harness-side iteration + plotting. The cheatsheet
# must advertise them or the engineer cannot know they exist. (Defined AFTER
# both cheatsheets -- this block appends to them.) The worked example's field
# is SAMPLED per process (see tools/_examples.py) so no single field becomes
# an attractor the model drifts into on novel questions.
from cytools_agent.tools._examples import example as _example
_EX_NAME, _EX_EXPR, _ = _example("map_cheat")
_MAP_CHEAT = (
    "\n  compute_for_each(ks_inds, {name: expression, ...}) -> evaluates each "
    "expression once PER id (with ks_ind bound) and stores aligned lists named "
    "`name` in the scratchpad; also returns stats (n/mean/min/max/sum) per "
    "numeric list -- report those numbers directly. USE THIS instead of "
    "writing your own loop over polytopes. Example (replace the field with "
    "what YOUR task needs): compute_for_each(ids, "
    f"{{'{_EX_NAME}': \"{_EX_EXPR}\"}})\n"
    "  make_plot(kind, x, y=None, xlabel='', ylabel='', title='') -> builds and "
    "saves the figure from stored list NAMES (e.g. make_plot(kind='scatter', "
    "x='h21', y='n_vertices')). USE THIS instead of writing matplotlib code.\n"
    "  search_polytopes(condition, objective='largest_h11') -> budget-aware "
    "search over h11 levels for a polytope satisfying `condition` (a boolean "
    "expression over ks_ind). USE THIS for 'largest/smallest h11 such that "
    "...' questions -- never write your own loop over h11 levels (the "
    "database is shared)."
)
from cytools_agent.tools.mapping import MAP_TOOLS_ENABLED
if MAP_TOOLS_ENABLED:
    TOOL_CHEATSHEET += _MAP_CHEAT

# A/B (CYTOOLS_SCHEMA_ACT): grammar-constrained act. Instead of advertising
# act as a TOOL and hoping the model emits a well-formed tool_call, the reply
# itself is decoded under this JSON Schema (Ollama structured outputs: the
# schema compiles to a grammar that masks logits). The model is then PHYSICALLY
# unable to reply with prose, an empty message, or a call missing `done`/
# `intent` -- the failure modes the recovery machinery below exists for.
# `reflection` is first so free-form interpretation has somewhere to go before
# the constrained fields (a pressure valve against railroading).
# Ollama-transport-specific BY DESIGN: a capable-model transport (e.g.
# Anthropic tool use) keeps the tool-call path; the act protocol is the same.
ACT_FORMAT = {
    "type": "object",
    "properties": {
        "reflection": {"type": "string"},
        "intent": {"type": "string", "minLength": 3},
        "code": {"type": "string"},
        "done": {"type": "boolean"},
        "answer": {"type": "string"},
    },
    "required": ["intent", "code", "done"],
}

ENGINEER_SYSTEM_SCHEMA_NOTE = (
    " Reply with EXACTLY ONE JSON act object per turn -- fields: reflection "
    "(interpretation of the previous output; empty first), intent (what this "
    "step does and why), code (Python to run now; empty string if only "
    "finishing), done (true when the task is solved), answer (the concrete "
    "result; set when done)."
)

# the engineer's one tool; hand-written so no dummy function is needed
_ACT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "act",
        "description": (
            "Take ONE engineering step and get the code's output back. Set "
            "reflection (interpretation of the previous output; empty first), "
            "intent (what you will do now and why), and code (Python to run "
            "now; CYTools tools are callable, the namespace persists, print "
            "what you need). When the task is fully solved, set done=true and "
            "put the concrete result in answer."),
        "parameters": {
            "type": "object",
            "properties": {
                "reflection": {"type": "string", "description":
                               "Interpretation of the previous step's output; "
                               "empty on the first step."},
                "intent": {"type": "string", "description":
                           "What you will do now and why."},
                "code": {"type": "string", "description":
                         "Python to run now; print what you need to see."},
                "done": {"type": "boolean", "description":
                         "True when the task is fully solved."},
                "answer": {"type": "string", "description":
                           "The concrete final result (set when done)."},
            },
            "required": ["intent"],
        },
    },
}


# native-Ollama transport (shared with the PM)
# ---------------------------------------------
def _ollama_chat(model, messages, think, tools=None, as_json=False, label=""):
    """One chat turn via Ollama's NATIVE /api/chat, which (unlike the
    OpenAI-compatible /v1 endpoint) honors `think`: think=False fully
    suppresses qwen3's chain-of-thought (~40x faster). Each call is timed and
    emitted as an `llm_call` event (label, seconds, think, prompt size) so per
    call latency is visible. Returns the message dict.

    as_json: True -> Ollama JSON mode (well-formed JSON, no schema). A DICT ->
    full structured output: Ollama compiles the JSON Schema to a decoding
    grammar and masks logits, so the reply is GUARANTEED to be a valid
    instance -- no prose, no missing required fields, no malformed JSON."""
    payload = {"model": model, "stream": False, "think": think,
               "messages": messages}
    # Ollama's vram-based default num_ctx (4096 here) SILENTLY truncates long
    # prompts from the front -- measured: an 8k-token chat returned
    # prompt_eval_count=4096 and the model lost the SYSTEM prompt. Late
    # engineer steps overflow 4096, so the act protocol itself falls out of
    # context. Default raised to 16384 (A/B-validated: removed all 600s
    # grinds); override with CYTOOLS_NUM_CTX, 0 to use the server default.
    num_ctx = int(os.environ.get("CYTOOLS_NUM_CTX", "16384") or 0)
    if num_ctx:
        payload["options"] = {"num_ctx": num_ctx}
    if tools:
        payload["tools"] = tools
    if isinstance(as_json, dict):
        payload["format"] = as_json          # schema-constrained decoding
    elif as_json:
        payload["format"] = "json"
    req = urllib.request.Request(
        OLLAMA_BASE + "/api/chat", json.dumps(payload).encode(),
        {"Content-Type": "application/json"})
    _t = time.monotonic()
    # a wedged Ollama call must not hang the session forever (eval_orch's
    # SIGALRM only covers eval runs, not interactive sessions)
    with urllib.request.urlopen(req, timeout=600) as resp:
        msg = json.loads(resp.read())["message"]
    emit("llm_call", label=label or "?", think=bool(think),
         s=round(time.monotonic() - _t, 2), n_msgs=len(messages),
         prompt_chars=sum(len(str(m.get("content") or "")) for m in messages))
    return msg


def _is_noop_print(code):
    """True if `code` is ONLY print()s of bare literals (e.g. print('Step 1:
    ...'), print(4)) -- status narration or a typed answer, no real work. Lets
    the loop-breaker catch a no-progress loop even when the model varies the
    printed string."""
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return False
    if not tree.body:
        return False
    for node in tree.body:
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "print"):
            return False
        if not all(isinstance(a, ast.Constant) for a in node.value.args):
            return False
    return True


def _parse_json(text):
    """Parse the first JSON object in `text` (small models add stray prose)."""
    text = _strip_template_tags(text or "")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        i = text.find("{")
        if i >= 0:
            try:
                return json.JSONDecoder().raw_decode(text, i)[0]
            except json.JSONDecodeError:
                pass
    return {}


# A/B (CYTOOLS_FINISH_FORGIVE): accept the finish signal where the model
# actually puts it. Observed (qwen3:8b): the engineer completes a step, then
# writes `answer = <result>` / `done = True` as PYTHON VARIABLES instead of
# act-tool fields -- the same protocol-vs-scratchpad conflation as the fixed
# run_python['done']=True illusion -- and the round dies at the step limit.
# When the scratchpad holds an unambiguous finish (done is True, or an
# `answer` variable was assigned), read it as the act fields. The same
# grounded() gate still applies, so this cannot admit fabricated answers.
# DEFAULT ON since the 2026-06-10 A/B (necessary half of the only passing
# configuration); CYTOOLS_FINISH_FORGIVE=0 disables.
from cytools_agent.tools.mapping import env_flag
FINISH_FORGIVE = env_flag("CYTOOLS_FINISH_FORGIVE", default=True)
# Schema-constrained act decoding (see ACT_FORMAT). DEFAULT ON since the
# round-3 A/B (arm H: best single-run arm, fastest, kills the malformed/
# empty/missing-done failure class); CYTOOLS_SCHEMA_ACT=0 disables.
SCHEMA_ACT = env_flag("CYTOOLS_SCHEMA_ACT", default=True)


def _scratchpad_finish():
    """The finish signal read from scratchpad variables: an assigned `answer`
    is the result (assigning it IS the intent), `done=True` alone counts only
    if an answer exists. Returns "" when neither is present."""
    ans = _code._NS.get("answer")
    return "" if ans is None else str(ans)


# the act-protocol interpreter
# ----------------------------
def _scratchpad_run_python(code):
    """run_python with the live scratchpad contents appended, so the engineer
    always sees what it has accumulated."""
    return (_code.run_python(code)
            + f"\n[scratchpad now holds: {_code.namespace_summary()}]")


def _act_args(msg):
    """The act-call arguments from a native message (or recovered from text if
    the model wrote the call as prose). The dict, or None if no act call."""
    for tc in (msg.get("tool_calls") or []):
        args = tc.get("function", {}).get("arguments")
        return _parse_json(args) if isinstance(args, str) else (args or {})
    fb = extract_tool_call(msg.get("content") or "", {"act"})
    return fb["arguments"] if fb and "arguments" in fb else None


def run_engineer(model, evidence, round_no, prompt, max_steps=14,
                 think=False, deadline=None):
    """Run the engineer to completion, streaming observations into `evidence`.
    Each coding step is one observation (ran_code/received_output captured from
    the real run; a non-empty intent is required; a finishing answer is
    rejected unless it appears in a real COMPUTED output). Returns
    (report, n_new_observations, ok) -- ok is False if it hit the step limit
    without finishing."""
    system = ENGINEER_SYSTEM + (ENGINEER_SYSTEM_SCHEMA_NOTE if SCHEMA_ACT
                                else "")
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": prompt}]
    # a stale finish variable from a previous round must not auto-finish this
    # one with the OLD answer
    if FINISH_FORGIVE:
        _code._NS.pop("answer", None)
        _code._NS.pop("done", None)
    pending = {"o": None}
    n0 = len(evidence)
    nags = 0
    code_hist = []           # normalized ran_code this round, to catch loops
    t_llm = [0.0]
    n_llm = [0]
    t_code = [0.0]

    def add(obs):
        obs.update(round=round_no, t=time.time(), kind=obs.get("kind", "obs"))
        evidence.append(obs)
        pending["o"] = obs
        write_evidence(evidence)
        emit("active", who="engineer", phase="working",   # heartbeat per step
             round=round_no, step=len(evidence) - n0)

    def interpret(text):
        if text and pending["o"] and not pending["o"]["interpretation"]:
            pending["o"]["interpretation"] = text
            write_evidence(evidence)

    def tool_reply(out):
        if SCHEMA_ACT:   # no tool call to attach to; results return as user
            messages.append({"role": "user", "content": f"[output]\n{out}"})
        else:
            messages.append({"role": "tool", "tool_name": "act",
                             "content": out})

    def finish(report, ok):
        emit("engineer_timing", round=round_no, llm_calls=n_llm[0],
             llm_s=round(t_llm[0], 1), code_s=round(t_code[0], 1),
             obs=len(evidence) - n0, ok=ok)
        return report, len(evidence) - n0, ok

    # budget: an observation that ERRORED (traceback) is refunded -- recovery
    # from a pointed error message is the designed path, so it must not eat
    # the steps the real work needs (L6 ladder: budget burned on early churn,
    # then the walk died before the deliverable). `hard` caps total LLM calls
    # so an unbroken error loop still terminates.
    consumed, total, hard = 0, 0, max_steps + 8
    while consumed < max_steps and total < hard:
        if deadline is not None and time.monotonic() > deadline:
            # the session's wall-clock budget is spent: end the round
            # honestly instead of letting prompts balloon until the
            # transport times out (measured: 6 of 25 held-out questions
            # died that way)
            return finish("(session time budget exhausted)", False)
        total += 1
        consumed += 1            # refunded below if this step's code errored
        _t = time.monotonic()
        if SCHEMA_ACT:
            # the reply IS the act object, guaranteed schema-valid by the
            # decoding grammar -- no tool indirection, no recovery path
            msg = _ollama_chat(model, messages, think, as_json=ACT_FORMAT,
                               label=f"engineer.r{round_no}")
        else:
            msg = _ollama_chat(model, messages, think, tools=[_ACT_SCHEMA],
                               label=f"engineer.r{round_no}")
        t_llm[0] += time.monotonic() - _t
        n_llm[0] += 1
        messages.append(msg)

        args = (_parse_json(msg.get("content")) if SCHEMA_ACT
                else _act_args(msg))
        if not args:                             # model skipped the tool
            messages.append({"role": "user", "content":
                "You did not call `act`. You MUST call the act tool now with "
                "concrete `code` to gather an observation -- not plain text."})
            continue

        interpret(str(args.get("reflection") or "").strip())
        intent = str(args.get("intent") or "").strip()
        code = str(args.get("code") or "").strip()
        answer = str(args.get("answer") or "").strip()
        done = bool(args.get("done")) or bool(answer)

        if code and not intent and nags < 2:      # require a stated intent
            nags += 1
            tool_reply("Every act call needs a non-empty `intent`: one short "
                       "sentence on what this step does and why. Resend it.")
            continue

        if code:
            _t = time.monotonic()
            out = _scratchpad_run_python(code)
            t_code[0] += time.monotonic() - _t
            if "Traceback (most recent call last)" in out:
                consumed -= 1    # error + pointed feedback: recovery is free
            add({"intent": intent or "(none)", "ran_code": code,
                 "received_output": out, "interpretation": "",
                 "valid_python": valid_python(code)})
            if done and answer and grounded(answer, evidence[n0:]):
                interpret(answer)
                return finish(answer, True)
            # finish-forgiveness: the model put the result in a scratchpad
            # `answer` variable instead of the act field -- unambiguous intent;
            # the grounded() gate still applies
            if FINISH_FORGIVE and not (done and answer):
                sp_ans = _scratchpad_finish()
                if sp_ans and grounded(sp_ans, evidence[n0:]):
                    interpret(sp_ans)
                    return finish(sp_ans, True)
            # loop-breaker: bucket literal/narration prints together so varying
            # the printed string can't dodge the check; stop a spinning round
            norm = "<noop-print>" if _is_noop_print(code) else \
                " ".join(code.split())
            reps = code_hist.count(norm) + 1
            code_hist.append(norm)
            if reps >= 3:        # confirmed loop -- stop burning the step budget
                break
            if reps == 2:        # spinning -> nudge toward different code
                tool_reply("No real progress -- you repeated equivalent code or "
                           "only printed status text. Run DIFFERENT code that "
                           "actually computes the asked-for quantity (loop over "
                           "the fetched objects), or finish if you have it.")
                continue
            if done and answer:
                note = ("\n[your answer is not in this output; compute and "
                        "print EXACTLY the answer before finishing]")
            elif "(no output" not in out:
                # produced a real result but did NOT signal done -- remind it to
                # finish, else a reasonable output just loops to the step limit
                note = ("\n[if that is this step's result, FINISH: call act with "
                        "done=true and answer set to it]")
            else:
                note = ""
            tool_reply(out + note)
            continue

        if done and grounded(answer, evidence[n0:]):    # finish on prior obs
            interpret(answer)
            return finish(answer, True)

        tool_reply("(no grounded answer yet; run code that prints EXACTLY the "
                   "answer, then finish)")

    if len(evidence) == n0:                       # never ran code: record that
        add({"intent": "(engineer ran no code)", "ran_code": "",
             "received_output": "(no code was run within the step limit)",
             "interpretation": "engineer failed to gather evidence",
             "valid_python": True})
    return finish("(engineer: step limit reached)", False)

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
    "  ks_stats(h11, h21=None) -> {count, exists, h21_values}\n"
    "  get_polytope_info(ks_ind) -> {h11, h21, n_rigid_divisors, genera_2face, "
    "facedim_to_nfaces, ...}\n"
    "  get_heights(ks_ind, n=None, kind='NTFE') -> {shape, heights}; the "
    "triangulations are get_heights(ks_ind)['heights'] (a list of height vectors)\n"
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
    call latency is visible. Returns the message dict."""
    payload = {"model": model, "stream": False, "think": think,
               "messages": messages}
    if tools:
        payload["tools"] = tools
    if as_json:
        payload["format"] = "json"
    req = urllib.request.Request(
        OLLAMA_BASE + "/api/chat", json.dumps(payload).encode(),
        {"Content-Type": "application/json"})
    _t = time.monotonic()
    with urllib.request.urlopen(req) as resp:
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


def run_engineer(model, evidence, round_no, prompt, max_steps=14, think=False):
    """Run the engineer to completion, streaming observations into `evidence`.
    Each coding step is one observation (ran_code/received_output captured from
    the real run; a non-empty intent is required; a finishing answer is
    rejected unless it appears in a real COMPUTED output). Returns
    (report, n_new_observations, ok) -- ok is False if it hit the step limit
    without finishing."""
    messages = [{"role": "system", "content": ENGINEER_SYSTEM},
                {"role": "user", "content": prompt}]
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
        messages.append({"role": "tool", "tool_name": "act", "content": out})

    def finish(report, ok):
        emit("engineer_timing", round=round_no, llm_calls=n_llm[0],
             llm_s=round(t_llm[0], 1), code_s=round(t_code[0], 1),
             obs=len(evidence) - n0, ok=ok)
        return report, len(evidence) - n0, ok

    for _step in range(max_steps):
        _t = time.monotonic()
        msg = _ollama_chat(model, messages, think, tools=[_ACT_SCHEMA],
                           label=f"engineer.r{round_no}")
        t_llm[0] += time.monotonic() - _t
        n_llm[0] += 1
        messages.append(msg)

        args = _act_args(msg)
        if args is None:                         # model skipped the tool
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
            add({"intent": intent or "(none)", "ran_code": code,
                 "received_output": out, "interpretation": "",
                 "valid_python": valid_python(code)})
            if done and answer and grounded(answer, evidence[n0:]):
                interpret(answer)
                return finish(answer, True)
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
            note = ("\n[your answer is not in this output; compute and print "
                    "EXACTLY the answer before finishing]"
                    if done and answer else "")
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

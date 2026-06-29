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
# Description:  The project manager: turn the user's request into a plan, then
#               WORK DOWN THE LIST -- dispatch each plan step to the engineer
#               in order (the plan is the routing; this is what prevents the
#               looping/dribbling a free-choice loop caused). A step that does
#               not finish is retried once, then the walk stops rather than
#               plowing into dependent steps. run_session is the conductor.
#               PM_SYSTEM is model-read; the rest is human-read.
# -----------------------------------------------------------------------------

# external imports
import json
import os
import time

# local imports
from cytools_agent.tools import code as _code
from cytools_agent.tools.glossary import glossary_context
from cytools_agent.orchestrator.engineer import (SCHEMA_ACT, TOOL_CHEATSHEET,
                                                 _ollama_chat,
                                                 _parse_json, run_engineer)

# A/B (rides CYTOOLS_SCHEMA_ACT): grammar-constrained plan -- the decoder
# guarantees 2-5 {do, produce} steps, ending the coerce-and-retry dance.
_PLAN_FORMAT = {
    "type": "object",
    "properties": {"todo": {
        "type": "array", "minItems": 2, "maxItems": 5,
        "items": {"type": "object",
                  "properties": {"do": {"type": "string", "minLength": 3},
                                 "produce": {"type": "string"}},
                  "required": ["do", "produce"]}}},
    "required": ["todo"],
}
from cytools_agent.orchestrator.evidence import (backing, emit, grounded,
                                                 last_computed_value,
                                                 render_evidence,
                                                 reset_evidence, reset_session,
                                                 save_log, session_provenance,
                                                 write_evidence)
from cytools_agent.orchestrator.pipeline import PIPELINE, try_pipeline

# model-read
PM_SYSTEM = (
    "You are a project manager directing one engineer who can run Python and "
    "call CYTools tools. You never run code yourself. You read a shared "
    "EVIDENCE log of what the engineer has actually run: trust ran_code and "
    "received_output as ground truth, and treat intent/interpretation as the "
    "engineer's claims. Never state a number that does not appear in some "
    "received_output. Be concise and concrete. Write only in English."
)

_PLOT_WORDS = ("plot", "scatter", "graph", "chart", "histogram")


# A plan step is a STRUCTURED {"do": action, "produce": output} dict. _step_text
# renders one for display/matching/glossary (and tolerates a plain string).
def _step_text(step):
    if isinstance(step, dict):
        do, prod = step.get("do", ""), step.get("produce", "")
        return f"{do} (produce: {prod})" if prod else do
    return str(step)


def _coerce_steps(todo):
    """Normalize the model's todo into a list of {do, produce} dicts -- tolerant
    of a step given as a bare string."""
    out = []
    for s in todo if isinstance(todo, list) else []:
        if isinstance(s, dict) and str(s.get("do", "")).strip():
            out.append({"do": str(s.get("do", "")).strip(),
                        "produce": str(s.get("produce", "")).strip()})
        elif isinstance(s, str) and s.strip():
            out.append({"do": s.strip(), "produce": ""})
    return out


def _plan_covers(direct, todo):
    """The plan must reach the deliverable. We check the plot case explicitly
    -- it is the end-step the model most often drops."""
    if any(w in direct.lower() for w in _PLOT_WORDS):
        return any(any(w in _step_text(s).lower() for w in _PLOT_WORDS)
                   for s in todo)
    return True


def _ensure_deliverable(direct, todo):
    """Safety net: append the deliverable step if the plan dropped it."""
    if any(w in direct.lower() for w in _PLOT_WORDS) \
            and not _plan_covers(direct, todo):
        return list(todo) + [{"do": "make the requested plot of the results",
                              "produce": "a saved plot"}]
    return todo


def _force_two_steps():
    """Last-resort >=2-step plan when the model refuses to decompose: a compute
    step and a deliverable step -- CLEAN and generic, never embedding the raw
    (possibly garbled) restatement -- so the engineer never faces the whole task
    at once and never reads a verbose blob."""
    return [
        {"do": "compute the values the deliverable needs",
         "produce": "the per-item numbers/lists to be summarized"},
        {"do": "produce the requested final result or plot from those values",
         "produce": "the deliverable (the plot, or the final number)"},
    ]


def _produce_met(step, observations):
    """Programmatic produce-check: verify the step actually made its declared
    output. Enforces the concrete, high-value case -- a step whose deliverable
    is a PLOT must really save a figure (the '[saved ... figure(s)]' marker is
    unfakable). Returns (met, reason)."""
    text = _step_text(step).lower()
    if any(w in text for w in _PLOT_WORDS):
        saved = any("[saved " in str(o.get("received_output", ""))
                    for o in observations)
        if not saved:
            return False, ("the deliverable is a plot but no figure was saved "
                           "-- build it with plt (figures auto-save)")
    return True, ""


class ProjectManager:
    """Restate the request, plan it, and write the final answer. Each method is
    a single stateless JSON call (think is per-call: planning reasons, the rest
    does not)."""

    def __init__(self, model, think=False, plan_think=True):
        self.model = model
        self.think = think             # routing CoT; off is ~40x faster
        self.plan_think = plan_think   # decomposition needs reasoning

    def _json(self, instruction, user, think=None, label="PM", schema=None):
        th = self.think if think is None else think
        msg = _ollama_chat(
            self.model,
            [{"role": "system", "content": PM_SYSTEM + " " + instruction},
             {"role": "user", "content": user}], th,
            as_json=schema or True, label=label)
        return _parse_json(msg.get("content"))

    def translate(self, user_message, context=""):
        """Restate the request plainly, UNPACKING jargon via the glossary.
        `context` (prior turns + stored variables) lets a follow-up's
        references ('them', 'the same polytopes') be restated concretely."""
        gloss = glossary_context(user_message) or ""
        if context:
            user_message_in = (f"{context}\n\nNEW REQUEST (restate THIS, "
                               f"resolving its references against the "
                               f"conversation above):\n{user_message}")
        else:
            user_message_in = user_message
        instr = (
            "Restate the user's request in plainer, more direct words an "
            "engineer can act on, in AT MOST 3 sentences. Only RESTATE it: do "
            "NOT solve it, reason step by step, write code, or choose function "
            "arguments. If it uses specialized terms, briefly unpack each in "
            "plain words USING the glossary below -- do not invent meanings or "
            "add concepts the user did not imply, and do not invent numeric "
            "ranges or limits ('each h21' means every h21 that occurs, not a "
            'made-up range). Reply as JSON {"direct_speech": "..."}.')
        out = self._json(instr + ("\n\n" + gloss if gloss else ""),
                         user_message_in, label="PM.translate")
        direct = out.get("direct_speech", "")
        # a runaway restatement (the model reasoning in-content instead of just
        # restating) is long and poisons planning -- fall back to the request
        # (with context the budget is looser: resolved references add words)
        budget = 2 * len(user_message) + (600 if context else 200)
        if not direct or len(direct) > budget:
            return user_message
        return direct

    def plan(self, direct_speech):
        """Decompose into a COMPLETE plan of >=2 steps, EACH with an identified
        output, ending at the deliverable. Rejected/retried otherwise; if the
        model still won't comply it is forced into a 2-step compute+deliverable
        plan (never a single step). Uses plan_think (decomposition reasons)."""
        instruction = (
            "Break the work into AT LEAST 2 concrete steps, covering the task "
            "end to end (never the whole task in one item, never a step that "
            "computes nothing). Each step is an object with TWO fields: \"do\" "
            "(ONE action) and \"produce\" (EXACTLY ONE concrete output, "
            "described with meaning -- e.g. 'a list of integers, one NTFE count "
            "per polytope', 'a single number = the mean', or 'a saved "
            "histogram'). ONE output per step: if the task has SEVERAL "
            "deliverables (e.g. a number AND a plot), make EACH its own step -- "
            "never combine them. (But DO combine the per-item computations that "
            "feed the SAME output into a single step.) Each step's produce feeds "
            "the next; the LAST step produces the final deliverable. You MAY "
            "name a function as a hint (e.g. get_cy_info) but NEVER with "
            "arguments, and do NOT invent function names. Example for 'histogram "
            "the NTFE triangulation counts of h11=3 polytopes and report the "
            "mean': {\"todo\": [{\"do\": \"fetch the h11=3 polytopes\", "
            "\"produce\": \"a list of polytope ids\"}, {\"do\": \"for each "
            "polytope get its NTFE triangulation count\", \"produce\": \"a list "
            "of integer counts, one per polytope\"}, {\"do\": \"compute the mean "
            "of the counts\", \"produce\": \"a single number = the mean\"}, "
            "{\"do\": \"histogram the counts\", \"produce\": \"a saved "
            "histogram\"}]}. Reply as JSON {\"todo\": [{\"do\": \"...\", "
            "\"produce\": \"...\"}, ...]}.")
        todo = []
        for _ in range(3):     # reject incomplete / single-step plans; retry
            raw = self._json(instruction, direct_speech,
                             think=self.plan_think, label="PM.plan",
                             schema=_PLAN_FORMAT if SCHEMA_ACT else None
                             ).get("todo")
            todo = _coerce_steps(raw)
            if len(todo) >= 2 and _plan_covers(direct_speech, todo):
                return todo
        # enforce >=2 even when the model won't decompose: a clean generic split
        if len(todo) < 2:
            todo = _force_two_steps()
        return _ensure_deliverable(direct_speech, todo)

    def addresses(self, step, observations):
        """Per-step coherence gate, judged from the engineer's actual WORK
        (intents + code + output), NOT the surface form of the final answer --
        so a correct result that is just a raw numeric structure (e.g. a list
        of vectors) is not misread as off-step. Lenient by design: it gates
        against DRIFT (doing a
        different step), so it answers false only when the work clearly
        performs another step or a clearly unrelated quantity; when in doubt,
        true. think=False keeps it fast; a malformed reply defaults to True."""
        if not observations:
            return True              # nothing to judge; do not block the walk
        work = "\n".join(
            f"- intent: {o.get('intent', '')}\n  code: {o.get('ran_code', '')}"
            f"\n  output: {str(o.get('received_output', ''))[:160]}"
            for o in observations[-4:])
        out = self._json(
            "Decide whether the engineer's WORK addresses the dispatched STEP. "
            "Judge by what the code and intent actually DID, not the surface "
            "form of the output (a bare list or array of numbers can be a "
            "correct result). Answer false ONLY if the work clearly performs "
            "a DIFFERENT step or computes a clearly "
            'unrelated quantity; when in doubt, true. Reply JSON '
            '{"addresses": true|false}.',
            f"STEP:\n{_step_text(step)}\n\nENGINEER WORK:\n{work}", think=False,
            label="PM.addresses")
        return bool(out.get("addresses", True))

    def verify_produce(self, step, observations):
        """VERIFY #2: did the work actually PRODUCE the step's declared output,
        with the right TYPE and SHAPE? Catches a clear mismatch -- e.g. a single
        number reported when 'a list, one value per polytope' was required (the
        recurring len(info) bug). Lenient on cosmetics; returns (produced,
        issue). A malformed reply defaults to produced=True (never blocks)."""
        produce = step.get("produce", "") if isinstance(step, dict) else ""
        if not observations or not produce:
            return True, ""
        work = "\n".join(
            f"- code: {o.get('ran_code', '')}\n  output: "
            f"{str(o.get('received_output', ''))[:160]}"
            for o in observations[-4:])
        out = self._json(
            "The step had to PRODUCE this output: \"" + produce + "\". From the "
            "engineer's code+outputs, did it actually produce THAT -- the right "
            "TYPE and SHAPE? Answer false ONLY for a clear type/shape mismatch "
            "(e.g. a single number when a list with one value per item was "
            "required, or a missing collection); cosmetic differences are fine "
            "and when in doubt answer true. Reply JSON {\"produced\": "
            "true|false, \"issue\": \"<the mismatch, else empty>\"}.",
            f"REQUIRED OUTPUT: {produce}\n\nENGINEER WORK:\n{work}",
            think=False, label="PM.verify_produce")
        return bool(out.get("produced", True)), str(out.get("issue") or "").strip()

    def summarize(self, direct_speech, evidence, completed=True, missing=""):
        """Compose the final user answer from the evidence. If the run did NOT
        complete, say so honestly rather than inventing a result. `missing`
        names deliverable(s) a previous draft left out (verify_answer's
        finding) so the redraft includes them."""
        note = ("" if completed else
                " NOTE: the run did NOT finish -- a step hit its limit. Report "
                "honestly what was and was not achieved; do NOT present a "
                "result as if the task succeeded.")
        if missing:
            note += (" Your previous draft OMITTED this requested deliverable: "
                     + missing + ". State it explicitly this time, copying any "
                     "numeric value exactly from a received_output.")
        out = self._json(
            "Using ONLY the evidence below, give the user the concrete final "
            "result for their request (actual numbers; if a plot/file was "
            "saved, give its path). Be brief and invent nothing." + note +
            ' Reply as JSON {"message": "..."}.',
            f"Request:\n{direct_speech}\n\n{evidence}", label="PM.summarize")
        return out.get("message") or "(done)"

    def verify_answer(self, question, answer):
        """Cross-check the final answer against the ORIGINAL request: does it
        actually deliver everything asked (e.g. a question wanting a plot AND a
        number got both)? Returns (complete, missing). Lenient: a malformed
        reply defaults to complete, so it never blocks a good answer."""
        out = self._json(
            "Check whether the ANSWER delivers everything the REQUEST asked "
            "for -- every distinct deliverable (e.g. a plot AND a number). "
            "Reply JSON {\"complete\": true|false, \"missing\": \"<the "
            "deliverable(s) not provided, else empty>\"}.",
            f"REQUEST:\n{question}\n\nANSWER:\n{answer}", think=False,
            label="PM.verify")
        return bool(out.get("complete", True)), str(out.get("missing") or "").strip()


class OrchestratorChat:
    """Stateful multi-turn orchestrator. Each .chat() runs a full PM+engineer
    session, but the run_python scratchpad PERSISTS across turns -- the id
    lists and columns one turn stores stay usable -- and each turn's
    translate/compile sees a context block with the recent turns and the
    stored variables, so follow-ups resolve ('plot those vs h21', 'restrict
    to the favorable ones'). Sessions are still archived per turn, so the
    viewer shows each turn separately.

        chat = OrchestratorChat(model="qwen3:8b")
        chat.chat("Fetch the first 20 polytopes at h11=3 and their NTFE counts")
        chat.chat("Now scatter those counts against h21")
    """

    def __init__(self, model="qwen3:8b", **session_kw):
        self.model = model
        self.session_kw = session_kw
        self.turns = []      # (question, answer)

    def _context(self):
        if not self.turns:
            return ""
        lines = ["[Conversation so far -- resolve references like 'those'/"
                 "'them' against it:]"]
        for q, a in self.turns[-3:]:
            lines.append(f"USER ASKED: {q}")
            lines.append(f"ANSWER WAS: {a[:300]}")
        lines.append("[variables stored from these turns: "
                     + _code.namespace_summary() + "]")
        id_lists = [n for n, v in _code._NS.items()
                    if n not in _code._PRELOADED
                    and isinstance(v, (list, tuple)) and v
                    and all(isinstance(i, str) and i.startswith("h11-")
                            for i in v)]
        if id_lists:
            lines.append("[stored polytope-id lists (reusable via "
                         "use_stored): " + ", ".join(id_lists) + "]")
        return "\n".join(lines)

    def chat(self, question):
        """One conversational turn; returns the PM's answer."""
        answer = run_session(question, model=self.model,
                             reset=not self.turns, context=self._context(),
                             **self.session_kw)
        self.turns.append((question, answer))
        return answer

    def reset(self):
        """Forget the conversation AND the stored variables."""
        self.turns = []
        _code.reset_namespace()


def _answer_key(msg, question=""):
    """The comparable RESULT of an answer: its last number after stripping
    digits that are not results -- file paths (fig_1.png, /home/...), domain
    terms (2-face, h11=4, c2), self-consistency annotations of nested runs,
    and SPEC ECHOES (numbers that appear in the question itself, e.g. the
    '50' of 'the first 50 polytopes'). Lessons learned the hard way: raw
    \\d+ extraction on CY prose reads '2-face' as the answer 2, and the
    last number of a sentence is often the restated question, not the
    result."""
    import re as _re

    def nums_of(text):
        text = " ".join(t for t in (text or "").split()
                        if "/" not in t and ".png" not in t)
        text = _re.sub(  # domain digits, mirroring eval.grading._DOMAIN_NOISE
            r"h[\^_]?\{?\d+(?:\s*,\s*\d+)?\}?(?:\s*=\s*-?\d+)?"
            r"|\bc_?\{?\d+\}?"
            r"|\d+-(?:face|faces|fold|folds|dimensional|cycle|cycles|form"
            r"|forms)"
            r"|\b\d+[dD]\b", " ", text, flags=_re.I)
        return [f"{float(n):g}"     # canonical: '384.' == '384' == '384.0'
                for n in _re.findall(r"-?\d+\.?\d*", text)]

    text = (msg or "").split("(self-consistency")[0].split("(LOW CONFIDENCE")[0]
    nums = nums_of(text)
    if not nums:
        return (msg or "").strip()[:40]
    spec = set(nums_of(question))
    results = [n for n in nums if n not in spec]
    return (results or nums)[-1]


def run_session_voted(user_message, votes=3, agree=2, **kw):
    """Numeric self-consistency: run up to `votes` independent sessions and
    stop as soon as `agree` of them land on the same final number (LLM errors
    are diverse, so agreement is strong evidence of correctness; the
    deterministic geometry means matching numbers were computed the same
    way). Returns the agreed answer, or the modal one annotated as
    LOW-CONFIDENCE -- never silently picks a singleton."""
    runs = []   # (key, answer)
    for v in range(votes):
        msg = run_session(user_message, **kw)
        key = _answer_key(msg, question=user_message)
        runs.append((key, msg))
        n = sum(k == key for k, _ in runs)
        if n >= agree:
            return msg + (f"\n\n(self-consistency: {n}/{v + 1} independent "
                          f"runs agreed on {key})")
    counts = {}
    for k, _ in runs:
        counts[k] = counts.get(k, 0) + 1
    best = max(counts, key=counts.get)
    msg = next(m for k, m in runs if k == best)
    return msg + (f"\n\n(LOW CONFIDENCE: {votes} runs disagreed -- "
                  f"results seen: {counts})")


# the session loop -- the conductor
# ---------------------------------
def run_session(user_message, model="qwen3:8b", max_rounds=6, verbose=True,
                pm_think=False, plan_think=True, eng_think=False,
                reset=True, context="", max_seconds=900):
    """Run one PM+engineer session and return the PM's reply. The PM plans,
    then WORKS DOWN THE LIST: each plan step is dispatched to the engineer in
    order (no free re-choosing, which previously dribbled/looped). A step that
    does not finish is retried once, then the walk stops. Evidence and progress
    stream to scratch/ live, and the whole session is archived at the end.

    reset=False keeps the scratchpad from a previous session in this process
    (chat turns build on stored lists); context carries prior-turn text for
    reference resolution. Both are managed by OrchestratorChat. max_seconds
    is the session's wall-clock budget: a walk that cannot converge ends
    honestly instead of growing its prompts until the transport times out."""
    def log(tag, body):
        if verbose:
            print(f"\n{tag}\n{body}", flush=True)

    stamp = int(time.time())
    deadline = (time.monotonic() + max_seconds) if max_seconds else None
    reset_evidence()
    reset_session()
    _code.reset_figures()      # so this run archives only the figures it makes
    if reset:
        _code.reset_namespace()  # clear the scratchpad so vars (polytope_ids)
                                 # don't leak in from a prior session; chat
                                 # turns pass reset=False to BUILD on them
    emit("question", text=user_message)
    # versions + cache paths that determine what a computation returns, so an
    # irreproducible result (e.g. [104] trilayer count) is attributable
    emit("provenance", **session_provenance())
    # record this process's sampled prompt examples, so any session's exact
    # prompts are reconstructible (CYTOOLS_EXAMPLE_SEED reproduces a draw)
    from cytools_agent.tools._examples import EXAMPLE_CHOICES
    emit("examples", choices={k: v[0] for k, v in EXAMPLE_CHOICES.items()})
    pm = ProjectManager(model, think=pm_think, plan_think=plan_think)
    evidence = []   # cumulative across rounds; streamed to the file live

    # the evidence backbone: every curated-tool call this session makes
    # (pipeline stages, engineer code, anything) streams its harness-written
    # ledger row into the same evidence log, as kind="tool_call" rows that
    # no model authors
    from cytools_agent.tools import ledger
    ledger.reset()

    def _sink(row):
        evidence.append(dict(row))
        write_evidence(evidence)
    ledger.set_sink(_sink)

    emit("active", who="PM", phase="translating the request")
    direct = pm.translate(user_message, context=context)
    log("[PM direct speech]", direct)
    emit("direct_speech", text=direct)

    # fast path: the typed pipeline (fetch -> map -> reduce -> plot). The
    # model fills the template's slots once; the harness executes and
    # composes the answer from computed values. Any misfit falls through to
    # the normal plan-and-walk below.
    if PIPELINE:
        emit("active", who="PM", phase="compiling the pipeline")
        ans = try_pipeline(pm, direct, evidence, TOOL_CHEATSHEET, log,
                           context=context, raw=user_message)
        if ans is not None:
            write_evidence(evidence)
            complete, missing = pm.verify_answer(user_message, ans)
            # the LLM verifier often misses that "[saved N figure(s)]" IS the
            # plot deliverable (the marker is harness-written, unfakable) --
            # don't annotate a plot complaint onto an answer that has it
            if (not complete and missing and "[saved " in ans
                    and any(w in missing.lower() for w in _PLOT_WORDS)):
                complete, missing = True, ""
            if not complete and missing:
                ans += f"\n\n(verification -- possibly incomplete: {missing})"
            log("[PM -> user (pipeline)]", ans)
            emit("respond", message=ans)
            emit("active", who="none", phase="done")
            log("[log saved]", save_log(user_message, ans, stamp))
            return ans

    emit("active", who="PM", phase="planning")
    todo = pm.plan(direct)
    log("[PM plan]", json.dumps(todo, indent=2))
    emit("plan", todo=[_step_text(s) for s in todo])

    rnd = [0]
    walk = todo[:max_rounds]
    n_steps = len(walk)

    def dispatch(step, idx):
        rnd[0] += 1
        do = step["do"] if isinstance(step, dict) else str(step)
        produce = step.get("produce", "") if isinstance(step, dict) else ""
        step_str = _step_text(step)
        log(f"[dispatch -- round {rnd[0]}]", step_str)
        emit("dispatch", round=rnd[0], task=step_str,
             plan=[_step_text(s) for s in todo])
        gloss = glossary_context(step_str) or ""
        ev = render_evidence()
        # prompt-size cap: a long session's full evidence render bloats every
        # later prompt (slower generation each round -- the blow-up that
        # caused transport timeouts). Past the threshold, show only the
        # recent tail; the scratchpad summary carries accumulated state.
        if len(ev) > 9000:
            ev = render_evidence(last=6)
        # STANDARDIZED dispatch: a fixed form built from the SELF-CONTAINED
        # structured step (DO / PRODUCE). No overall GOAL -- showing the whole
        # question made the engineer overshoot the step (solve everything, then
        # get flagged off-step) and just added text; the {do, produce} step
        # carries what this step needs.
        prompt = (f"{TOOL_CHEATSHEET}\n\n{ev}\n\n"
                  f"[variables already in the scratchpad: "
                  f"{_code.namespace_summary()}]\n\n"
                  f"STEP {idx}/{n_steps} -- DO: {do}"
                  + (f"\nPRODUCE: {produce}" if produce else "")
                  + (f"\n\n{gloss}" if gloss else ""))
        emit("active", who="engineer", phase="working", round=rnd[0])
        n_before = len(evidence)
        # cap each run_python this step issues to the session budget remaining,
        # so a single long call cannot overrun the deadline and get hard-killed
        # by the outer process. Cleared in finally so a stale (past) deadline
        # never leaks to clamp unrelated run_python calls (e.g. the MCP server).
        _code.set_deadline(deadline)
        try:
            report, n_new, ok = run_engineer(
                model, evidence, rnd[0], prompt, think=eng_think,
                deadline=deadline)
        finally:
            _code.set_deadline(None)
        new_obs = evidence[n_before:]      # this dispatch's observations
        log("[engineer report]", report)
        emit("engineer_report", round=rnd[0], report=report, n_obs=n_new, ok=ok)
        return ok, report, new_obs

    # work down the list, in order. A step that hits its limit OR drifts off
    # the ask (coherence gate) is retried once with a corrective dispatch; if
    # it still fails, STOP -- do not plow ahead to dependent steps.
    completed = True
    for idx, step in enumerate(walk, 1):
        if deadline is not None and time.monotonic() > deadline:
            log("[walk stopped: session time budget exhausted]",
                _step_text(step))
            emit("time_budget_exhausted", step=_step_text(step))
            completed = False
            break
        ok, report, new_obs = dispatch(step, idx)
        why = ""
        if ok:
            met, reason = _produce_met(step, new_obs)    # VERIFY #1: produce-check
            if not met:
                log("[produce unmet]", reason)
                emit("produce_unmet", round=rnd[0], step=_step_text(step),
                     reason=reason)
                ok, why = False, reason
            else:
                produced, issue = pm.verify_produce(step, new_obs)  # VERIFY #2
                if not produced:                          # wrong type/shape
                    log("[produce mismatch]", issue)
                    emit("produce_mismatch", round=rnd[0], step=_step_text(step),
                         issue=issue)
                    ok, why = False, (issue or "the output had the wrong "
                                      "type/shape for what the step required")
                elif not pm.addresses(step, new_obs):    # finished but off-step
                    log("[off-step: engineer work did not address the step]",
                        report)
                    emit("off_step", round=rnd[0], step=_step_text(step),
                         report=report)
                    ok, why = False, "the work drifted off this step"
        if not ok:
            do = step["do"] if isinstance(step, dict) else str(step)
            produce = step.get("produce", "") if isinstance(step, dict) else ""
            corr = {"do": "Do EXACTLY this and report ONLY its result -- " + do
                          + " (the previous attempt did not finish or did not "
                          + "produce the required output"
                          + (f": {why}" if why else "")
                          + "; read the evidence/tracebacks, then finish "
                            "precisely this)",
                    "produce": produce}
            ok, _, retry_obs = dispatch(corr, idx)
            if ok and not _produce_met(step, retry_obs)[0]:   # re-verify produce
                ok = False
            elif ok and not pm.verify_produce(step, retry_obs)[0]:  # re-verify #2
                ok = False
        if not ok:
            log("[walk stopped: step failed twice]", _step_text(step))
            emit("step_failed", step=_step_text(step))
            completed = False
            break

    emit("active", who="PM", phase="composing the final answer")
    msg = pm.summarize(direct, render_evidence(), completed)
    # the model sometimes returns an empty/non-JSON compose reply though the
    # value was already computed and grounded -- surface it deterministically
    # instead of the contentless "(done)" fallback (no load on the weak model).
    if completed and (not msg or msg.strip() == "(done)"):
        recovered = last_computed_value(evidence)
        if recovered is not None:
            log("[summarize empty -- surfaced computed value]", recovered)
            emit("recovered_value", value=recovered)
            msg = recovered
    # VERIFY #3: cross-check the answer against the ORIGINAL request -- flag
    # honestly if a requested deliverable is missing rather than papering over it
    emit("active", who="PM", phase="verifying the answer")
    complete, missing = pm.verify_answer(user_message, msg)
    if not complete and missing:
        # close the loop: verify_answer FOUND the gap, so redraft once with
        # the missing deliverable named (the number is usually sitting in the
        # evidence; the first draft just dropped it) -- then re-verify
        log("[verify: possibly incomplete -- redrafting]", missing)
        emit("verify", complete=False, missing=missing, redraft=True)
        redraft = pm.summarize(direct, render_evidence(), completed,
                               missing=missing)
        complete2, _ = pm.verify_answer(user_message, redraft)
        if complete2:
            msg = redraft
        else:
            msg += f"\n\n(verification -- possibly incomplete: {missing})"
    # backbone backing: classify how the answer's numbers are supported --
    # tool-call rows (harness-recorded, model could not author) vs free-form
    # printed output (authentic but model-shaped)
    label, rows_used = backing(msg, evidence)
    # backing() credits a number to a tool_call row if it appears ANYWHERE in
    # that row -- which over-credits a ubiquitous digit (observed [104]: a
    # fabricated "0" matched a row's '"n_points_interior_to_facets": 0' and was
    # stamped row-backed). An UNFINISHED walk is exactly where the composer
    # invents a number, so there also require the strict grounded() check (the
    # answer's result must appear in a real COMPUTED output); if it fails, drop
    # to unbacked so the redraft/refuse path below runs. Gated on `completed`
    # so a normal, finished answer is never second-guessed.
    if (not completed and label in ("row-backed", "stdout-backed")
            and not grounded(msg, evidence)):
        log("[incomplete run + ungrounded number -- treating as unbacked]", "")
        label, rows_used = "unbacked", []
    # truth-ledger enforcement: a number found in NO captured output must not
    # be surfaced as the result (observed: a hallucinated count emitted with
    # only a "do not trust" note). Redraft once from the evidence using ONLY
    # computed numbers; if it is STILL unbacked the value was never computed,
    # and the honest answer says so rather than stating a fabricated number.
    if label == "unbacked":
        log("[unbacked numbers -- redrafting from evidence]", "")
        emit("verify", complete=False, missing="unbacked numbers", redraft=True)
        redraft = pm.summarize(
            direct, render_evidence(), completed,
            missing="every number in your draft appears in NO computed output. "
                    "Use ONLY numbers that appear in the evidence outputs; if "
                    "the value was never computed, say you could not compute it "
                    "rather than stating a number from memory.")
        l2, r2 = backing(redraft, evidence)
        if l2 != "unbacked":            # recovered a grounded value
            msg, label, rows_used = redraft, l2, r2
        else:                           # still ungrounded -> honest version
            msg = redraft
    emit("backing", label=label, rows=rows_used)
    if label == "row-backed":
        msg += (f"\n\n(evidence: every number is backed by harness-recorded "
                f"tool calls, ledger rows {rows_used})")
    elif label == "stdout-backed":
        msg += ("\n\n(evidence: backed by free-form printed output only -- "
                "weaker than tool-call backing; treat with care)")
    elif label == "unbacked":
        msg += ("\n\n(evidence: contains numbers NOT found in any captured "
                "output -- do not trust them)")
    log("[PM -> user]", msg)
    emit("respond", message=msg)
    emit("active", who="none", phase="done")
    log("[log saved]", save_log(user_message, msg, stamp))
    return msg

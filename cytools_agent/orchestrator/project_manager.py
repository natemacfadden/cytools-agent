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
from cytools_agent.orchestrator.engineer import (TOOL_CHEATSHEET,
                                                 LEAN_CHEATSHEET, _ollama_chat,
                                                 _parse_json, run_engineer)
from cytools_agent.orchestrator.evidence import (emit, render_evidence,
                                                 reset_evidence, reset_session,
                                                 save_log)

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

    def _json(self, instruction, user, think=None, label="PM"):
        th = self.think if think is None else think
        msg = _ollama_chat(
            self.model,
            [{"role": "system", "content": PM_SYSTEM + " " + instruction},
             {"role": "user", "content": user}], th, as_json=True, label=label)
        return _parse_json(msg.get("content"))

    def translate(self, user_message):
        """Restate the request plainly, UNPACKING jargon via the glossary."""
        gloss = glossary_context(user_message) or ""
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
                         user_message, label="PM.translate")
        direct = out.get("direct_speech", "")
        # a runaway restatement (the model reasoning in-content instead of just
        # restating) is long and poisons planning -- fall back to the request
        if not direct or len(direct) > 2 * len(user_message) + 200:
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
                             think=self.plan_think, label="PM.plan").get("todo")
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

    def summarize(self, direct_speech, evidence, completed=True):
        """Compose the final user answer from the evidence. If the run did NOT
        complete, say so honestly rather than inventing a result."""
        note = ("" if completed else
                " NOTE: the run did NOT finish -- a step hit its limit. Report "
                "honestly what was and was not achieved; do NOT present a "
                "result as if the task succeeded.")
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


# the session loop -- the conductor
# ---------------------------------
def run_session(user_message, model="qwen3:4b", max_rounds=6, verbose=True,
                pm_think=False, plan_think=True, eng_think=False):
    """Run one PM+engineer session and return the PM's reply. The PM plans,
    then WORKS DOWN THE LIST: each plan step is dispatched to the engineer in
    order (no free re-choosing, which previously dribbled/looped). A step that
    does not finish is retried once, then the walk stops. Evidence and progress
    stream to scratch/ live, and the whole session is archived at the end."""
    def log(tag, body):
        if verbose:
            print(f"\n{tag}\n{body}", flush=True)

    stamp = int(time.time())
    reset_evidence()
    reset_session()
    _code.reset_figures()      # so this run archives only the figures it makes
    _code.reset_namespace()    # clear the scratchpad so vars (e.g. polytope_ids)
                               # don't leak in from a prior session in this proc
    emit("question", text=user_message)
    pm = ProjectManager(model, think=pm_think, plan_think=plan_think)
    evidence = []   # cumulative across rounds; streamed to the file live

    emit("active", who="PM", phase="translating the request")
    direct = pm.translate(user_message)
    log("[PM direct speech]", direct)
    emit("direct_speech", text=direct)
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
        # A/B (CYTOOLS_LEAN_PROMPT): lean trims the per-turn scaffolding so the
        # signal (step + recipe) isn't buried -- condensed cheatsheet, recipe-
        # only glossary, and only the last few observations.
        lean = bool(os.environ.get("CYTOOLS_LEAN_PROMPT"))
        cheatsheet = LEAN_CHEATSHEET if lean else TOOL_CHEATSHEET
        gloss = glossary_context(step_str, recipe_only=lean) or ""
        ev = render_evidence(last=3) if lean else render_evidence()
        # STANDARDIZED dispatch: a fixed form built from the SELF-CONTAINED
        # structured step (DO / PRODUCE). No overall GOAL -- showing the whole
        # question made the engineer overshoot the step (solve everything, then
        # get flagged off-step) and just added text; the {do, produce} step
        # carries what this step needs.
        prompt = (f"{cheatsheet}\n\n{ev}\n\n"
                  f"[variables already in the scratchpad: "
                  f"{_code.namespace_summary()}]\n\n"
                  f"STEP {idx}/{n_steps} -- DO: {do}"
                  + (f"\nPRODUCE: {produce}" if produce else "")
                  + (f"\n\n{gloss}" if gloss else ""))
        emit("active", who="engineer", phase="working", round=rnd[0])
        n_before = len(evidence)
        report, n_new, ok = run_engineer(
            model, evidence, rnd[0], prompt, think=eng_think)
        new_obs = evidence[n_before:]      # this dispatch's observations
        log("[engineer report]", report)
        emit("engineer_report", round=rnd[0], report=report, n_obs=n_new, ok=ok)
        return ok, report, new_obs

    # work down the list, in order. A step that hits its limit OR drifts off
    # the ask (coherence gate) is retried once with a corrective dispatch; if
    # it still fails, STOP -- do not plow ahead to dependent steps.
    completed = True
    for idx, step in enumerate(walk, 1):
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
    # VERIFY #3: cross-check the answer against the ORIGINAL request -- flag
    # honestly if a requested deliverable is missing rather than papering over it
    emit("active", who="PM", phase="verifying the answer")
    complete, missing = pm.verify_answer(user_message, msg)
    if not complete and missing:
        log("[verify: possibly incomplete]", missing)
        emit("verify", complete=False, missing=missing)
        msg += f"\n\n(verification -- possibly incomplete: {missing})"
    log("[PM -> user]", msg)
    emit("respond", message=msg)
    emit("active", who="none", phase="done")
    log("[log saved]", save_log(user_message, msg, stamp))
    return msg

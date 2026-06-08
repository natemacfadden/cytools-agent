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
import time

# local imports
from cytools_agent.tools import code as _code
from cytools_agent.tools.glossary import glossary_context
from cytools_agent.orchestrator.engineer import (TOOL_CHEATSHEET, _ollama_chat,
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


def _plan_covers(direct, todo):
    """The plan must reach the deliverable. We check the plot case explicitly
    -- it is the end-step the model most often drops."""
    if any(w in direct.lower() for w in _PLOT_WORDS):
        return any(any(w in s.lower() for w in _PLOT_WORDS) for s in todo)
    return True


def _ensure_deliverable(direct, todo):
    """Safety net: append the deliverable step if the plan dropped it."""
    if any(w in direct.lower() for w in _PLOT_WORDS) \
            and not _plan_covers(direct, todo):
        return list(todo) + ["make the requested plot of the results"]
    return todo


def _force_two_steps(direct):
    """Last-resort >=2-step plan when the model refuses to decompose: a compute
    step (build the values) and a deliverable step -- so the engineer never
    faces the whole task at once."""
    return [
        f"compute the values the task needs -- output: the numbers/lists to be "
        f"summarized, for: {direct}",
        f"produce the requested final result/plot from those values -- output: "
        f"the deliverable, for: {direct}",
    ]


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
            "Break the work into AT LEAST 2 concrete steps for the engineer, "
            "covering the task end to end (never put the whole task in one "
            "item, and never include a step that computes nothing). For EACH "
            "step state BOTH the action AND its explicit OUTPUT -- the concrete "
            "data it produces, described with meaning: e.g. 'output: a list of "
            "numbers, each the largest curve volume of one polytope', or "
            "'output: a single integer = the count', or 'output: a saved "
            "scatter plot of X vs Y'. Each step's output feeds the next; the "
            "LAST step's output IS the requested deliverable (the plot, or the "
            "final number). You MAY name a function as a hint (e.g. "
            "get_cy_info) but NEVER with arguments. Example for 'scatter the "
            "number of prime toric divisors vs the Euler characteristic for "
            "h11=3 polytopes': todo = [\"fetch the h11=3 polytopes -- output: "
            "a list of polytope ids\", \"for each polytope compute its prime-"
            "toric-divisor count and Euler characteristic -- output: two lists "
            "of numbers, one value per polytope\", \"scatter the two lists -- "
            "output: a saved scatter plot\"]. Reply as JSON {\"todo\": "
            "[\"step -- output: ...\", ...]}.")
        todo = []
        for _ in range(3):     # reject incomplete / single-step plans; retry
            todo = self._json(instruction, direct_speech,
                              think=self.plan_think,
                              label="PM.plan").get("todo", [])
            if isinstance(todo, list) and len(todo) >= 2 \
                    and _plan_covers(direct_speech, todo):
                return todo
        # enforce >=2 even when the model won't decompose: split into a
        # compute step and a deliverable step rather than dumping the whole task
        if not (isinstance(todo, list) and len(todo) >= 2):
            todo = _force_two_steps(direct_speech)
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
            f"STEP:\n{step}\n\nENGINEER WORK:\n{work}", think=False,
            label="PM.addresses")
        return bool(out.get("addresses", True))

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
    _code.reset_figures()   # so this run archives only the figures it makes
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
    emit("plan", todo=todo)

    rnd = [0]

    def dispatch(step):
        rnd[0] += 1
        log(f"[dispatch -- round {rnd[0]}]", step)
        emit("dispatch", round=rnd[0], task=step, plan=todo)
        gloss = glossary_context(step) or ""
        prompt = (f"{TOOL_CHEATSHEET}\n\n{render_evidence()}\n\n"
                  f"[variables already in the scratchpad: "
                  f"{_code.namespace_summary()}]\n\nYour task:\n{step}"
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
    for step in todo[:max_rounds]:
        ok, report, new_obs = dispatch(step)
        if ok and not pm.addresses(step, new_obs):    # finished but off-step
            log("[off-step: engineer work did not address the step]", report)
            emit("off_step", round=rnd[0], step=step, report=report)
            ok = False                                # retry like a failure
        if not ok:
            ok, _, _ = dispatch(
                "Do EXACTLY this step and report ONLY its result: " + step +
                " (the previous attempt drifted off the task or hit its step "
                "limit -- read the evidence and tracebacks, then finish "
                "precisely this step).")
        if not ok:
            log("[walk stopped: step failed twice]", step)
            emit("step_failed", step=step)
            completed = False
            break

    emit("active", who="PM", phase="composing the final answer")
    msg = pm.summarize(direct, render_evidence(), completed)
    log("[PM -> user]", msg)
    emit("respond", message=msg)
    emit("active", who="none", phase="done")
    log("[log saved]", save_log(user_message, msg, stamp))
    return msg

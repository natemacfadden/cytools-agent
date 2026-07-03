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
# Description:  The single typed-emission path shared by every eval arm. A
#               rung/arm is asked (via FINAL_INSTRUCTION) to end its reply with
#               a machine-readable <final>{"kind","value"} block; this module
#               guarantees the block is present so the deterministic typed
#               grader (eval/answer.py grade_typed) can read it -- no prose
#               parsing, no regex. When a reply lacks the block, a separate
#               model call (blind to the truth) extracts the committed answer
#               into one; that call is the only model left in the grading path,
#               and its verdict is still pure code.
#
#               Local-Ollama arms (eval.py, eval_single_pm.py, the ladder) use
#               the backstop. An external-model arm (eval_claude.py) cannot --
#               the finalizer client is Ollama's -- so it disables the backstop
#               and relies on the arm's own reliable self-emission.
#
# All functions here are human-read (developer tooling).
# -----------------------------------------------------------------------------

from eval.answer import FINAL_INSTRUCTION, parse_final
from eval.grading import TIMED_OUT

__all__ = ["FINAL_INSTRUCTION", "ensure_final", "finalizing"]


def _finalize_blind(text, question, model, timeout=120):
    """Backstop when a reply did not emit a <final> block: extract its committed
    answer as a typed value with a separate model call that is blind to the
    truth (so it cannot bias toward a match) and constrained to emit only the
    block. Returns the block string, or "" if it can't parse one out."""
    from eval._harness import client
    sysmsg = (
        "Extract the assistant's COMMITTED answer to the question as a typed "
        "value. Output ONLY one line and nothing else:\n"
        '<final>{"kind": "<int|float|list|bool|impossible|none>", '
        '"value": <v>}</final>\n'
        "value is a bare number, a JSON array, true/false, or null.\n"
        'Use kind "impossible" (value null) ONLY when the assistant '
        "affirmatively concludes no valid answer exists -- the object is not "
        "found, the target is unreachable / infeasible, or a solver determined "
        "there is no solution.\n"
        'Use kind "none" (value null) when the assistant hit an ERROR, was cut '
        "off, or otherwise could not complete the computation. A failure to "
        "compute (an exception, a missing field, an interrupted run) is NOT a "
        "determination of impossibility -- use \"none\", not \"impossible\".")
    user = (f"QUESTION:\n{question}\n\nASSISTANT ANSWER:\n{text}\n\n"
            "Extract the committed answer. /no_think")
    try:
        resp = client.chat.completions.create(
            model=model, timeout=timeout, temperature=0,
            messages=[{"role": "system", "content": sysmsg},
                      {"role": "user", "content": user}])
        raw = (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""
    return raw if parse_final(raw) is not None else ""


def ensure_final(text, question, model, backstop=True):
    """Guarantee the answer carries a <final> block. Keep an existing one (the
    model already emitted it); never finalize a
    TIMEOUT/ERROR sentinel (grading quarantines those); else, when backstop is
    on, append the blind finalizer's extraction. backstop=False (external-model
    arms whose finalizer client cannot reach the model) leaves it untouched and
    trusts the arm's self-emission."""
    if not isinstance(text, str):
        return text
    if text == TIMED_OUT or text.startswith("(error:") or parse_final(text):
        return text
    if not backstop:
        return text
    block = _finalize_blind(text, question, model)
    return (text + "\n" + block) if block else text


def finalizing(run_fn, model, backstop=True):
    """Wrap a run_fn(question) -> answer so it (a) appends FINAL_INSTRUCTION to
    the prompt it sends and (b) guarantees a <final> block on the reply. Lets an
    eval arm keep its plain run_fn and opt into typed emission at the call
    site."""
    def wrapped(question):
        ans = run_fn(question + FINAL_INSTRUCTION)
        return ensure_final(ans, question, model, backstop=backstop)
    return wrapped

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
# Description:  Experiment "A": answers are pointers to run_python variables, not
#               typed values. The grader reads the real object from the namespace
#               (_code._NS) and checks it, so the report can't disagree with the
#               code. Doesn't verify the code (problem "B", punted). A/B vs the
#               value path: eval_single_pm --nodes.
#
# All functions here are human-read (developer tooling).
# -----------------------------------------------------------------------------

import json
import re

from cytools_agent.tools import code as _code
from eval.answer import FINAL_INSTRUCTION, _decimals, check, grade_typed, parse_final
from eval.emit import ensure_final
from eval.grading import TIMED_OUT

# Same block shape as answer._FINAL_RE, but the payload may carry "node" instead
# of "value", so parse it here rather than reuse the value-only parser.
_FINAL_RE = re.compile(r"<final>\s*(\{.*?\})\s*</final>", re.S | re.I)

# The node-pointer contract, appended at the answering layer (parallels
# answer.FINAL_INSTRUCTION). The model must leave the answer in a live variable.
NODE_FINAL_INSTRUCTION = (
    "\n\nDo your computation with the run_python tool. Leave your committed "
    "answer bound to a variable in the run_python session (any name, e.g. "
    "`answer = ...`); do NOT retype the numeric result yourself. When you are "
    "done, end your reply with, on its own line, exactly:\n"
    '<final>{"kind": "<int|float|list|bool|impossible|none>", "node": '
    '"<the_variable_name>"}</final>\n'
    "The grader reads the value straight from that variable, so it must still "
    "exist in the session and hold exactly the committed answer (a number, a "
    "JSON-like list, or a bool). If the task has no valid answer, use "
    '{"kind": "impossible", "value": null}; if you could not determine it, use '
    '{"kind": "none", "value": null}.'
)


def _parse_final_any(text):
    """Last valid <final> block as a dict, accepting either a "node" pointer or
    a literal "value" (or a bare kind for impossible/none). None if absent."""
    if not isinstance(text, str):
        return None
    for m in reversed(list(_FINAL_RE.finditer(text))):
        try:
            obj = json.loads(m.group(1))
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict) and ("value" in obj or "node" in obj):
            obj.setdefault("kind", "none")
            return obj
    return None


def _coerce(v):
    """Make a namespace value comparable by check(): numpy scalars/arrays down to
    plain Python, tuples to lists, recursively. Leaves plain types untouched."""
    try:
        import numpy as np
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
    except ImportError:
        pass
    if isinstance(v, (list, tuple)):
        return [_coerce(x) for x in v]
    return v


def resolve_and_grade(ans, truth, ns=None):
    """Grade a node-pointer reply. Quarantines TIMEOUT/ERROR like grade_typed;
    resolves a "node" pointer against the run_python namespace (a dangling
    pointer fails); grades a literal-value or impossible/none block as-is; a
    reply with no block falls back to the shared blind finalizer.
    Returns (status, final_dict_used)."""
    if ns is None:
        ns = _code._NS
    if ans == TIMED_OUT:
        return "TIMEOUT", None
    if isinstance(ans, str) and ans.startswith("(error:"):
        return "ERROR", None

    final = _parse_final_any(str(ans))
    if final is None:
        return None, None      # caller applies the no-block backstop

    node = final.get("node")
    if node is not None and final.get("value") is None:
        if node not in ns:
            return "FAIL", final      # pointed at a variable that isn't there
        # keep `node` in the record so the results show a pointer was used
        # (vs. the model ignoring the contract and typing a literal value)
        final = {"kind": final.get("kind", "none"),
                 "value": _coerce(ns[node]), "node": node}
    return ("PASS" if check(final, truth) else "FAIL"), final


def run_node_arm(run, model, question, timeout, truth, raw=False):
    """One node-pointer run + grade, reusing the harness `run`. When the model
    emits no <final> at all, backstop with the same blind finalizer the value
    arm uses, so the A/B isolates the node mechanism, not the backstop."""
    ans = run(model, question + NODE_FINAL_INSTRUCTION, timeout, raw=raw)
    status, final = resolve_and_grade(ans, truth)
    if status is None:                       # no committed block -> backstop
        from eval.answer import grade_typed
        ans = ensure_final(ans, question, model)
        return grade_typed(ans, truth), ans, parse_final(str(ans))
    return status, ans, final


# =============================================================================
# Experiment (b): harness-captured result nodes.
#
# Variant "A" asked the model to name a variable and point at it, which dangled
# (claimed vars it never bound; -18 pts on qwen3:14b). Here the harness numbers
# each run_python last-expression value (code.CAPTURE_RESULT_NODES) and echoes
# "[node N] <value>"; the model points at a number it was shown, so the pointer
# can't dangle.
# =============================================================================

RESULT_NODE_FINAL_INSTRUCTION = (
    "\n\nDo your computation with the run_python tool, and make the LAST line of "
    "your code the bare expression that equals the answer (do not wrap it in "
    "print). run_python echoes it back as `[node N] <value>` -- N is that value's "
    "node number. Do NOT retype the number yourself. When you are done, end your "
    "reply with, on its own line, exactly:\n"
    '<final>{"kind": "<int|float|list|bool|impossible|none>", "node": N}</final>\n'
    "where N is the node number whose shown value equals your committed answer. "
    "The grader reads that value directly. If the task has no valid answer use "
    '{"kind": "impossible", "value": null}; if you could not determine it use '
    '{"kind": "none", "value": null}.'
)


def resolve_result_and_grade(ans, truth):
    """Grade a result-node reply. Here "node" is an integer id into the
    harness-captured store (code._NODES). Out-of-range id -> FAIL (pointed at a
    node never produced). Literal value / impossible / none grade as-is; no block
    -> None (caller backstops)."""
    if ans == TIMED_OUT:
        return "TIMEOUT", None
    if isinstance(ans, str) and ans.startswith("(error:"):
        return "ERROR", None

    final = _parse_final_any(str(ans))
    if final is None:
        return None, None

    node = final.get("node")
    if node is not None and final.get("value") is None:
        try:
            resolved = _code.get_result_node(node)
        except (IndexError, ValueError, TypeError):
            return "FAIL", final          # pointed at a node that isn't there
        final = {"kind": final.get("kind", "none"),
                 "value": _coerce(resolved), "node": node}
    return ("PASS" if check(final, truth) else "FAIL"), final


def run_result_node_arm(run, model, question, timeout, truth, raw=False):
    """One harness-node run + grade. Turns node capture + id echo on for the
    run, then resolves the selected id against code._NODES."""
    _code.CAPTURE_RESULT_NODES = True
    _code.ECHO_NODE_IDS = True
    try:
        ans = run(model, question + RESULT_NODE_FINAL_INSTRUCTION, timeout, raw=raw)
        status, final = resolve_result_and_grade(ans, truth)
    finally:
        _code.CAPTURE_RESULT_NODES = False
        _code.ECHO_NODE_IDS = False
    if status is None:                        # no committed block -> backstop
        ans = ensure_final(ans, question, model)
        return grade_typed(ans, truth), ans, parse_final(str(ans))
    return status, ans, final


# =============================================================================
# Experiment (c): the watchful eye. Verify, don't relocate.
#
# The model answers in its natural value mode (types the number), so no addressing
# burden and no regression. The harness silently captures what the model computed
# (last bare expressions + anything printed) and flags whether the committed value
# is grounded, i.e. equals one of the captured values. Softer than a hard pointer,
# but free.
# =============================================================================

def _flatten_numbers(nodes, out):
    """All scalar numbers appearing anywhere in the captured nodes (recursing
    into lists/tuples/arrays), so a value derived as an element of a printed
    structure can still ground. Bools kept separate by the caller."""
    for n in nodes:
        c = _coerce(n)
        if isinstance(c, (list, tuple)):
            _flatten_numbers(c, out)
        else:
            out.append(c)


def _near(a, b):
    """a matches b up to the reporting precision, so a computed 2223.199 grounds
    a reported 2223.2, and an int matches its float form."""
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    d = _decimals(b)
    return round(a, d) == round(b, d) or abs(a - b) <= 1e-6


def is_grounded(final, nodes):
    """Fuzzy grounding: does the committed value trace to something computed?
    (1) typed-equal to a whole captured node (reuses check), or (2) numeric and
    near some captured number (rounding / element-of-structure), or (3) a list
    whose every entry is near some captured number. Bools are reported but weak
    (only two values -> coincidental)."""
    if not final:
        return False
    val = final.get("value")
    if val is None:
        return False
    cand = [_coerce(n) for n in nodes]
    if any(check(final, c) for c in cand):          # (1) exact-ish, any type
        return True
    flat = []
    _flatten_numbers(nodes, flat)
    nums = [x for x in flat if not isinstance(x, bool)]
    if isinstance(val, bool):
        return any(isinstance(x, bool) and x == val for x in flat)  # weak
    if isinstance(val, (int, float)):
        return any(_near(x, val) for x in nums)     # (2) rounding / containment
    if isinstance(val, list):                       # (3) every entry grounded
        entries = []
        _flatten_numbers([val], entries)
        entries = [e for e in entries if not isinstance(e, bool)]
        return bool(entries) and all(any(_near(x, e) for x in nums) for e in entries)
    return False


def run_watch_arm(run, model, question, timeout, truth, raw=False):
    """(c) run: value-mode answer + silent grounding check. Grades pass/fail
    exactly like the value arm; returns a 4th field: a dict {ok, n_nodes, nodes}
    where ok = is the committed value grounded (fuzzy). `nodes` reprs are logged
    for diagnosis."""
    _code.CAPTURE_RESULT_NODES = True
    _code.ECHO_NODE_IDS = False        # capture silently: output == value arm
    try:
        ans = run(model, question + FINAL_INSTRUCTION, timeout, raw=raw)
        ans = ensure_final(ans, question, model)
        final = parse_final(str(ans))
        status = grade_typed(ans, truth)
        grounded = {"ok": is_grounded(final, _code._NODES),
                    "n_nodes": len(_code._NODES),
                    "nodes": [repr(n)[:80] for n in _code._NODES]}
    finally:
        _code.CAPTURE_RESULT_NODES = False
    return status, ans, final, grounded

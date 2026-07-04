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
# Description:  Experiment "A": answers are POINTERS TO NODES, not typed values.
#               Instead of retyping its result into the <final> block (where a
#               model can transcribe or confabulate a wrong number), the model
#               leaves its answer in a run_python variable and points at it:
#
#                   <final>{"kind": "int", "node": "answer"}</final>
#
#               The grader then reads the REAL Python object out of the
#               persistent run_python namespace (_code._NS) and grades that with
#               the same deterministic check() as the value path. The reported
#               value can no longer disagree with what the code computed -- it IS
#               what the code computed. Nothing verifies the code itself (that is
#               problem "B", punted); this only closes the report-the-number gap.
#
#               A/B against the value path: run eval.eval_single_pm with --nodes.
#
# All functions here are human-read (developer tooling).
# -----------------------------------------------------------------------------

import json
import re

from cytools_agent.tools import code as _code
from eval.answer import check, parse_final
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
    pointer FAILs); a literal-value or impossible/none block is graded as-is;
    a reply with no block at all falls back to the shared blind finalizer so the
    only difference from the value arm is how a *committed* answer is read.
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

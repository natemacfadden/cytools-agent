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
# Description:  Typed committed-answer schema + a DETERMINISTIC checker for the
#               system ladder. Every rung ends its output with a machine-
#               readable block:
#
#                   <final>{"kind": "<int|float|list|bool|impossible|none>",
#                           "value": <json value>}</final>
#
#               Grading then compares that typed value to the corpus truth BY
#               TYPE -- exact for int/bool, tolerance for float, ordered-or-
#               entry-set for list, a marker for impossible. There is no prose
#               parsing, no substring matching, and no model in the decision:
#               the answer is extracted at the rung's own layer, the check is
#               pure code. This is what makes grading reproducible + auditable
#               and immune to the digit-leak false positives of prose matching.
#
# All functions here are human-read (developer tooling).
# -----------------------------------------------------------------------------

import json
import re

KINDS = ("int", "float", "list", "bool", "impossible", "none")

# The block a rung appends as the LAST thing in its reply. Non-greedy up to the
# closing tag; DOTALL so a value may span lines; last block wins (see below).
_FINAL_RE = re.compile(r"<final>\s*(\{.*?\})\s*</final>", re.S | re.I)

# Instruction each model-driven rung appends at its own final layer, rather than
# baking it into the shared question, so the contract is applied uniformly at the
# point of answering.
FINAL_INSTRUCTION = (
    "\n\nWhen you are done, end your reply with your committed answer on its "
    "own line, in exactly this form:\n"
    '<final>{"kind": "<int|float|list|bool|impossible|none>", "value": <v>}'
    "</final>\n"
    "Rules: put ONLY the committed answer in <value> -- a bare number, a JSON "
    "array (e.g. [1,2,3] or [[1,2],[3,4]]), true/false, or null. No units, no "
    "prose, no commas as thousands separators. Use kind \"impossible\" (value "
    "null) if the task has no valid answer and you are reporting that; use "
    "kind \"none\" (value null) only if you genuinely could not determine it."
)


def build_final(kind, value):
    """Serialize a typed answer block from a real Python value, so a caller that
    knows its answer structurally can author the block directly instead of
    re-parsing it out of prose."""
    if kind not in KINDS:
        raise ValueError(f"bad kind {kind!r}")
    return "<final>" + json.dumps({"kind": kind, "value": value}) + "</final>"


def parse_final(text):
    """The parsed {kind, value} dict from the LAST valid <final> block, else
    None. Last-wins so a rung that shows an example block earlier and its real
    answer last is read correctly."""
    if not isinstance(text, str):
        return None
    for m in reversed(list(_FINAL_RE.finditer(text))):
        try:
            obj = json.loads(m.group(1))
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict) and "value" in obj:
            obj.setdefault("kind", "none")
            return obj
    return None


def _round(x):
    """Canonical numeric form for comparison: 384.0 == 384, 3.0000001 ~ 3."""
    f = round(float(x), 3)
    return int(f) if f == int(f) else f


def _decimals(x):
    """Decimal places in a truth literal, so a float is checked to the SAME
    precision it was rounded to: truth 1.46 (2 dp) accepts a computed 1.4583
    (rounds to 1.46) but rejects 1.5."""
    s = repr(float(x))
    return len(s.split(".", 1)[1]) if "." in s else 0


def _entries(v):
    """Normalize a list/tuple answer to a sorted list of entries, each entry a
    tuple of rounded numbers (or a rounded scalar). Order-insensitive at the
    top level (entry order is arbitrary), but an entry's internal order is
    preserved -- matching the corpus's [i,j,k,value] / ray semantics."""
    if not isinstance(v, (list, tuple)):
        v = [v]
    out = []
    for e in v:
        if isinstance(e, (list, tuple)):
            out.append(tuple(_round(x) for x in e))
        else:
            out.append(_round(e))
    # sort by string form so mixed shapes are still orderable
    return sorted(out, key=lambda e: str(e))


def _list_match(val, truth):
    try:
        return _entries(val) == _entries(truth)
    except (ValueError, TypeError):
        return False


def grade_typed(ans, truth):
    """The grader for every eval arm. Quarantines ERROR (harness failure) and
    TIMEOUT sentinels, then decides PASS/FAIL by a deterministic typed check of
    the <final> block -- no prose matching, no regex. A missing/unparseable
    block -> FAIL."""
    from eval.grading import TIMED_OUT
    if ans == TIMED_OUT:
        return "TIMEOUT"
    if isinstance(ans, str) and ans.startswith("(error:"):
        return "ERROR"
    return "PASS" if check(parse_final(ans), truth) else "FAIL"


def check(final, truth):
    """PASS (True) / FAIL (False) of a parsed <final> dict vs the typed truth.
    `final` is the dict from parse_final (or None)."""
    if final is None:
        return False
    kind = str(final.get("kind", "")).lower()
    val = final.get("value")

    # negative test: truth == "IMPOSSIBLE" -- pass iff the rung committed the
    # impossible marker (it attempted and reported no valid answer).
    if isinstance(truth, str) and truth.strip().upper() == "IMPOSSIBLE":
        return kind == "impossible"

    # a non-answer never matches a real truth
    if kind == "none" or (val is None and kind != "impossible"):
        return False

    # a single-value map column is a scalar answer wrapped in a 1-element list
    # (e.g. is_favorable_M for one polytope -> [False]); unwrap it so it can
    # match a scalar truth rather than being force-failed as a list-vs-scalar.
    if (isinstance(val, (list, tuple)) and len(val) == 1
            and not isinstance(truth, (list, tuple))):
        val = val[0]

    if isinstance(truth, bool):                     # bool BEFORE int (bool<:int)
        return isinstance(val, bool) and val == truth
    if isinstance(truth, int):
        try:
            return not isinstance(val, bool) and _round(val) == truth
        except (TypeError, ValueError):
            return False
    if isinstance(truth, float):
        try:
            v = float(val)
        except (TypeError, ValueError):
            return False
        prec = _decimals(truth)
        return round(v, prec) == round(truth, prec) or abs(v - truth) <= 1e-6
    if isinstance(truth, (list, tuple)):
        return _list_match(val, truth)
    return str(val).strip().lower() == str(truth).strip().lower()

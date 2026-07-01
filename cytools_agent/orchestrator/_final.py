# =============================================================================
#    Copyright (C) 2026  Nate MacFadden for the Liam McAllister Group
#    (GPL-3.0-or-later; see coordinator.py header.)
# =============================================================================
#
# Session-scoped capture of the orchestrator's TYPED committed answer. The
# pipeline sets it at the point of computation (a real Python int/float/list/
# bool -- NOT a re-parse of prose); run_session reads it and appends a
# <final>{...}</final> block to its reply. The grader then compares that typed
# value to the corpus truth deterministically (see eval/answer.py, which parses
# the SAME block format). Reset per session so a value can't leak between the
# independent runs of a self-consistency vote.

import json

_STATE = {"set": False, "kind": None, "value": None}


def reset_final():
    _STATE.update(set=False, kind=None, value=None)


def set_final(kind, value):
    """Record the committed answer. Last write wins (a later reduce overrides an
    earlier map column). kind is one of int/float/list/bool/impossible/none."""
    _STATE.update(set=True, kind=kind, value=value)


def get_final():
    """(kind, value) if a typed answer was captured this session, else None."""
    return (_STATE["kind"], _STATE["value"]) if _STATE["set"] else None


def final_block(kind, value):
    """The canonical serialized block appended to a reply. Must match the
    parser in eval/answer.py (parse_final)."""
    return "<final>" + json.dumps({"kind": kind, "value": value}) + "</final>"


def kind_of(val):
    """Map a computed Python value to a schema kind, or None if it isn't a
    cleanly typed answer (e.g. an id string from argmax -- left to the
    downstream finalizer)."""
    if isinstance(val, bool):
        return "bool"
    if isinstance(val, int):
        return "int"
    if isinstance(val, float):
        return "float"
    if isinstance(val, (list, tuple)):
        return "list"
    return None

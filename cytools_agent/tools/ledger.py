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
# Description:  The evidence backbone: a ledger of tool invocations written
#               exclusively by the harness at the tool boundary. Models cause
#               rows by calling tools and may read rows; they can never author
#               or edit one. Each row records the exact call (tool, args) and
#               its structured result -- so any claim can be audited against
#               "what CYTools actually returned" without trusting any model's
#               prose or print statements.
#
#               Reentrancy guard: only top-level tool calls are recorded.
#               compute_for_each evaluating get_polytope_info 100x internally
#               is one row (the compute_for_each call), not 101.
#
#               All human-read.
# -----------------------------------------------------------------------------

# external imports
import functools
import json
import time

_ROWS = []          # this process's rows, in order
_SINK = None        # optional callback(row) -- the orchestrator wires one
_DEPTH = 0          # reentrancy: record only depth-0 calls

_ARG_CAP = 300      # per-argument serialized length
_RES_CAP = 1500     # serialized result length


def _safe(val, cap):
    """Compact, JSON-safe rendering of a call argument or result."""
    try:
        s = json.dumps(val, default=repr)
    except (TypeError, ValueError):
        s = repr(val)
    if len(s) > cap:
        s = s[:cap] + f"...(+{len(s) - cap} chars)"
    return s


def record(tool, args_repr, result_repr):
    row = {"row": len(_ROWS), "kind": "tool_call", "tool": tool,
           "args": args_repr, "result": result_repr, "t": time.time()}
    _ROWS.append(row)
    if _SINK:
        _SINK(row)
    return row["row"]


def wrap(fn):
    """Ledger a tool: every top-level call records (args, result) or
    (args, the exception). The wrapped function is otherwise transparent."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _DEPTH
        if _DEPTH > 0:
            return fn(*args, **kwargs)
        _DEPTH += 1
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            record(fn.__name__, _fmt_args(args, kwargs),
                   f"raised {type(e).__name__}: {str(e)[:200]}")
            raise
        finally:
            _DEPTH -= 1
        record(fn.__name__, _fmt_args(args, kwargs), _safe(result, _RES_CAP))
        return result
    wrapper.__wrapped_by_ledger__ = True
    return wrapper


def _fmt_args(args, kwargs):
    parts = [_safe(a, _ARG_CAP) for a in args]
    parts += [f"{k}={_safe(v, _ARG_CAP)}" for k, v in kwargs.items()]
    return ", ".join(parts)


def rows():
    return list(_ROWS)


def last_id():
    return len(_ROWS) - 1


def set_sink(cb):
    """The orchestrator (or any host) registers a callback to receive each
    row as it is written -- e.g. to stream into a session's evidence log."""
    global _SINK
    _SINK = cb


def reset():
    global _ROWS
    _ROWS = []

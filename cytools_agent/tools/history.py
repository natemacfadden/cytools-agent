# =============================================================================
# This file is part of CYTools-agent.
#
# CYTools-agent is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# CYTools-agent is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# CYTools-agent. If not, see <https://www.gnu.org/licenses/>.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  Call logging shared across cytools-agent tools. The `logged`
#               decorator records each successful structured tool call;
#               save_history renders the log to a runnable, self-documenting
#               Python script.
# -----------------------------------------------------------------------------

# 'standard' imports
import collections
import datetime
import functools

# session-wide call log
# ---------------------
_HISTORY = []  # list of {"tool", "module", "args", "kwargs"}
_SESSION_START = datetime.datetime.now().isoformat(timespec="seconds")

# non-model-facing
# ----------------
def logged(fn):
    """
    Decorator that records each successful call of `fn` in _HISTORY.

    Captures the tool name, its source module (so save_history can render the
    correct import even when tools live in different files), and the call args.
    `functools.wraps` preserves the signature/docstring so FastMCP can still
    introspect the wrapped function.

    Parameters
    ----------
    fn : callable
        The tool function to wrap.

    Returns
    -------
    callable
        The wrapped function.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        _HISTORY.append({
            "tool": fn.__name__,
            "module": fn.__module__,
            "args": list(args),
            "kwargs": dict(kwargs),
        })
        return result
    return wrapper

# model-facing
# ------------
def save_history(path: str) -> dict:
    """
    Write the log of structured tool calls so far to a runnable Python script.

    The script imports the tools used (grouped by source module), then replays
    each call wrapped in a print so running it re-executes the workflow and
    echoes every result.

    Only save when the user has asked for it or agreed to it. ASK THE USER for
    the file `path` -- do NOT invent one. Saving writes to the user's disk.

    Parameters
    ----------
    path : str
        The destination file path for the script.

    Returns
    -------
    dict
        A dict with the number of calls written and the path.
    """
    # group imports by source module so tools from any file render correctly
    mods = collections.defaultdict(set)
    for h in _HISTORY:
        mods[h["module"]].add(h["tool"])

    lines = [
        f"# cytools-agent session {_SESSION_START}",
        f"# {len(_HISTORY)} call(s)",
    ]
    for module in sorted(mods):
        lines.append(f"from {module} import {', '.join(sorted(mods[module]))}")
    lines.append("")
    for i, h in enumerate(_HISTORY):
        parts = [repr(a) for a in h["args"]]
        parts += [f"{k}={v!r}" for k, v in h["kwargs"].items()]
        call = f"{h['tool']}({', '.join(parts)})"
        lines.append(f'print(f"call {i}: `{call}` returned `{{{call}}}`")')
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return {"saved": len(_HISTORY), "path": path}

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
# Description:  Code escape hatch for open-ended work the structured tools do
#               not cover: run_python (a persistent namespace) and cytools_help
#               (look up CYTools signatures/docstrings without running code).
# -----------------------------------------------------------------------------

# external imports
import ast
import contextlib
import difflib
import inspect
import glob
import io
import linecache
import os
import sys
import time
import traceback

import numpy as np
import cytools

try:
    import matplotlib
    matplotlib.use("Agg")  # headless: run_python saves figures, never shows
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# local imports
from cytools_agent.tools import polytope, triangulation, cy

# persistent namespace: raw cytools + trusted tool functions. numpy (np),
# matplotlib.pyplot (plt) and the raw cytools library are preloaded too -- the
# engineer needs them for arrays, plots, and anything the tools don't cover.
_NS = {
    "cytools": cytools,
    "np": np,
    "plt": plt,
    "Polytope": cytools.Polytope,
    "get_polytope": polytope.get_polytope,
    "fetch_polytopes": polytope.fetch_polytopes,
    "get_polytope_info": polytope.get_polytope_info,
    "ks_stats": polytope.ks_stats,
    "get_heights": triangulation.get_heights,
    "get_triangulation_info": triangulation.get_triangulation_info,
    "get_cy": cy.get_cy,
    "get_cy_info": cy.get_cy_info,
    "get_cy_cones": cy.get_cy_cones,
}
# Adapt to the model's instinct: small models often write `import get_cy_info`
# even though the tools are preloaded. Registering each preloaded callable in
# sys.modules under its own name makes `import get_cy_info` bind to the function
# (and be callable), so we enable the behavior instead of erroring on it. We
# skip names already in sys.modules, so real modules (e.g. cytools) are never
# shadowed.
# KNOWN HAZARD: sys.modules now holds non-module objects, which can confuse
# third-party code that iterates it expecting modules (inspect.getmodule,
# importlib.reload, some pytest collection paths). Scoped to this process and
# accepted; if a library trips on it, filter with inspect.ismodule.
for _name, _obj in _NS.items():
    if _name not in sys.modules:
        sys.modules[_name] = _obj

_PRELOADED = list(_NS)   # tool names, captured before run_python adds vars
# the curated CALLABLE tools to steer the model to -- NOT the raw `cytools`
# module / `np` / `plt` / `Polytope` (telling it to "call cytools" sends it to
# the raw library, e.g. cytools.fetch_polytopes() with its misleading 1000
# default). np/plt/cytools are still available; they're mentioned separately.
_TOOL_NAMES = [n for n in _PRELOADED
               if n not in ("cytools", "np", "plt", "Polytope")]

_MAX_OUTPUT = 4000  # cap returned stdout to protect the context window
# Wall-clock cap for ONE run_python call (seconds; 0 disables). Observed: an
# engineer step burned 14+ minutes of CPU recomputing get_polytope_info (all
# fields, incl. automorphisms) for thousands of polytopes -- with no LLM in
# the loop to notice. The cap turns runaway computation into a pointed,
# recoverable error. Main-thread only (signals); other threads are uncapped.
_RUN_TIMEOUT = float(os.environ.get("CYTOOLS_RUN_TIMEOUT", "150"))


class _RunPythonTimeout(Exception):
    pass
_FIG_DIR = os.environ.get("CYTOOLS_AGENT_FIG_DIR", "scratch")   # sandboxable
_fig_count = 0


# human-read
def _assigned_names(tree):
    """Top-level variable names the code assigns (for the no-output hint)."""
    names = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names += [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif (isinstance(node, (ast.AnnAssign, ast.AugAssign))
              and isinstance(node.target, ast.Name)):
            names.append(node.target.id)
    return names


# human-read
def reset_figures():
    """Clear figures from the active figure dir and reset the counter, so a new
    session archives only the figures IT made. Sequential runs in one process
    otherwise share the dir, and save_log (which globs fig_*.png) would sweep
    up a prior run's leftovers."""
    global _fig_count
    _fig_count = 0
    for f in glob.glob(os.path.join(_FIG_DIR, "fig_*.png")):
        os.remove(f)


# human-read
def reset_namespace():
    """Clear user-added variables from the persistent run_python scratchpad, so
    each session starts CLEAN. The namespace (_NS) is module-global, so without
    this a variable from one session (e.g. polytope_ids) leaks into the next --
    corrupting multi-run processes like the eval harness. Preloaded tools and
    modules (captured in _PRELOADED) are kept."""
    for name in [n for n in _NS if n not in _PRELOADED]:
        del _NS[name]


# human-read
def _save_open_figures():
    """Save open matplotlib figures; return a note with paths."""
    if plt is None or not plt.get_fignums():
        return ""
    global _fig_count
    os.makedirs(_FIG_DIR, exist_ok=True)
    paths = []
    for n in plt.get_fignums():
        _fig_count += 1
        path = os.path.abspath(os.path.join(_FIG_DIR, f"fig_{_fig_count}.png"))
        plt.figure(n).savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close("all")
    return f"\n[saved {len(paths)} figure(s): {', '.join(paths)}]"


# human-read
def _describe(val):
    """A short size hint for a scratchpad value."""
    if isinstance(val, np.ndarray):
        return f"ndarray{tuple(val.shape)}"
    if isinstance(val, (str, list, tuple, dict, set)):
        return f"{type(val).__name__}[len {len(val)}]"
    return type(val).__name__


# human-read
def namespace_summary():
    """The user-defined names live in the persistent run_python scratchpad,
    each with a size hint -- so a multi-step session can see what it has
    already built (preloaded tools, modules, and privates are omitted)."""
    parts = [f"{name}={_describe(val)}" for name, val in _NS.items()
             if name not in _PRELOADED and not name.startswith("_")
             and not inspect.ismodule(val)]
    return ", ".join(parts) if parts else "(empty)"


# human-read
def _format_user_traceback(exc, code):
    """A traceback showing ONLY the user's code -- the offending line with its
    source text -- not run_python's exec/compile machinery. The source lines
    are surfaced by registering `code` in linecache under the <run_python>
    name (exec'd strings have no file, so tracebacks otherwise omit the line).
    Falls back to the full traceback if no user frame is found (e.g. a
    SyntaxError raised before execution)."""
    linecache.cache["<run_python>"] = (
        len(code), None, code.splitlines(keepends=True), "<run_python>")
    frames = [f for f in traceback.extract_tb(exc.__traceback__)
              if f.filename == "<run_python>"]
    only = "".join(traceback.format_exception_only(type(exc), exc))
    if not frames:
        return only if isinstance(exc, SyntaxError) else traceback.format_exc()
    return ("Traceback (most recent call last):\n"
            + "".join(traceback.format_list(frames)) + only)


# model-read
def run_python(code: str) -> str:
    """
    Execute Python in a persistent session and return its stdout.

    The namespace persists across calls, so variables and imports from earlier
    calls remain available. Preloaded (no import needed): `np` (numpy), `plt`
    (matplotlib.pyplot, for plots), `cytools`, `Polytope`, the trusted tool
    functions (`fetch_polytopes`, `get_polytope_info`,
    `get_heights`, `get_triangulation_info`, `get_cy_info`,
    `get_cy_cones`), and `get_polytope(ks_ind)` /
    `get_cy(ks_ind, heights)` for raw objects
    (e.g. `get_cy(...).intersection_numbers(in_basis=True, format="dense")` or
    `.second_chern_class(...)`). Prefer these over raw cytools. Anything you
    want to see must be printed; bare expression values are not returned. Any
    matplotlib figure you make is saved to disk automatically (its path is
    reported), so just build the plot -- no need to call savefig.

    Parameters
    ----------
    code : str
        Python source to execute.

    Returns
    -------
    str
        Captured stdout (a traceback is appended if the code raised), truncated
        to the last few KB if very long.
    """
    import signal
    import threading

    timed = (_RUN_TIMEOUT > 0
             and threading.current_thread() is threading.main_thread())

    def _timeout(*_):
        raise _RunPythonTimeout(
            f"run_python call exceeded {_RUN_TIMEOUT:.0f}s of wall clock and "
            f"was stopped. Compute LESS per call: only the field you need "
            f"(e.g. len(f.points()) directly, NOT get_polytope_info's full "
            f"record), on a SMALLER sample, saving partial results to the "
            f"scratchpad between calls.")

    buf = io.StringIO()
    if timed:
        _t0 = time.monotonic()
        prev = signal.signal(signal.SIGALRM, _timeout)
        prev_t = signal.setitimer(signal.ITIMER_REAL, _RUN_TIMEOUT)
    try:
        tree = ast.parse(code, "<run_python>")
        last = tree.body[-1] if tree.body else None
        with contextlib.redirect_stdout(buf):
            if isinstance(last, ast.Expr):
                # run all but the last statement, then echo the last
                # expression's value (Jupyter-style), so a bare expression is
                # not silently discarded
                exec(compile(ast.Module(tree.body[:-1], []), "<run_python>",
                             "exec"), _NS)
                val = eval(compile(ast.Expression(last.value), "<run_python>",
                                   "eval"), _NS)
                if val is not None:
                    print(repr(val))
            else:
                exec(compile(code, "<run_python>", "exec"), _NS)
        out = buf.getvalue()
        if not out:
            # nothing printed: name the variables the code assigned so the
            # model prints one instead of fabricating an answer
            names = _assigned_names(tree)
            hint = f" You assigned: {', '.join(names)}." if names else ""
            out = ("(no output -- nothing was printed." + hint
                   + " print() the value you need; do not report values you "
                   "did not see.)")
    except Exception as e:
        out = buf.getvalue() + "\n" + _format_user_traceback(e, code)
        missing = getattr(e, "name", None)
        if isinstance(e, ImportError) or (isinstance(e, NameError)
                                          and missing in _PRELOADED):
            # the tools are preloaded, not importable modules: a model that
            # writes `import CYTools` / `import get_cy_info` hits this. Point it
            # at the CURATED tools (NOT the raw `cytools` module, which would
            # send it to cytools.fetch_polytopes() and the misleading default).
            out += ("\n[these tools are already available here -- call them "
                    "directly, no import: " + ", ".join(_TOOL_NAMES) + ". Also "
                    "preloaded: np (numpy), plt (matplotlib.pyplot, for plots), "
                    "and the raw cytools library for anything the tools above "
                    "don't cover.]")
        elif isinstance(e, NameError):
            # a near-miss for a real tool name (e.g. get_polytopes ->
            # fetch_polytopes) -- point to it; the model's intent is clear
            near = difflib.get_close_matches(missing or "", _TOOL_NAMES, n=2,
                                             cutoff=0.6)
            if near:
                out += (f"\n[no {missing!r} here -- did you mean: "
                        + ", ".join(near) + "?]")
            else:
                # an undefined *variable* (often a doc placeholder copied
                # verbatim): assign/fetch it first, do not import it
                out += (f"\n[name {missing!r} is not defined yet -- assign it "
                        f"before use; the scratchpad holds: "
                        f"{namespace_summary()}]")
    finally:
        if timed:
            # disarm ours, restore the previous handler FIRST (so a pending
            # outer deadline fires into ITS handler), then re-arm the outer
            # timer with its remaining time minus what this call consumed
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, prev)
            if prev_t[0]:
                left = prev_t[0] - (time.monotonic() - _t0)
                signal.setitimer(signal.ITIMER_REAL, max(0.1, left))
    out += _save_open_figures()  # persist any plots so they can be viewed
    if len(out) > _MAX_OUTPUT:
        out = "...(truncated)...\n" + out[-_MAX_OUTPUT:]
    return out


# model-read
def cytools_help(name: str) -> dict:
    """
    Look up the signature and docstring of a CYTools object, without running it.

    Resolves dotted names like "Polytope.triangulate" in the run_python
    namespace and returns the call signature plus the docstring.

    Parameters
    ----------
    name : str
        A dotted name, e.g. "Polytope.triangulate" or "get_polytope".

    Returns
    -------
    dict
        {"signature", "doc"} for the object, or {"error": reason} if unresolved.
    """
    try:
        obj = eval(name, _NS)
    except Exception as e:
        return {"error": f"could not resolve {name!r}: {e}"}
    try:
        sig = name.split(".")[-1] + str(inspect.signature(obj))
    except (TypeError, ValueError):
        sig = None
    return {"signature": sig, "doc": inspect.getdoc(obj)}

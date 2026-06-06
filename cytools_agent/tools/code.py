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
import inspect
import io
import os
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

# persistent namespace: raw cytools + trusted tool functions
_NS = {
    "cytools": cytools,
    "np": np,
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
_MAX_OUTPUT = 4000  # cap returned stdout to protect the context window
_FIG_DIR = "scratch"
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


# model-read
def run_python(code: str) -> str:
    """
    Execute Python in a persistent session and return its stdout.

    The namespace persists across calls, so variables and imports from earlier
    calls remain available. Preloaded: `cytools`, `np`, `Polytope`, the trusted
    tool functions (`fetch_polytopes`, `get_polytope_info`,
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
    buf = io.StringIO()
    try:
        tree = ast.parse(code)
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
                exec(code, _NS)
        out = buf.getvalue()
        if not out:
            # nothing printed: name the variables the code assigned so the
            # model prints one instead of fabricating an answer
            names = _assigned_names(tree)
            hint = f" You assigned: {', '.join(names)}." if names else ""
            out = ("(no output -- nothing was printed." + hint
                   + " print() the value you need; do not report values you "
                   "did not see.)")
    except Exception:
        out = buf.getvalue() + "\n" + traceback.format_exc()
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

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
from cytools_agent.tools.history import logged

# persistent namespace shared across run_python / cytools_help calls. It holds
# raw cytools plus the trusted, model-facing tool functions, so code can call
# the known-good helpers instead of rediscovering the cytools API.
# ------------------------------------------------------------------
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
    "get_cy_info_at_point": cy.get_cy_info_at_point,
    "get_cy_cones": cy.get_cy_cones,
}
_MAX_OUTPUT = 4000  # cap returned stdout to protect the context window
_FIG_DIR = "scratch"  # run_python saves any plots here so they can be viewed
_fig_count = 0


def _save_open_figures():
    """Save any figures the code left open; return a note with their paths."""
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


# model-facing
# ------------
@logged
def run_python(code: str) -> str:
    """
    Execute Python in a persistent session and return its stdout.

    The namespace persists across calls, so variables and imports from earlier
    calls remain available. Preloaded: `cytools`, `np`, `Polytope`, the trusted
    tool functions (`fetch_polytopes`, `get_polytope_info`,
    `get_heights`, `get_triangulation_info`, `get_cy_info_at_point`,
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
        with contextlib.redirect_stdout(buf):
            exec(code, _NS)
        out = buf.getvalue() or "(no output)"
    except Exception:
        out = buf.getvalue() + "\n" + traceback.format_exc()
    out += _save_open_figures()  # persist any plots so they can be viewed
    if len(out) > _MAX_OUTPUT:
        out = "...(truncated)...\n" + out[-_MAX_OUTPUT:]
    return out


@logged
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

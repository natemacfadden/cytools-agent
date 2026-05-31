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
# Description:  Code escape hatch for open-ended work the structured tools do
#               not cover: run_python (a persistent namespace) and cytools_help
#               (look up CYTools signatures/docstrings without running code).
# -----------------------------------------------------------------------------

# 'standard' imports
import contextlib
import inspect
import io
import traceback

# 3rd party imports
import numpy as np

# CYTools import
import cytools

# cytools-agent imports
from cytools_agent.tools.history import logged
from cytools_agent.tools.polytope import get_polytope

# persistent namespace shared across run_python / cytools_help calls
# ------------------------------------------------------------------
_NS = {
    "cytools": cytools,
    "np": np,
    "get_polytope": get_polytope,
    "Polytope": cytools.Polytope,
}
_MAX_OUTPUT = 4000  # cap returned stdout to protect the context window

# model-facing
# ------------
@logged
def run_python(code: str) -> str:
    """
    Execute Python in a persistent session and return its stdout.

    The namespace persists across calls, so variables and imports from earlier
    calls remain available. Preloaded: `cytools`, `np`, `Polytope`, and
    `get_polytope(ks_ind)` to rebuild a fetched polytope. Anything you want to
    see must be printed; bare expression values are not returned.

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

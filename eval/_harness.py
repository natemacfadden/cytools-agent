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
# Description:  Shared setup for all eval scripts: Ollama client, tool set,
#               agent factory, and timeout machinery.
#
#               Everything under eval/ is human-read (developer tooling); none
#               of it is shown to the model.
# -----------------------------------------------------------------------------

# external imports
import os
import signal

from openai import OpenAI

import eval._env  # noqa: F401  (env pins; must precede cytools_agent imports)

# local imports
from cytools_agent.tools import MODEL_TOOLS
from cytools_agent.tools.glossary import glossary_context
from cytools_agent.schema import function_to_schema
from cytools_agent.agent import Agent
from cytools_agent.prompt import DEFAULT_SYSTEM_PROMPT
from eval.grading import TIMED_OUT

base = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
client = OpenAI(base_url=base + "/v1", api_key="ollama")

# same tool set as the MCP server (save_history is auto-registered by Agent)
TOOL_FNS = MODEL_TOOLS
tools = [function_to_schema(fn) for fn in TOOL_FNS]
tool_impls = {fn.__name__: fn for fn in TOOL_FNS}


def make_agent(model, max_steps=20, verbosity=0):
    return Agent(client, model, DEFAULT_SYSTEM_PROMPT, tools, tool_impls,
                 max_steps=max_steps, verbosity=verbosity,
                 message_hook=glossary_context)


# ---------------------------------------------------------------------------
# The L1 baseline: raw cytools in a plain REPL, vanilla loop. This is the
# good-faith counterfactual "what you get by handing an agent the library
# as-published" -- deliberately WITHOUT the curated layer (no id scheme, no
# forgiveness, no pointed errors, no iteration/search tools, no glossary, no
# guards). The gap between this and the curated-tool agent measures the tool
# layer's contribution. Same model, same loop, same step budget.
# ---------------------------------------------------------------------------
_RAW_NS = {}

RAW_SYSTEM_PROMPT = (
    "You are a research assistant answering questions about Calabi-Yau "
    "manifolds and reflexive polytopes. You have a Python tool; the cytools "
    "library is installed (import cytools) along with numpy. Discover the "
    "API with help(...) as needed. Print any value you need to see. Your "
    "final reply must state the concrete result (actual numbers)."
)


# model-read (docstring becomes the raw tool's schema)
def run_python_raw(code: str) -> str:
    """
    Execute Python in a persistent session and return its captured stdout.

    The namespace persists across calls. The cytools library and numpy (np)
    are importable. Print anything you want to see.

    Parameters
    ----------
    code : str
        Python source to execute.

    Returns
    -------
    str
        Captured stdout, with the traceback appended if the code raised.
    """
    import contextlib
    import io
    import traceback
    if not _RAW_NS:
        import cytools
        import numpy as np
        _RAW_NS.update(cytools=cytools, np=np, help=help)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, _RAW_NS)
        out = buf.getvalue() or "(no output)"
    except Exception:
        out = buf.getvalue() + "\n" + traceback.format_exc(limit=3)
    return out[-4000:]


def make_raw_agent(model, max_steps=20, verbosity=0):
    _RAW_NS.clear()
    return Agent(client, model, RAW_SYSTEM_PROMPT,
                 [function_to_schema(run_python_raw)],
                 {"run_python_raw": run_python_raw},
                 max_steps=max_steps, verbosity=verbosity)


# timeout (BaseException so the agent's `except Exception` can't swallow it)
class _TimedOut(BaseException):
    pass


signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(_TimedOut()))


def run(model, prompt, timeout=600, max_steps=20, raw=False):
    ag = (make_raw_agent if raw else make_agent)(model, max_steps=max_steps)
    signal.alarm(timeout)
    try:
        return (ag.chat(prompt) or "").strip().replace("\n", " ")
    except _TimedOut:
        return TIMED_OUT
    finally:
        signal.alarm(0)

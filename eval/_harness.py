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
# Description:  Shared harness for all eval scripts: Ollama client, tool set,
#               timeout machinery, and the run()/grade() helpers.
# -----------------------------------------------------------------------------

# external imports
import os
import signal

from openai import OpenAI

# local imports
from cytools_agent.tools import polytope, triangulation, cy, code, history
from cytools_agent.schema import function_to_schema
from cytools_agent.agent import Agent
from cytools_agent.prompt import DEFAULT_SYSTEM_PROMPT

# client
base = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
client = OpenAI(base_url=base + "/v1", api_key="ollama")

# tool set (single source of truth)
TOOL_FNS = [
    polytope.fetch_polytopes, polytope.get_polytope_info, polytope.ks_stats,
    triangulation.get_heights, triangulation.get_triangulation_info,
    cy.get_cy_info, cy.get_cy_cones,
    code.run_python, code.cytools_help,
    history.save_history,
]
tools = [function_to_schema(fn) for fn in TOOL_FNS]
tool_impls = {fn.__name__: fn for fn in TOOL_FNS}


def make_agent(model, max_steps=20, verbosity=0):
    ag = Agent(client, model, DEFAULT_SYSTEM_PROMPT, tools, tool_impls,
               max_steps=max_steps, verbosity=verbosity)
    ag.tool_impls["save_history"] = ag.save_script   # bind to this instance
    return ag


# timeout (BaseException so the agent's `except Exception` can't swallow it)
class _TimedOut(BaseException):
    pass


signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(_TimedOut()))

TIMED_OUT = "(timed out)"


def run(model, prompt, timeout=300, max_steps=20):
    ag = make_agent(model, max_steps=max_steps)
    signal.alarm(timeout)
    try:
        return (ag.chat(prompt) or "").strip().replace("\n", " ")
    except _TimedOut:
        return TIMED_OUT
    finally:
        signal.alarm(0)

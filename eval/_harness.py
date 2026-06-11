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

# pin the sampled prompt examples for eval stability (see eval_orch.py);
# must happen before cytools_agent imports read the env
os.environ.setdefault("CYTOOLS_EXAMPLE_SEED", "0")
# evals re-run identical queries constantly: opt in to the on-disk KS cache
# (a dev feature, off by default so end users don't accumulate a large file)
os.environ.setdefault("CYTOOLS_AGENT_KS_CACHE", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scratch", "ks_cache.json"))

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


# timeout (BaseException so the agent's `except Exception` can't swallow it)
class _TimedOut(BaseException):
    pass


signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(_TimedOut()))


def run(model, prompt, timeout=600, max_steps=20):
    ag = make_agent(model, max_steps=max_steps)
    signal.alarm(timeout)
    try:
        return (ag.chat(prompt) or "").strip().replace("\n", " ")
    except _TimedOut:
        return TIMED_OUT
    finally:
        signal.alarm(0)

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
# Description:  Regression eval for the agent. Runs a fixed task suite N times
#               per task against a given model and auto-grades the answers, so
#               prompt/model/tool changes can be checked for regressions.
#
# Usage (in the cytools-agent env, with Ollama serving the model):
#     python eval/run_eval.py qwen3:4b 5
# -----------------------------------------------------------------------------

# external imports
import os
import sys

from openai import OpenAI

# local imports
from cytools_agent.tools import polytope, triangulation, code, files, history
from cytools_agent.schema import function_to_schema
from cytools_agent.agent import Agent
from cytools_agent.prompt import DEFAULT_SYSTEM_PROMPT

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3:4b"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5

base = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
client = OpenAI(base_url=base + "/v1", api_key="ollama")

TOOL_FNS = [
    polytope.fetch_polytopes, polytope.get_polytope_info,
    triangulation.all_inequiv_heights, triangulation.get_triangulation_info,
    code.run_python, code.cytools_help,
    files.read_file, history.save_history,
]
tools = [function_to_schema(fn) for fn in TOOL_FNS]
tool_impls = {fn.__name__: fn for fn in TOOL_FNS}

# tasks: (label, prompt, max_steps, grader). graders check the known answer.
def has_ids(a):
    return all(i in a for i in ("h11-5_h21-20_ind-0", "h11-5_h21-29_ind-0",
                                "h11-5_h21-29_ind-1"))

TASKS = [
    ("simple fetch", "Fetch 3 polytopes at h11=5", 6, has_ids),
    ("favorable frac", "What fraction of polytopes at h11=2 are N-favorable? "
     "Use a sample of 20.", 10,
     lambda a: any(s in a.lower() for s in ("1.0", "100%", "20/20", "all 20"))),
    ("NTFE avg", "On average, how many NTFEs do polytopes at h11=2 have? "
     "Sample 5.", 10, lambda a: "1.2" in a),
    ("max simplices", "What is the maximum simplex count among "
     "triangulations of polytopes at h11=3? Sample 3 polytopes.", 14,
     lambda a: "12" in a),
]


def main():
    print(f"###### {MODEL}  (N={N} each) ######")
    for label, task, steps, check in TASKS:
        passes, answers = 0, []
        for _ in range(N):
            agent = Agent(client, MODEL, DEFAULT_SYSTEM_PROMPT, tools,
                          tool_impls, max_steps=steps, verbosity=0)
            ans = (agent.chat(task) or "").strip().replace("\n", " ")
            ok = check(ans)
            passes += ok
            answers.append(("PASS" if ok else "FAIL", ans[:80]))
        print(f"\n## [{label}] {passes}/{N} passed")
        for tag, a in answers:
            print(f"   {tag}: {a}")


if __name__ == "__main__":
    main()

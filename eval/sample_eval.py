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
# Description:  Run a model (the cytools-agent) on a stratified SAMPLE of the
#               verified corpus and grade its free-text answer against the
#               ground truth. Measures whether the agent can actually answer
#               the corpus questions with its tools.
#
# Usage:  python eval/sample_eval.py qwen3:8b [n_samples] [timeout_s]
# -----------------------------------------------------------------------------

# external imports
import json
import os
import random
import re
import signal
import sys

from openai import OpenAI

# local imports
from cytools_agent.tools import (polytope, triangulation, cy, code, files,
                                 history)
from cytools_agent.schema import function_to_schema
from cytools_agent.agent import Agent
from cytools_agent.prompt import DEFAULT_SYSTEM_PROMPT

base = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
client = OpenAI(base_url=base + "/v1", api_key="ollama")

TOOL_FNS = [
    polytope.fetch_polytopes, polytope.get_polytope_info, polytope.ks_stats,
    triangulation.get_heights, triangulation.get_triangulation_info,
    cy.get_cy_info, cy.get_cy_cones,
    code.run_python, code.cytools_help,
    files.read_file, history.save_history,
]
tools = [function_to_schema(fn) for fn in TOOL_FNS]
tool_impls = {fn.__name__: fn for fn in TOOL_FNS}


class _TimedOut(BaseException):
    pass


signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(_TimedOut()))

TIMED_OUT = "(timed out)"   # sentinel: inconclusive, not a wrong answer


def run(model, prompt, timeout=300):
    ag = Agent(client, model, DEFAULT_SYSTEM_PROMPT, tools, tool_impls,
               max_steps=20, verbosity=0)
    signal.alarm(timeout)
    try:
        return (ag.chat(prompt) or "").strip().replace("\n", " ")
    except _TimedOut:
        return TIMED_OUT
    finally:
        signal.alarm(0)


def grade(ans, truth):
    """Return 'TIMEOUT' (inconclusive), 'PASS', or 'FAIL'."""
    if ans == TIMED_OUT:
        return "TIMEOUT"
    return "PASS" if hit(ans, truth) else "FAIL"


def _flat(x):
    if isinstance(x, (list, tuple)):
        return [e for s in x for e in _flat(s)]
    return [x]


def _claimed_ints(text):
    """Integers the answer presents as its result: the leading number of a
    bold span, or a number right after an 'answer/total/count' marker. Avoids
    crediting digits that merely echo the question (e.g. 'h11=3')."""
    claims = []
    for span in re.findall(r"\*\*(.+?)\*\*", text):
        m = re.match(r"\s*(-?[\d,]+)\b", span)
        if m:
            claims.append(m.group(1).replace(",", ""))
    for m in re.finditer(r"(?:answer|total|count)\D{0,12}?(-?\d[\d,]*)",
                         text, re.I):
        claims.append(m.group(1).replace(",", ""))
    return claims


def hit(text, ans):
    """Approximate: does the ground-truth value appear in the answer text?"""
    t = text.lower()
    if isinstance(ans, bool):
        return (bool(re.search(r"\byes\b|\btrue\b", t)) if ans
                else bool(re.search(r"\bno\b|\bfalse\b|\bnot\b|non-", t)))
    if isinstance(ans, int):
        claims = _claimed_ints(text)
        if claims:                       # grade the stated answer, not echoes
            return str(ans) in claims
        return str(ans) in re.sub(r",", "", text)
    if isinstance(ans, float):
        return any(f"{round(ans, d)}" in text for d in (1, 2, 3, 4, 6))
    if isinstance(ans, (list, tuple)):
        if re.sub(r"\s", "", str(ans)) in re.sub(r"\s", "", text):
            return True
        return all((f"{round(e, 3)}" in text if isinstance(e, float)
                    else str(e) in re.sub(r",", "", text)) for e in _flat(ans))
    return str(ans).lower() in t


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3:8b"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    timeout = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    random.seed(0)

    rows = [json.loads(line)
            for line in open(os.path.join(os.path.dirname(__file__),
                                           "corpus.jsonl"))]
    by_kind = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r)
    kinds = list(by_kind)
    random.shuffle(kinds)
    sample = [random.choice(by_kind[kind]) for kind in kinds][:k]

    print(f"###### {model} on {len(sample)} sampled corpus questions ######",
          flush=True)
    passed = failed = timed = 0
    for i, r in enumerate(sample):
        ans = run(model, r["question"], timeout)
        status = grade(ans, r["answer"])
        passed += status == "PASS"
        failed += status == "FAIL"
        timed += status == "TIMEOUT"
        print(f"\n[{i+1}] {r['kind']} (id {r['id']})  {status}", flush=True)
        print(f"    Q: {r['question'][:95]}", flush=True)
        print(f"    truth: {r['answer']}", flush=True)
        print(f"    got:   {ans[:110]}", flush=True)
    scored = passed + failed
    rate = f"{passed}/{scored}" if scored else "0/0"
    print(f"\n###### {model}: {rate} scored correct "
          f"({timed} timed out, inconclusive) ######", flush=True)


if __name__ == "__main__":
    main()

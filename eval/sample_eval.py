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
# Description:  Run a model on a stratified sample of the verified corpus and
#               grade its free-text answer against ground truth.
#
# Usage:  python eval/sample_eval.py qwen3:8b [n_samples] [timeout_s]
# -----------------------------------------------------------------------------

# external imports
import json
import os
import random
import re
import sys

# local imports
from eval._harness import run, TIMED_OUT

def _flat(x):
    if isinstance(x, (list, tuple)):
        return [e for s in x for e in _flat(s)]
    return [x]


def _claimed_ints(text):
    """Integers the answer explicitly states: leading number in a bold span,
    or after an 'answer/total/count' marker. Avoids crediting echoed digits."""
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
    """Does the ground-truth value appear in the answer text?"""
    t = text.lower()
    if isinstance(ans, bool):
        return (bool(re.search(r"\byes\b|\btrue\b", t)) if ans
                else bool(re.search(r"\bno\b|\bfalse\b|\bnot\b|non-", t)))
    if isinstance(ans, int):
        claims = _claimed_ints(text)
        if claims:
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


def grade(ans, truth):
    """'TIMEOUT' (inconclusive), 'PASS', or 'FAIL'."""
    if ans == TIMED_OUT:
        return "TIMEOUT"
    return "PASS" if hit(ans, truth) else "FAIL"


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
    print(f"\n###### {model}: {passed}/{scored} scored correct "
          f"({timed} timeout) ######", flush=True)


if __name__ == "__main__":
    main()

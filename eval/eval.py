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
# Description:  Corpus-based evaluation for the CYTools agent. Two modes:
#
#   Sampling (default): stratified random sample over corpus kinds, one
#   question per kind. Good for a quick overall pass-rate estimate.
#     python -m eval.eval qwen3:8b [k=12] [timeout=300]
#
#   Targeted: specific corpus ids, repeated reps times. Use this to measure
#   whether a fix helps before committing it (run BEFORE, apply fix, run AFTER,
#   compare; undo if it doesn't help).
#     python -m eval.eval qwen3:8b --ids 54,57,58 [--reps 3] [--timeout 260]
#
# Both modes share the same grader and report PASS / FAIL / TIMEOUT (the last
# is inconclusive and excluded from the scored denominator).
#
# All functions here are human-read (developer tooling).
# -----------------------------------------------------------------------------

# external imports
import json
import os
import random
import re
import sys

# local imports
from eval._harness import run, TIMED_OUT


# grader
# ------
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


def _print_result(label, status, question, truth, ans):
    print(f"\n{label}  {status}", flush=True)
    print(f"    Q: {question[:95]}", flush=True)
    print(f"    truth: {truth}", flush=True)
    print(f"    got:   {ans[:110]}", flush=True)


def _summary(npass, nfail, ntimeout):
    scored = npass + nfail
    print(f"\n###### {npass}/{scored} scored correct "
          f"({ntimeout} timeout) ######", flush=True)


# modes
# -----
def run_sample(model, k, timeout):
    """Stratified random sample: one question per kind, up to k total."""
    random.seed(0)
    path = os.path.join(os.path.dirname(__file__), "corpus.jsonl")
    rows = [json.loads(line) for line in open(path)]
    by_kind = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r)
    kinds = list(by_kind)
    random.shuffle(kinds)
    sample = [random.choice(by_kind[kind]) for kind in kinds][:k]

    print(f"###### {model} on {len(sample)} sampled corpus questions ######",
          flush=True)
    npass = nfail = ntimeout = 0
    for i, r in enumerate(sample):
        ans = run(model, r["question"], timeout)
        status = grade(ans, r["answer"])
        npass += status == "PASS"
        nfail += status == "FAIL"
        ntimeout += status == "TIMEOUT"
        _print_result(f"[{i+1}] {r['kind']} (id {r['id']})", status,
                      r["question"], r["answer"], ans)
    _summary(npass, nfail, ntimeout)


def run_targeted(model, ids, reps, timeout):
    """Specific corpus ids, repeated reps times each."""
    corpus = os.path.join(os.path.dirname(__file__), "corpus.jsonl")
    rows = {r["id"]: r for r in (json.loads(line) for line in open(corpus))}
    print(f"###### {model} on ids {ids} x{reps} ######", flush=True)
    npass = nfail = ntimeout = 0
    for i in ids:
        r = rows[i]
        results = []
        for _ in range(reps):
            ans = run(model, r["question"], timeout)
            status = grade(ans, r["answer"])
            results.append((status, ans))
            npass += status == "PASS"
            nfail += status == "FAIL"
            ntimeout += status == "TIMEOUT"
        p = sum(s == "PASS" for s, _ in results)
        to = sum(s == "TIMEOUT" for s, _ in results)
        tag = f"{p}/{reps}" + (f" ({to} timeout)" if to else "")
        print(f"\n[{i}] {r['kind']}  {tag}", flush=True)
        print(f"    Q: {r['question'][:95]}", flush=True)
        print(f"    truth: {r['answer']}", flush=True)
        for status, ans in results:
            print(f"    {status}: {ans[:110]}", flush=True)
    _summary(npass, nfail, ntimeout)


USAGE = ("usage: python -m eval.eval model [k] [timeout]\n"
         "       python -m eval.eval model --ids 1,2,3 "
         "[--reps N] [--timeout S]")


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(USAGE)
        sys.exit(0 if args else 1)

    model = args[0]
    rest = args[1:]

    if "--ids" in rest:
        i = rest.index("--ids")
        ids = [int(x) for x in rest[i + 1].split(",")]
        reps = int(rest[rest.index("--reps") + 1]) if "--reps" in rest else 3
        timeout = int(rest[rest.index("--timeout") + 1]) \
            if "--timeout" in rest else 300
        run_targeted(model, ids, reps, timeout)
    else:
        k = int(rest[0]) if rest else 12
        timeout = int(rest[1]) if len(rest) > 1 else 300
        run_sample(model, k, timeout)


if __name__ == "__main__":
    main()

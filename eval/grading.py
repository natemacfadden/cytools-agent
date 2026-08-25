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
# Description:  Model-agnostic corpus run loop, shared by eval.py (local Ollama
#               agent) and eval_claude.py (headless Claude Code). A
#               `run_fn(question) -> answer_text` is the only thing that differs,
#               so both are scored by the same typed grader (eval/answer.py
#               grade_typed) on the same corpus, directly comparable. The run_fn
#               emits a <final> block (wrap it with eval.emit.finalizing).
#
# All functions here are human-read (developer tooling).
# -----------------------------------------------------------------------------

# external imports
import json
import math
import os
import random

# local imports
from eval.answer import grade_typed   # the shared typed grader

TIMED_OUT = "(timed out)"   # sentinel a run_fn returns when it times out


# corpus + reporting
# ------------------
def _rows():
    path = os.path.join(os.path.dirname(__file__), "corpus.jsonl")
    return [json.loads(line) for line in open(path)]


def wilson(k, n, z=1.96):
    """Wilson 95% score interval for k successes in n trials. Used instead of
    the normal (CLT) interval because eval samples here are small (n <= ~100),
    where CLT intervals are badly miscalibrated near 0% and 100%."""
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    d = 1 + z * z / n
    center = (ph + z * z / (2 * n)) / d
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, center - half), min(1.0, center + half))


def _phrasings(r):
    """All wordings of a row's question: the canonical one plus any true
    paraphrases (same selection procedure, same quantity, reworded -- NEVER
    coincidence pairs like favorable-vs-plain, which are different questions
    that merely agree on today's data)."""
    return [r["question"]] + list(r.get("paraphrases", []))


def _summary(header, npass, nfail, ntimeout):
    scored = npass + nfail
    lo, hi = wilson(npass, scored)
    ci = f", 95% CI [{lo:.0%}, {hi:.0%}]" if scored else ""
    print(f"\n###### {header}: {npass}/{scored} scored correct{ci} "
          f"({ntimeout} timeout) ######", flush=True)


# modes: run_fn(question) -> answer text (or TIMED_OUT)
# ------------------------------------------------------
def run_sample(run_fn, header, k):
    """Stratified random sample: one question per kind, up to k total."""
    random.seed(0)
    by_kind = {}
    for r in _rows():
        by_kind.setdefault(r["kind"], []).append(r)
    kinds = list(by_kind)
    random.shuffle(kinds)
    sample = [random.choice(by_kind[kind]) for kind in kinds][:k]

    print(f"###### {header} on {len(sample)} sampled corpus questions ######",
          flush=True)
    npass = nfail = ntimeout = 0
    for i, r in enumerate(sample):
        ans = run_fn(random.choice(_phrasings(r)))
        status = grade_typed(ans, r["answer"])
        npass += status == "PASS"
        nfail += status == "FAIL"
        ntimeout += status == "TIMEOUT"
        print(f"\n[{i+1}] {r['kind']} (id {r['id']})  {status}", flush=True)
        print(f"    Q: {r['question'][:95]}", flush=True)
        print(f"    truth: {r['answer']}", flush=True)
        print(f"    got:   {ans[:110]}", flush=True)
    _summary(header, npass, nfail, ntimeout)


def run_targeted(run_fn, header, ids, reps):
    """Specific corpus ids, repeated reps times each. Besides the pooled
    pass rate, reports pass^k (all reps of an id correct): the reliability
    number -- an agent that is right half the time on every question has
    pass@1 ~ 50% but pass^k ~ 0."""
    rows = {r["id"]: r for r in _rows()}
    print(f"###### {header} on ids {ids} x{reps} ######", flush=True)
    npass = nfail = ntimeout = 0
    all_pass = 0
    for i in ids:
        r = rows[i]
        phr = _phrasings(r)
        results = []
        # reps cycle through the phrasings, so with paraphrases present the
        # pass^k line doubles as a wording-consistency check
        for rep in range(reps):
            ans = run_fn(phr[rep % len(phr)])
            status = grade_typed(ans, r["answer"])
            results.append((status, ans))
            npass += status == "PASS"
            nfail += status == "FAIL"
            ntimeout += status == "TIMEOUT"
        p = sum(s == "PASS" for s, _ in results)
        to = sum(s == "TIMEOUT" for s, _ in results)
        all_pass += p == reps
        tag = f"{p}/{reps}" + (f" ({to} timeout)" if to else "")
        print(f"\n[{i}] {r['kind']}  {tag}", flush=True)
        print(f"    Q: {r['question'][:95]}", flush=True)
        print(f"    truth: {r['answer']}", flush=True)
        for status, ans in results:
            print(f"    {status}: {ans[:110]}", flush=True)
    print(f"\npass^{reps}: {all_pass}/{len(ids)} ids correct on every rep",
          flush=True)
    _summary(header, npass, nfail, ntimeout)

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
# Description:  Model-agnostic corpus grading + run loop, shared by eval.py
#               (local Ollama agent) and eval_claude.py (headless Claude Code).
#               A `run_fn(question) -> answer_text` is the only thing that
#               differs between them, so both are scored by the SAME grader on
#               the SAME corpus -- directly comparable.
#
# All functions here are human-read (developer tooling).
# -----------------------------------------------------------------------------

# external imports
import json
import os
import random
import re

TIMED_OUT = "(timed out)"   # sentinel a run_fn returns when it times out


# grader
# ------
def _flat(x):
    if isinstance(x, (list, tuple)):
        return [e for s in x for e in _flat(s)]
    return [x]


def _nums(text):
    """Every number in the text, in order, rounded to 3 dp for comparison."""
    return [round(float(x), 3) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]


# Digits that are part of CYTools jargon, NOT a stated result -- e.g. the "2" in
# "2-face", the "11"/"4" in "h11=4", the "2" in "c2" or "(2,1)", the "3" in "K3".
# Stripping these before number-matching stops a truth value from being credited
# just because it collides with a domain term in the answer's prose.
_DOMAIN_NOISE = re.compile(
    r"h\^?\d+(?:\s*,\s*\d+)?(?:\s*=\s*-?\d+)?"          # h11, h21, h^1,1, h11=4
    r"|\bc_?\d+\b"                                       # c2, c_2
    r"|\d+-(?:face|faces|fold|folds|dimensional|cycle|cycles|form|forms|plane)"
    r"|\(\s*-?\d+\s*,\s*-?\d+\s*\)"                      # (1,1), (2,1)
    r"|\b[KP]\d+\b|\bZ_?\d+\b|\bCP\d+\b|\bSU\(\d+\)|\bE\d\b"  # K3, P1, Z2, CP3...
    r"|\b\d+[dD]\b",                                     # 4d, 3D
    re.I)


def _denoise(text):
    """Blank out domain-term digits so number-matching sees only real values."""
    return _DOMAIN_NOISE.sub(" ", text)


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
        if claims:                                  # explicit "answer: N"/**N**
            return str(ans) in claims
        # else: a standalone integer (not inside a longer number, and not part
        # of a domain term like 2-face / h11=4) in the de-noised text
        return bool(re.search(rf"(?<!\d){re.escape(str(ans))}(?!\d)",
                              re.sub(r",", "", _denoise(text))))
    if isinstance(ans, float):
        return any(f"{round(ans, d)}" in text for d in (1, 2, 3, 4, 6))
    if isinstance(ans, (list, tuple)):
        # exact list literal present (handles ordered / nested forms)
        if re.sub(r"\s", "", str(ans)) in re.sub(r"\s", "", text):
            return True
        # else demand a CONTIGUOUS run of numbers whose multiset matches the
        # whole answer -- so e.g. "12 simplices" can't satisfy a [1,...,2,...]
        # truth just because the digits 1 and 2 appear somewhere
        target = sorted(round(float(e), 3) for e in _flat(ans))
        nums, w = _nums(_denoise(text)), len(target)
        return any(sorted(nums[i:i + w]) == target
                   for i in range(len(nums) - w + 1))
    return str(ans).lower() in t


def grade(ans, truth):
    """'TIMEOUT' (inconclusive), 'PASS', or 'FAIL'."""
    if ans == TIMED_OUT:
        return "TIMEOUT"
    return "PASS" if hit(ans, truth) else "FAIL"


# corpus + reporting
# ------------------
def _rows():
    path = os.path.join(os.path.dirname(__file__), "corpus.jsonl")
    return [json.loads(line) for line in open(path)]


def _summary(header, npass, nfail, ntimeout):
    scored = npass + nfail
    print(f"\n###### {header}: {npass}/{scored} scored correct "
          f"({ntimeout} timeout) ######", flush=True)


# modes -- run_fn(question) -> answer text (or TIMED_OUT)
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
        ans = run_fn(r["question"])
        status = grade(ans, r["answer"])
        npass += status == "PASS"
        nfail += status == "FAIL"
        ntimeout += status == "TIMEOUT"
        print(f"\n[{i+1}] {r['kind']} (id {r['id']})  {status}", flush=True)
        print(f"    Q: {r['question'][:95]}", flush=True)
        print(f"    truth: {r['answer']}", flush=True)
        print(f"    got:   {ans[:110]}", flush=True)
    _summary(header, npass, nfail, ntimeout)


def run_targeted(run_fn, header, ids, reps):
    """Specific corpus ids, repeated reps times each."""
    rows = {r["id"]: r for r in _rows()}
    print(f"###### {header} on ids {ids} x{reps} ######", flush=True)
    npass = nfail = ntimeout = 0
    for i in ids:
        r = rows[i]
        results = []
        for _ in range(reps):
            ans = run_fn(r["question"])
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
    _summary(header, npass, nfail, ntimeout)

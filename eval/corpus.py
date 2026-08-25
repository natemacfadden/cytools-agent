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
# Description:  Corpus build and verification. Two modes:
#
#   build: collect Q&A from notebook-mining agent transcripts, gate on
#   self-contained reproducibility, and write eval/corpus.jsonl.
#     python -m eval.corpus build
#
#   verify: re-execute every corpus entry's code in a fresh interpreter and
#   check that the printed result matches the stored answer.
#     python -m eval.corpus verify
#
#   selfcheck: grade every stored truth against itself through the typed
#   grader (build_final -> parse_final -> check), across all corpora with
#   stored answers. Needs no cytools/network; catches truths the grader
#   cannot represent (a truth that fails selfcheck can never be answered
#   correctly by any model).
#     python -m eval.corpus selfcheck
#
# All functions here are human-read (developer tooling).
# -----------------------------------------------------------------------------

# external imports
import ast
import html
import json
import os
import subprocess
import sys
from collections import Counter

PY = sys.executable   # run snippets with the same interpreter as this script
ROOT = os.path.dirname(os.path.dirname(__file__))
CORPUS = os.path.join(os.path.dirname(__file__), "corpus.jsonl")

# `build` is provenance for the existing corpus.jsonl, not a portable tool: it
# scrapes the agent transcripts from the one session that produced the corpus
# (paths below are machine-specific). `verify` is the reusable entry point.
TASKS = ("/private/tmp/claude-501/-Users-natemacfadden-cytools-agent/"
         "934d9ed9-728d-469b-ac40-9b402c5d6310/tasks")
AGENTS = {
    "cytools_ext": "ac7b1f8c680326ce8",
    "geom": "aca32575739167921",
    "physics": "af9b79662d3f6ce1c",
    "sm_in_IIB": "a5496b4ce304c9130",
}


# shared: standalone code execution + answer comparison
# ------------------------------------------------------
def _run_code(code):
    return subprocess.run([PY, "-c", code], capture_output=True, text=True,
                          cwd=ROOT, timeout=180)


def _eq(out, ans):
    out = out.strip()
    try:
        val = ast.literal_eval(out)
    except (ValueError, SyntaxError):
        return out == str(ans)
    if val == ans:
        return True
    try:
        import numpy as np
        return np.allclose(np.array(val, float), np.array(ans, float),
                           atol=1e-4)
    except Exception:
        return False


# build
# -----
def _texts(path):
    for line in open(path):
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") != "assistant":
            continue
        c = ev.get("message", {}).get("content", "")
        yield c if isinstance(c, str) else "".join(
            b.get("text", "") for b in c
            if isinstance(b, dict) and b.get("type") == "text")


def _final_jsonl(path):
    best = []
    for text in _texts(path):
        rows = []
        for ln in text.splitlines():
            ln = html.unescape(ln.strip())
            if ln.startswith('{"question"'):
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
        if rows:
            best = rows
    return best


def build():
    seen, kept, dropped = set(), [], []
    for label, aid in AGENTS.items():
        for r in _final_jsonl(os.path.join(TASKS, f"{aid}.output")):
            q = r.get("question")
            if not q or q in seen:
                continue
            seen.add(q)
            code = r.get("code", "")
            if not code or "print(" not in code:
                dropped.append((label, r.get("kind"), "no print"))
                continue
            try:
                p = _run_code(code)
            except subprocess.TimeoutExpired:
                dropped.append((label, r.get("kind"), "timeout"))
                continue
            if p.returncode != 0:
                dropped.append((label, r.get("kind"),
                                 "error: " + p.stderr.strip()[-60:]))
                continue
            if _eq(p.stdout, r.get("answer")):
                r["agent"] = label
                kept.append(r)
            else:
                dropped.append((label, r.get("kind"),
                                 "got " + p.stdout.strip()[:40]))

    with open(CORPUS, "w") as f:
        for i, r in enumerate(kept):
            r["id"] = i
            f.write(json.dumps(r) + "\n")

    print(f"kept {len(kept)}, dropped {len(dropped)} -> {CORPUS}")
    print("by agent:", dict(Counter(r["agent"] for r in kept)))
    print(f"distinct kinds: {len(set(r.get('kind') for r in kept))}")
    for d in dropped[:25]:
        print("  DROP", d)


# verify
# ------
def verify():
    rows = [json.loads(line) for line in open(CORPUS)]
    ok = mismatch = err = skip = 0
    bad = []
    for r in rows:
        code = r.get("code", "")
        if not (code.lstrip().startswith(("from ", "import "))
                and "print(" in code):
            skip += 1
            continue
        p = _run_code(code)
        if p.returncode != 0:
            err += 1
            bad.append(("ERR", r["id"], r["kind"], p.stderr.strip()[-80:]))
        elif _eq(p.stdout, r["answer"]):
            ok += 1
        else:
            mismatch += 1
            bad.append(("MISMATCH", r["id"], r["kind"],
                        f"got {p.stdout.strip()[:40]} want {r['answer']}"))

    print(f"verified {ok}, mismatch {mismatch}, error {err}, "
          f"skipped {skip} / {len(rows)} total")
    for b in bad[:20]:
        print(" ", b)


# selfcheck
# ---------
def _paraphrase_lint(name, r):
    """A paraphrase must be the SAME question reworded -- same selection
    procedure, same quantity. A wording that adds/drops a favorability
    restriction or changes the Hodge-number filter is a different cut of the
    database that may only coincidentally pick the same polytope today, so it
    is rejected here rather than trusted as a paraphrase."""
    import re
    probs = []
    q = r.get("question", "")
    sig = lambda s: ("favorable" in s.lower(),
                     sorted(re.findall(r"h(?:11|21)\s*=?\s*\d+", s)))
    for p in r.get("paraphrases", []):
        if sig(p) != sig(q):
            probs.append((name, r.get("id"), r.get("kind"),
                          "paraphrase changes the selection procedure "
                          f"(favorability/Hodge filter): {p[:60]!r}"))
    return probs

def selfcheck():
    """Every stored truth, graded against itself through the typed grader.
    Returns the list of failures (empty = all self-gradable)."""
    from eval.answer import build_final, check, parse_final, truth_kind
    here = os.path.dirname(__file__)
    bad = []
    notes = []
    total = 0
    for name in ("corpus.jsonl", "corpus_quarantined.jsonl",
                 "pm_corpus.jsonl", "pm_corpus_quarantined.jsonl",
                 "ms_corpus.jsonl", "ladder.jsonl", "heldout.jsonl"):
        path = os.path.join(here, name)
        if not os.path.exists(path):
            continue
        for line in open(path):
            r = json.loads(line)
            truth = r.get("answer")
            if truth is None:        # held-out rows carry no stored truth
                continue
            # a prose string (other than the IMPOSSIBLE marker) is a human
            # note, not a typed truth -- the <final> contract has no string
            # kind, so it cannot be auto-graded. Surface it, don't fail it.
            if (isinstance(truth, str)
                    and truth.strip().upper() != "IMPOSSIBLE"):
                notes.append((name, r.get("id"), r.get("kind"), truth))
                continue
            total += 1
            kind = truth_kind(truth)
            value = None if kind == "impossible" else truth
            try:
                ok = check(parse_final(build_final(kind, value)), truth)
            except Exception as e:
                ok = False
                bad.append((name, r.get("id"), r.get("kind"),
                            f"grader raised {type(e).__name__}: {e}"))
                continue
            if not ok:
                bad.append((name, r.get("id"), r.get("kind"),
                            "truth does not grade against itself"))
            bad.extend(_paraphrase_lint(name, r))
    print(f"selfcheck: {total - len(bad)}/{total} truths self-grade"
          + (f" ({len(notes)} prose-note answers skipped)" if notes else ""))
    for n in notes:
        print("  NOTE (not auto-gradable)", n)
    for b in bad:
        print("  BAD", b)
    return bad


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build":
        build()
    elif cmd == "verify":
        verify()
    elif cmd == "selfcheck":
        sys.exit(1 if selfcheck() else 0)
    else:
        print("usage: python -m eval.corpus build | verify | selfcheck")
        sys.exit(0 if cmd in ("-h", "--help") else 1)


if __name__ == "__main__":
    main()

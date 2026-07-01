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
# Description:  Validation harness for evidence-based grading: re-grade every
#               ARCHIVED session (scratch/logs/session_*.json) BOTH ways --
#               prose (the Coordinator's final answer text) and evidence (truth must
#               appear in a computed received_output) -- and print every
#               disagreement with enough context to adjudicate by eye. The
#               sessions' questions are matched to pm_corpus truths.
#
#     python -m eval.regrade_logs
#
# All functions here are human-read (developer tooling).
# -----------------------------------------------------------------------------

# external imports
import glob
import json
import os

# local imports
from eval.eval_orch import evidence_grade
from eval.grading import hit

LOG_DIR = os.path.join("scratch", "logs")
PM_CORPUS = os.path.join(os.path.dirname(__file__), "pm_corpus.jsonl")


def _truths():
    rows = [json.loads(l) for l in open(PM_CORPUS)]
    return {r["question"].strip(): (r["id"], r["answer"]) for r in rows}


def main():
    truths = _truths()
    n = agree = 0
    disagreements = []
    unmatched = 0
    for path in sorted(glob.glob(os.path.join(LOG_DIR, "session_*.json"))):
        if "_fig_" in path or path.endswith("_replay.py"):
            continue
        d = json.load(open(path))
        q = (d.get("question") or "").strip()
        if q not in truths:
            unmatched += 1
            continue
        cid, truth = truths[q]
        answer = d.get("answer") or ""
        prose = "PASS" if hit(answer, truth) else "FAIL"
        evid = evidence_grade(truth, d.get("evidence") or [])
        # the honest gate from eval_orch: a stopped walk cannot PASS
        step_failed = any(e.get("event") == "step_failed"
                          for e in d.get("session") or [])
        if step_failed:
            prose_g = "FAIL" if prose == "PASS" else prose
            evid_g = "FAIL" if evid == "PASS" else evid
        else:
            prose_g, evid_g = prose, evid
        n += 1
        if prose_g == evid_g:
            agree += 1
        else:
            disagreements.append((os.path.basename(path), cid, truth,
                                  prose_g, evid_g, step_failed,
                                  answer[:120].replace("\n", " ")))

    print(f"regraded {n} archived sessions matched to pm_corpus "
          f"({unmatched} unmatched skipped)")
    print(f"agree {agree}/{n}; disagreements: {len(disagreements)}")
    for row in disagreements:
        path, cid, truth, prose_g, evid_g, sf, ans = row
        print(f"\n  {path} (corpus id {cid}, truth {truth}, "
              f"step_failed={sf})")
        print(f"    prose={prose_g}  evidence={evid_g}")
        print(f"    answer: {ans}")


if __name__ == "__main__":
    main()

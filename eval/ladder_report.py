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
# Description:  Aggregate system-ladder result JSONs into the comparison the
#               raw per-rung files do not give on their own: per-rung pass
#               rates with adjacent-rung deltas, a failure-mode breakdown
#               (pass / fail / timeout / error), and a per-question rung matrix
#               that localizes where a higher rung loses to a lower one. A
#               graded answer that is really a harness error (worker died) is
#               counted as 'error', not a wrong answer, so the pass rate is
#               over genuinely scored questions. Read-only over the result
#               JSONs; prints a report, writes diagnostics/ladder_report.md.
#               Human-read tooling.
#
#     python -m eval.ladder_report [--dir DIR] [--model M] [--corpus TAG]
# -----------------------------------------------------------------------------

# external imports
import glob
import json
import os
import sys
from collections import defaultdict

RUNG_ORDER = ["L0", "L1", "L2"]


def _status(row):
    """The row's status, but a graded answer that is actually a harness error
    (worker died, etc.) becomes its own bucket so infra failures are not
    counted as genuine wrong answers."""
    if str(row.get("answer", "")).startswith("(error:"):
        return "error"
    return row["status"]


def _load(dirpath, model=None, corpus=None):
    """Latest envelope per rung (optionally filtered by model / corpus)."""
    envs = {}
    for path in sorted(glob.glob(os.path.join(dirpath, "*.json"))):
        try:
            env = json.load(open(path))
        except Exception:
            continue
        if model and env.get("model") != model:
            continue
        if corpus and corpus not in str(env.get("corpus", "")):
            continue
        envs[env.get("rung")] = env   # sorted ascending, so latest wins
    return envs


def _tally(results):
    c = defaultdict(int)
    for r in results:
        c[_status(r)] += 1
    scored = c["PASS"] + c["FAIL"]
    rate = (c["PASS"] / scored) if scored else None
    return c, rate


def _per_question(results):
    """id -> (n_pass, n_reps) across reps."""
    by_id = defaultdict(list)
    for r in results:
        by_id[r["id"]].append(_status(r))
    return {q: (sum(s == "PASS" for s in sts), len(sts))
            for q, sts in by_id.items()}


def main():
    args = sys.argv[1:]

    def opt(flag, default=None):
        return args[args.index(flag) + 1] if flag in args else default

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dirpath = opt("--dir", os.path.join(here, "diagnostics", "system_ladder"))
    envs = _load(dirpath, opt("--model"), opt("--corpus"))
    rungs = [r for r in RUNG_ORDER if r in envs]
    if not rungs:
        print(f"no result JSONs in {dirpath}")
        sys.exit(1)

    lines = []

    def emit(s=""):
        lines.append(s)
        print(s)

    head = envs[rungs[0]]
    emit(f"# system ladder -- {head.get('model')} on "
         f"{os.path.basename(str(head.get('corpus')))}")
    emit(f"git {head.get('git_commit')} | cytools {head.get('cytools_version')}"
         f" | reps {head.get('reps')} | seed {head.get('example_seed')}")
    emit()

    # per-rung pass rate (over scored = pass + fail) and delta vs the rung
    # below it -- the delta is what attributes a layer's contribution
    emit("## rungs (pass% over scored = PASS+FAIL)")
    emit()
    emit("| rung | system | n | PASS | FAIL | TIMEOUT | error | pass% "
         "| delta |")
    emit("|---|---|---|---|---|---|---|---|---|")
    prev = None
    perq = {}
    for r in rungs:
        env = envs[r]
        perq[r] = _per_question(env["results"])
        c, rate = _tally(env["results"])
        pct = "n/a" if rate is None else f"{100 * rate:.0f}%"
        delta = ""
        if rate is not None and prev is not None:
            delta = f"{100 * (rate - prev):+.0f}pp"
        if rate is not None:
            prev = rate
        emit(f"| {r} | {str(env.get('system', ''))[:32]} "
             f"| {len(env['results'])} | {c['PASS']} | {c['FAIL']} "
             f"| {c['TIMEOUT']} | {c['error']} | {pct} | {delta} |")
    emit()

    # per-question x rung matrix (PASS/reps): shows which questions each rung
    # wins or loses, not just the aggregate
    kinds, qids = {}, set()
    for r in rungs:
        for row in envs[r]["results"]:
            kinds[row["id"]] = row.get("kind", "")
            qids.add(row["id"])
    emit("## per-question pass rate (PASS / reps)")
    emit()
    emit("| qid | kind | " + " | ".join(rungs) + " |")
    emit("|" + "---|" * (len(rungs) + 2))
    for q in sorted(qids):
        cells = []
        for r in rungs:
            np_, n_ = perq[r].get(q, (0, 0))
            cells.append(f"{np_}/{n_}" if n_ else "-")
        emit(f"| {q} | {str(kinds.get(q, ''))[:18]} | "
             + " | ".join(cells) + " |")
    emit()

    out_md = os.path.join(here, "diagnostics", "ladder_report.md")
    with open(out_md, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[written: {os.path.relpath(out_md, here)}]")


if __name__ == "__main__":
    main()

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
# Description:  Depth-first driver for the system ladder. system_ladder.py runs
#               one rung over the whole corpus per invocation (breadth-first by
#               rung); this instead takes one (randomly ordered) question at a
#               time and runs it through EVERY rung before moving on. So midway
#               you already have complete rung-by-rung comparisons for the
#               questions done so far -- easier to inspect than 'L0 done, rest
#               empty'. Writes the SAME per-rung result JSONs as system_ladder
#               (incrementally), so eval/ladder_report.py reads them unchanged.
#
#     python -m eval.ladder_dfs --model gpt-oss:20b [--reps 5]
#         [--corpus eval/ladder.jsonl] [--ids 1,3,5] [--seed 0]
#
# All functions here are human-read (developer tooling).
# -----------------------------------------------------------------------------

# external imports
import datetime
import json
import os
import random
import sys
import time

import eval._env  # noqa: F401  (env pins; must precede cytools_agent imports)

# local imports  (after the env pin)
from eval.grading import grade                                  # noqa: E402
from eval.system_ladder import (RUNGS, _run_isolated,           # noqa: E402
                                _git_commit, OUT_DIR,
                                DEFAULT_CORPUS)

RUNG_ORDER = ["L0", "L1", "L2", "L3", "L4"]
# L4 votes 3x internally, so its subprocess needs a bigger cap than L0-L3
TIMEOUT = {"L4": 900}
DEFAULT_TIMEOUT = 600


def _new_path(rung, model, corpus_tag, date):
    """Same naming as system_ladder; never overwrite a prior measurement."""
    p = os.path.join(OUT_DIR, f"{rung}__{model.replace(':', '-')}"
                              f"__{corpus_tag}__{date}.json")
    k = 2
    while os.path.exists(p):
        p = p.replace(".json", "").rstrip("_0123456789") + f"_{k}.json"
        k += 1
    return p


def main():
    args = sys.argv[1:]

    def opt(flag, default=None):
        return args[args.index(flag) + 1] if flag in args else default

    model = opt("--model", "gpt-oss:20b")
    corpus = opt("--corpus", DEFAULT_CORPUS)
    reps = int(opt("--reps", "5"))
    seed = int(opt("--seed", "0"))
    rows = {r["id"]: r for r in (json.loads(l) for l in open(corpus))}
    ids = ([int(x) for x in opt("--ids").split(",")] if "--ids" in args
           else sorted(rows))

    import cytools
    date = datetime.date.today().isoformat()
    corpus_tag = os.path.splitext(os.path.basename(corpus))[0]
    os.makedirs(OUT_DIR, exist_ok=True)

    # one envelope (and file) per rung; populated in depth-first order
    paths, envs = {}, {}
    for rung in RUNG_ORDER:
        paths[rung] = _new_path(rung, model, corpus_tag, date)
        envs[rung] = {
            "rung": rung, "system": RUNGS[rung][0], "model": model,
            "corpus": os.path.relpath(corpus), "ids": ids, "reps": reps,
            "timeout_s": TIMEOUT.get(rung, DEFAULT_TIMEOUT),
            "isolation": "subprocess-per-question", "traversal": "dfs",
            "git_commit": _git_commit(),
            "cytools_version": getattr(cytools, "version", "unknown"),
            "example_seed": os.environ.get("CYTOOLS_EXAMPLE_SEED", str(seed)),
            "date": date, "results": [],
        }

    # depth-first order: shuffle the (rep, id) pairs, then for each pair run
    # every rung before moving on -- so partial output spans all rungs
    plan = [(rep, qid) for rep in range(reps) for qid in ids]
    random.Random(seed).shuffle(plan)
    print(f"###### DFS ladder -- {model} on {corpus_tag}: {len(ids)} ids "
          f"x{reps} reps = {len(plan)} questions x {len(RUNG_ORDER)} rungs "
          f"######", flush=True)

    for n, (rep, qid) in enumerate(plan):
        q = rows[qid]
        print(f"\n-- question {n + 1}/{len(plan)}: id {qid} rep {rep} "
              f"({q.get('kind', '')}) truth={q['answer']} --", flush=True)
        for rung in RUNG_ORDER:
            to = TIMEOUT.get(rung, DEFAULT_TIMEOUT)
            t0 = time.monotonic()
            try:
                ans = _run_isolated(rung, model, q["question"], to)
            except Exception as e:
                ans = f"(error: {type(e).__name__}: {e})"
            dt = round(time.monotonic() - t0, 1)
            if q.get("kind") == "exploratory" or q.get("answer") is None:
                status = "RUBRIC"
            else:
                status = grade(ans, q["answer"])
            envs[rung]["results"].append(
                {"id": qid, "rep": rep, "kind": q["kind"], "status": status,
                 "secs": dt, "truth": q["answer"], "answer": str(ans)[:2000]})
            with open(paths[rung], "w") as f:
                json.dump(envs[rung], f, indent=2)
            print(f"   {rung}: {status} ({dt}s)  "
                  f"got: {str(ans)[:80]}", flush=True)

    print("\n###### DFS ladder done ######", flush=True)
    for rung in RUNG_ORDER:
        print(f"  {os.path.relpath(paths[rung])}", flush=True)


if __name__ == "__main__":
    main()

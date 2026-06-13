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
# Description:  Single-agent eval on the PM corpus (or any corpus file) -- the
#               architecture comparison arm: the SAME multi-step problems the
#               orchestrator is measured on, run through the plain Agent loop,
#               graded by the SAME grader. NOTE the agent talks to Ollama's
#               OpenAI-compatible endpoint, which cannot set num_ctx per
#               request -- start the server with OLLAMA_CONTEXT_LENGTH=16384
#               for a fair comparison against num_ctx-fixed orchestrator arms.
#
#     python -m eval.eval_single_pm qwen3:8b --ids 3,4,6,9 --reps 3
#         [--timeout 600] [--corpus eval/ladder.jsonl] [--out results.json]
#
# All functions here are human-read (developer tooling).
# -----------------------------------------------------------------------------

# external imports
import json
import os
import sys
import time

import eval._env  # noqa: F401  (env pins; must precede cytools_agent imports)

# local imports
from cytools_agent.tools import code as _code
from eval._harness import run
from eval.grading import grade

CORPUS = os.path.join(os.path.dirname(__file__), "pm_corpus.jsonl")


def main():
    args = sys.argv[1:]
    if not args:
        print("usage: python -m eval.eval_single_pm model --ids 1,2 "
              "[--reps N] [--timeout S] [--corpus path] [--out path]")
        sys.exit(1)
    model = args[0]
    raw = "--raw" in args          # L1 baseline: raw cytools, vanilla loop
    corpus = args[args.index("--corpus") + 1] if "--corpus" in args else CORPUS
    timeout = int(args[args.index("--timeout") + 1]) \
        if "--timeout" in args else 600
    reps = int(args[args.index("--reps") + 1]) if "--reps" in args else 1
    out_path = args[args.index("--out") + 1] if "--out" in args \
        else os.path.join("scratch", "eval_single_last.json")
    rows = {r["id"]: r for r in
            (json.loads(l) for l in open(corpus))}
    ids = ([int(x) for x in args[args.index("--ids") + 1].split(",")]
           if "--ids" in args else sorted(rows))

    print(f"###### single-agent eval ({model}) on {os.path.basename(corpus)} "
          f"ids {ids} x{reps} ######", flush=True)
    results = []
    for i in ids:
        for rep in range(reps):
            # fresh scratchpad per run, as run_session does for the orchestrator
            _code.reset_namespace()
            _code.reset_figures()
            t0 = time.monotonic()
            ans = run(model, rows[i]["question"], timeout, raw=raw)
            dt = round(time.monotonic() - t0, 1)
            status = grade(ans, rows[i]["answer"])
            results.append({"id": i, "rep": rep, "kind": rows[i]["kind"],
                            "status": status, "secs": dt, "answer": ans,
                            "truth": rows[i]["answer"]})
            print(f"\n[{i}.{rep}] {rows[i]['kind']}  {status}  ({dt}s)",
                  flush=True)
            print(f"    truth: {rows[i]['answer']}", flush=True)
            print(f"    got:   {str(ans)[:150]}", flush=True)
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    print("\n###### per-id (pass/scored) ######", flush=True)
    for i in ids:
        rs = [r for r in results if r["id"] == i]
        p = sum(r["status"] == "PASS" for r in rs)
        to = sum(r["status"] == "TIMEOUT" for r in rs)
        print(f"  id{i:<2} {rs[0]['kind']:28s} {p}/{len(rs) - to} scored pass"
              + (f", {to} timeout" if to else ""), flush=True)
    npass = sum(r["status"] == "PASS" for r in results)
    nto = sum(r["status"] == "TIMEOUT" for r in results)
    print(f"\n###### TOTAL {npass}/{len(results) - nto} scored pass, "
          f"{nto} timeout ######", flush=True)


if __name__ == "__main__":
    main()

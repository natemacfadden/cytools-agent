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
# Description:  Re-run the agent on SPECIFIC corpus questions (by id) to test
#               whether a fix helps. Runs each id reps times, reports
#               PASS/FAIL/TIMEOUT per rep and a summary.
#
# Usage:  python eval/rerun.py qwen3:8b 54,57,58 [reps] [timeout_s]
#         python eval/rerun.py qwen3:8b all        to run every id
# -----------------------------------------------------------------------------

# external imports
import json
import os
import sys

# local imports
from eval._harness import run
from eval.sample_eval import grade

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3:8b"
_idarg = sys.argv[2] if len(sys.argv) > 2 else ""
IDS = [] if _idarg in ("", "all") else [int(x) for x in _idarg.split(",")]
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
TIMEOUT = int(sys.argv[4]) if len(sys.argv) > 4 else 300


def main():
    rows = {r["id"]: r
            for r in (json.loads(line) for line in
                      open(os.path.join(os.path.dirname(__file__),
                                        "corpus.jsonl")))}
    ids = IDS or sorted(rows)
    print(f"###### {MODEL} on ids {ids} x{REPS} ######", flush=True)
    npass = nfail = ntimeout = 0
    for i in ids:
        r = rows[i]
        results = []
        for _ in range(REPS):
            ans = run(MODEL, r["question"], TIMEOUT)
            status = grade(ans, r["answer"])
            results.append((status, ans))
            npass += status == "PASS"
            nfail += status == "FAIL"
            ntimeout += status == "TIMEOUT"
        p = sum(s == "PASS" for s, _ in results)
        to = sum(s == "TIMEOUT" for s, _ in results)
        tag = f"{p}/{REPS}" + (f" ({to} timeout)" if to else "")
        print(f"\n[{i}] {r['kind']}  {tag}", flush=True)
        print(f"    Q: {r['question'][:95]}", flush=True)
        print(f"    truth: {r['answer']}", flush=True)
        for status, ans in results:
            print(f"    {status}: {ans[:110]}", flush=True)
    scored = npass + nfail
    print(f"\n###### {MODEL}: {npass}/{scored} scored correct "
          f"({ntimeout} timeout) ######", flush=True)


if __name__ == "__main__":
    main()

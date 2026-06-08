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
# Description:  Evaluation harness for the two-agent ORCHESTRATOR (PM+engineer)
#               on the PM corpus (eval/pm_corpus.jsonl). The orchestrator's
#               deliverable is usually a figure plus a stated result, so this
#               grades leniently with the SAME grader as the single-agent evals
#               (does the corpus summary value appear in the PM's final answer?)
#               AND, more importantly, extracts per-run DIAGNOSTICS from the
#               session log -- rounds, observations, step-limit hits, off-step
#               drift, and repeated-step loops -- which is what surfaces harness
#               friction.
#
#     python -m eval.eval_orch [--ids 4,6,9] [--model qwen3:4b] [--timeout 600]
#
# All functions here are human-read (developer tooling).
# -----------------------------------------------------------------------------

# external imports
import json
import os
import signal
import sys
import time

# local imports
from cytools_agent.orchestrator import run_session, read_evidence, read_session
from eval.grading import grade


CORPUS = os.path.join(os.path.dirname(__file__), "pm_corpus.jsonl")


def _rows():
    return [json.loads(line) for line in open(CORPUS)]


class _TimedOut(BaseException):
    pass


signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(_TimedOut()))


def _diagnostics(session, evidence):
    """Read the just-finished session + evidence and summarize HOW it went --
    the signals that reveal friction, not just pass/fail."""
    events = [e.get("event") for e in session]
    rounds = max((o.get("round", 0) for o in evidence), default=0)
    # a step that hit its limit without finishing
    step_failed = "step_failed" in events
    off_step = sum(e.get("event") == "off_step" for e in session)
    # repeated-step loop: the same ran_code run 3+ times in one round
    by_round = {}
    for o in evidence:
        by_round.setdefault(o.get("round"), []).append(
            (o.get("ran_code") or "").strip())
    max_repeat = max((codes.count(c) for codes in by_round.values()
                      for c in set(codes)), default=0)
    # error rate: observations whose output is a traceback
    n_err = sum("Traceback (most recent call last)" in
                str(o.get("received_output", "")) for o in evidence)
    return {"rounds": rounds, "n_obs": len(evidence),
            "step_failed": step_failed, "off_step": off_step,
            "max_step_repeat": max_repeat, "n_tracebacks": n_err}


def run_one(row, model, timeout):
    signal.alarm(timeout)
    t0 = time.monotonic()
    try:
        answer = run_session(row["question"], model=model, verbose=False)
        timed_out = False
    except _TimedOut:
        answer = "(timed out)"
        timed_out = True
    finally:
        signal.alarm(0)
    dt = time.monotonic() - t0
    diag = _diagnostics(read_session(), read_evidence())
    status = "TIMEOUT" if timed_out else grade(answer, row["answer"])
    # honest bar: a run that did not finish (step limit / walk stopped) is not a
    # PASS even if the truth digit happens to appear in the PM's prose
    if status == "PASS" and diag["step_failed"]:
        status = "FAIL"
    return {"id": row["id"], "kind": row["kind"], "status": status,
            "secs": round(dt, 1), "answer": answer, "truth": row["answer"],
            **diag}


def main():
    args = sys.argv[1:]
    model = args[args.index("--model") + 1] if "--model" in args else "qwen3:4b"
    timeout = int(args[args.index("--timeout") + 1]) \
        if "--timeout" in args else 600
    rows = {r["id"]: r for r in _rows()}
    ids = ([int(x) for x in args[args.index("--ids") + 1].split(",")]
           if "--ids" in args else sorted(rows))

    print(f"###### orchestrator eval ({model}) on ids {ids} ######", flush=True)
    results = []
    for i in ids:
        r = run_one(rows[i], model, timeout)
        results.append(r)
        flags = []
        if r["step_failed"]:
            flags.append("STEP-FAILED")
        if r["off_step"]:
            flags.append(f"off-step×{r['off_step']}")
        if r["max_step_repeat"] >= 3:
            flags.append(f"loop×{r['max_step_repeat']}")
        if r["n_tracebacks"]:
            flags.append(f"errs={r['n_tracebacks']}")
        print(f"\n[{i}] {r['kind']}  {r['status']}  "
              f"({r['secs']}s, {r['rounds']}r/{r['n_obs']}obs)"
              + ("  " + " ".join(flags) if flags else ""), flush=True)
        print(f"    truth: {r['truth']}", flush=True)
        print(f"    got:   {str(r['answer'])[:160]}", flush=True)

    npass = sum(r["status"] == "PASS" for r in results)
    nto = sum(r["status"] == "TIMEOUT" for r in results)
    print(f"\n###### {npass}/{len(results)} pass, {nto} timeout ######",
          flush=True)
    # machine-readable dump for later analysis
    with open(os.path.join("scratch", "eval_orch_last.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

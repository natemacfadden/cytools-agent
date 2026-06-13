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
# Description:  The ladder of systems -- the fairness design for demonstrating
#               this harness (see diagnostics/README.md). Five rungs hold the
#               model and questions fixed and vary one layer of the stack:
#
#                 L0  model alone (no tools)            -> the floor
#                 L1  raw cytools, vanilla loop         -> the baseline
#                 L2  curated tools, vanilla loop       -> + the tool layer
#                 L3  the orchestrator                  -> + the scaffolding
#                 L4  orchestrator + voting (3 runs)    -> + reliability
#
#               Results are written to diagnostics/system_ladder/ as
#               self-describing JSON (rung, model, corpus, commit, seed, date
#               + per-question results). One file per invocation; nothing is
#               overwritten.
#
#     python -m eval.system_ladder --rung L1 --model qwen3:8b \
#         [--ids 3,4,6,9] [--reps 3] [--corpus eval/pm_corpus.jsonl]
#         [--timeout 600]
#
# All functions here are human-read (developer tooling).
# -----------------------------------------------------------------------------

# external imports
import datetime
import json
import os
import subprocess
import sys
import time

import eval._env  # noqa: F401  (env pins; must precede cytools_agent imports)

# local imports
from eval.grading import grade

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "diagnostics", "system_ladder")
DEFAULT_CORPUS = os.path.join(os.path.dirname(__file__), "pm_corpus.jsonl")


def _run_L0(model, question, timeout):
    """Model alone: one chat call, no tools at all."""
    from eval._harness import client
    resp = client.chat.completions.create(
        model=model, timeout=timeout,
        messages=[{"role": "user", "content": question}])
    return (resp.choices[0].message.content or "").strip()


def _run_L1(model, question, timeout):
    from eval._harness import run
    return run(model, question, timeout, raw=True)


def _run_L2(model, question, timeout):
    from eval._harness import run
    return run(model, question, timeout, raw=False)


def _run_L3(model, question, timeout):
    from cytools_agent.orchestrator import run_session
    return run_session(question, model=model, verbose=False,
                       max_seconds=timeout)


def _run_L4(model, question, timeout):
    from cytools_agent.orchestrator import run_session_voted
    return run_session_voted(question, votes=3, model=model, verbose=False,
                             max_seconds=timeout)


RUNGS = {
    "L0": ("model alone, no tools", _run_L0),
    "L1": ("raw cytools in a plain REPL, vanilla agent loop", _run_L1),
    "L2": ("curated tool layer, vanilla agent loop", _run_L2),
    "L3": ("orchestrator (pipeline, schema decoding, checks)", _run_L3),
    "L4": ("orchestrator + voting x3", _run_L4),
}


def _run_isolated(rung, model, question, timeout):
    """Run one question in a fresh subprocess, so per-process state (the KS
    query budget, caches, sampled examples) cannot leak between questions.
    The worker prints the answer as JSON after a sentinel line."""
    p = subprocess.run(
        [sys.executable, "-m", "eval.system_ladder", "--worker",
         "--rung", rung, "--model", model, "--timeout", str(timeout)],
        input=question, capture_output=True, text=True, timeout=timeout + 120)
    marker = "###ANSWER###"
    if marker in p.stdout:
        return json.loads(p.stdout.rsplit(marker, 1)[1])["answer"]
    tail = (p.stderr or p.stdout).strip()[-200:]
    return f"(error: worker died: {tail})"


def _worker(rung, model, timeout):
    """Subprocess entry: read the question from stdin, run it, print the
    answer after a sentinel (stdout above the sentinel is the run's noise)."""
    question = sys.stdin.read()
    _, runner = RUNGS[rung]
    try:
        ans = runner(model, question, timeout)
    except Exception as e:
        ans = f"(error: {type(e).__name__}: {e})"
    print(f"\n###ANSWER###{json.dumps({'answer': str(ans)})}")


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def main():
    args = sys.argv[1:]

    def opt(flag, default=None):
        return args[args.index(flag) + 1] if flag in args else default

    rung = opt("--rung")
    if "--worker" in args:
        _worker(rung, opt("--model", "qwen3:8b"),
                int(opt("--timeout", "600")))
        return
    if rung not in RUNGS:
        print(f"usage: python -m eval.system_ladder --rung {{{'/'.join(RUNGS)}}} "
              f"--model M [--ids 3,4,6,9] [--reps 3] [--corpus path] "
              f"[--timeout 600]")
        sys.exit(1)
    model = opt("--model", "qwen3:8b")
    corpus = opt("--corpus", DEFAULT_CORPUS)
    timeout = int(opt("--timeout", "600"))
    reps = int(opt("--reps", "3"))
    rows = {r["id"]: r for r in (json.loads(l) for l in open(corpus))}
    ids = ([int(x) for x in opt("--ids").split(",")] if "--ids" in args
           else sorted(rows))

    desc, _ = RUNGS[rung]
    date = datetime.date.today().isoformat()
    corpus_tag = os.path.splitext(os.path.basename(corpus))[0]
    out_path = os.path.join(
        OUT_DIR, f"{rung}__{model.replace(':', '-').replace('/', '-')}"
                 f"__{corpus_tag}__{date}.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(out_path):   # never overwrite a measurement
        k = 2
        while os.path.exists(out_path.replace(".json", f"_{k}.json")):
            k += 1
        out_path = out_path.replace(".json", f"_{k}.json")

    import cytools
    envelope = {
        "rung": rung, "system": desc, "model": model,
        "corpus": os.path.relpath(corpus), "ids": ids, "reps": reps,
        "timeout_s": timeout, "isolation": "subprocess-per-question",
        "git_commit": _git_commit(),
        "cytools_version": getattr(cytools, "version", "unknown"),
        "example_seed": os.environ.get("CYTOOLS_EXAMPLE_SEED"),
        "date": date, "results": [],
    }
    print(f"###### system ladder {rung} ({desc}) -- {model} on "
          f"{corpus_tag} ids {ids} x{reps} ######", flush=True)
    for i in ids:
        for rep in range(reps):
            t0 = time.monotonic()
            try:
                ans = _run_isolated(rung, model, rows[i]["question"], timeout)
            except Exception as e:
                ans = f"(error: {type(e).__name__}: {e})"
            dt = round(time.monotonic() - t0, 1)
            # exploratory questions have no single truth value: they are
            # graded by the pre-registered rubric (right quantities, adequate
            # sample, conclusion matches own ledger rows), in a separate pass
            if rows[i].get("kind") == "exploratory" \
                    or rows[i].get("answer") is None:
                status = "RUBRIC"
            else:
                status = grade(ans, rows[i]["answer"])
            envelope["results"].append(
                {"id": i, "rep": rep, "kind": rows[i]["kind"],
                 "status": status, "secs": dt, "truth": rows[i]["answer"],
                 "answer": str(ans)[:2000]})
            print(f"[{i}.{rep}] {status} ({dt}s)  truth={rows[i]['answer']} "
                  f"got: {str(ans)[:90]}", flush=True)
            with open(out_path, "w") as f:
                json.dump(envelope, f, indent=2)

    npass = sum(r["status"] == "PASS" for r in envelope["results"])
    nto = sum(r["status"] == "TIMEOUT" for r in envelope["results"])
    print(f"\n###### {rung}: {npass}/{len(envelope['results']) - nto} scored "
          f"pass ({nto} timeout) -> {os.path.relpath(out_path)} ######",
          flush=True)


if __name__ == "__main__":
    main()

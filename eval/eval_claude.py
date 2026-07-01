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
# Description:  Corpus evaluation of Claude Code driving the cytools tools over
#               MCP (mcp_server.py). Runs `claude -p` headless once per question
#               and grades the answer with the SAME grader/corpus as eval.py --
#               so Claude-Code vs the local Ollama agent are directly
#               comparable. Defaults to the cheapest model (haiku).
#
#   Sampling (default): stratified random sample over corpus kinds.
#     python -m eval.eval_claude [k=12] [--model haiku]
#   Targeted: specific corpus ids, repeated reps times.
#     python -m eval.eval_claude --ids 54,57,58 [--reps 3] [--model haiku]
#
# Needs the `claude` CLI on PATH and the cytools-agent env (for the MCP server).
# Each question is a real Claude Code run (uses your Claude usage); total cost
# is reported at the end.
#
# All functions here are human-read (developer tooling).
# -----------------------------------------------------------------------------

# external imports
import json
import os
import subprocess
import sys

# local imports
from eval.emit import finalizing
from eval.grading import run_sample, run_targeted, TIMED_OUT

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_claude(question, model, timeout):
    """One headless Claude Code run against the cytools MCP server. Returns
    (answer_text, cost_usd); answer is TIMED_OUT on timeout."""
    cmd = [
        "claude", "-p", question,
        "--model", model,
        "--mcp-config", ".mcp.json", "--strict-mcp-config",
        "--allowedTools", "mcp__cytools__*",
        "--output-format", "json",
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, cwd=REPO_ROOT)
    except subprocess.TimeoutExpired:
        return TIMED_OUT, 0.0
    if p.returncode != 0:
        return f"(claude error: {p.stderr.strip()[-150:]})", 0.0
    try:
        d = json.loads(p.stdout)
    except json.JSONDecodeError:
        return f"(bad json: {p.stdout[:120]})", 0.0
    return (d.get("result") or ""), float(d.get("total_cost_usd") or 0.0)


USAGE = ("usage: python -m eval.eval_claude [k] [--model haiku]\n"
         "       python -m eval.eval_claude --ids 1,2,3 "
         "[--reps N] [--timeout S] [--model haiku]")


def main():
    args = sys.argv[1:]
    if args and args[0] in ("-h", "--help"):
        print(USAGE)
        sys.exit(0)

    model = "haiku"
    if "--model" in args:
        i = args.index("--model")
        model = args[i + 1]
        del args[i:i + 2]

    timeout = int(args[args.index("--timeout") + 1]) \
        if "--timeout" in args else 300
    spent = [0.0]

    def _run(q):
        ans, cost = run_claude(q, model, timeout)
        spent[0] += cost
        return ans

    # backstop off: the blind finalizer client is Ollama's and cannot reach a
    # Claude model, so trust Claude's (reliable) self-emission of the block.
    run_fn = finalizing(_run, model, backstop=False)
    header = f"claude:{model}"
    if "--ids" in args:
        ids = [int(x) for x in args[args.index("--ids") + 1].split(",")]
        reps = int(args[args.index("--reps") + 1]) if "--reps" in args else 3
        run_targeted(run_fn, header, ids, reps)
    else:
        k = int(args[0]) if args and not args[0].startswith("-") else 12
        run_sample(run_fn, header, k)

    print(f"###### total cost: ${spent[0]:.4f} ######", flush=True)


if __name__ == "__main__":
    main()

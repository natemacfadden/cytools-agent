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
# Description:  Corpus evaluation of the local Ollama agent. Shares the grader
#               and run loop with eval_claude.py via eval/grading.py; this file
#               only supplies the Ollama run function. Two modes:
#
#   Sampling (default): stratified random sample over corpus kinds.
#     python -m eval.eval qwen3:8b [k=12] [timeout=600]
#   Targeted: specific corpus ids, repeated reps times.
#     python -m eval.eval qwen3:8b --ids 54,57,58 [--reps 3] [--timeout 600]
#
# All functions here are human-read (developer tooling).
# -----------------------------------------------------------------------------

# external imports
import sys

# local imports
from eval._harness import run
from eval.grading import run_sample, run_targeted

USAGE = ("usage: python -m eval.eval model [k] [timeout]\n"
         "       python -m eval.eval model --ids 1,2,3 "
         "[--reps N] [--timeout S]")


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(USAGE)
        sys.exit(0 if args else 1)

    model = args[0]
    rest = args[1:]

    if "--ids" in rest:
        ids = [int(x) for x in rest[rest.index("--ids") + 1].split(",")]
        reps = int(rest[rest.index("--reps") + 1]) if "--reps" in rest else 3
        timeout = int(rest[rest.index("--timeout") + 1]) \
            if "--timeout" in rest else 600
        run_targeted(lambda q: run(model, q, timeout), model, ids, reps)
    else:
        k = int(rest[0]) if rest else 12
        timeout = int(rest[1]) if len(rest) > 1 else 600
        run_sample(lambda q: run(model, q, timeout), model, k)


if __name__ == "__main__":
    main()

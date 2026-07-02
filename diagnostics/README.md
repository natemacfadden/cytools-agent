# Diagnostics

Measured demonstration artifacts. Everything in here is produced by
`python -m eval.system_ladder` and is self-describing: each result file
carries its rung, model, corpus, ids, reps, git commit, example seed, and
date alongside the per-question results. Nothing in this tree is edited by
hand.

## system_ladder/

The fairness design: the contribution is a *system*, so each rung holds the
model and question set fixed and varies one layer of the stack. The deltas
between adjacent rungs attribute the win.

| Rung | System | The delta above the previous rung measures |
|---|---|---|
| L0 | model alone, no tools | floor: tools are necessary at all |
| L1 | raw cytools in a plain REPL, vanilla agent loop | (baseline -- the library as-published) |
| L2 | curated tool layer, same vanilla loop | the tool layer: ids, forgiveness, pointed errors, iteration/search tools, glossary, guards |

The orchestrator rungs (L3/L4) were removed on 2026-07-02 -- they did not beat
L2 and added bulk; see docs/DESIGN_LOG.md.

Run a rung (writes `system_ladder/<rung>__<model>__<corpus>__<date>.json`):

```sh
python -m eval.system_ladder --rung L1 --model qwen3:8b --ids 3,4,6,9 --reps 3
python -m eval.system_ladder --rung L2 --model qwen3:8b --corpus eval/corpus.jsonl --reps 5
```

Grading is identical across rungs: the typed `<final>` grader in eval/answer.py.
For the headline
demonstration, run all rungs on a held-out question set written by someone
other than the harness author, and report reps with both pass rates and
failure modes.

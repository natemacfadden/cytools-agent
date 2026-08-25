# Eval review — record (2026-08-25)

Background record for the review of the evaluation stack: what was found, what
was fixed, and the literature behind the recommendations. The live worklist of
what remains open is kept separately (see the review artifact); this file is
the archive so that context isn't lost when the worklist shrinks.

## Headline finding

Two polytopes carried ~45% of `corpus.jsonl`: the first h11=2 and first h11=3
polytopes appeared under three to four descriptions each ("first at h11=3",
"first *favorable* at h11=3", "first h11=3, h21=43" — one object). The same
Euler characteristic (−80) was asked five times under aliased kind labels.
Effective corpus size was roughly half its row count.

The second finding is structural: the human-written `heldout.jsonl` differs in
style from the AI-written corpora in kind, not degree — terse, ambiguous,
conversational, covering scaling behavior, feasibility, refusal, and unbounded
search. **The auto-graded corpora measure a different task than real usage.**

## What was fixed

### Pass 1 (commits `88e9fee..3354cf7`)

- **Nested-list grading.** `_entries` assumed entries nest at most one level,
  so a dense rank-3 intersection tensor raised inside the grader and returned
  FAIL — corpus id 32 could not be passed by anyone, failing against its own
  stored truth. Canonicalization is now recursive.
- **`corpus selfcheck`.** Every stored truth is graded against itself through
  the real grader, across all corpora, offline, as part of the test suite. A
  truth that fails can never be answered correctly by any model. All 143 pass.
  It caught a second defect on first run: a prose sentence in held-out question
  4's answer field, since blanked (held-out answers stay held out).
- **Infra-vs-capability.** `eval_claude` failure sentinels did not match the
  grader's `(error:` contract, so crashed headless runs scored as wrong
  answers.
- **Agent crash.** Malformed JSON in a structured tool call crashed `chat()`
  via an unguarded `json.loads`, losing the whole turn; it now returns to the
  model as a retryable tool result, matching the existing text-form recovery.
- **Statistics plumbing.** Wilson 95% intervals on every summary (correct at
  small n and extreme rates, where CLT intervals are not), and pass^k on
  targeted runs — reliability separated from a lucky average.
- **Corpus restructured.** Kind labels collapsed to one per concept so
  stratified sampling stops oversampling aliases; 14 duplicate facts removed to
  `corpus_removed.jsonl` (merged only where the object is provably identical
  *using facts the corpus itself states* and answers are equal, asserted at
  transform time); 25 ill-posed rows quarantined; paraphrase support added with
  a lint that refuses to treat a favorability cut as a rewording. Nothing
  deleted, ids stable.
- **Truth-stability fuzzer** (`eval/fuzz_truths.py`). Static mode classifies
  each row's reliance on unstated conventions (divisor basis, default
  triangulation, KS fetch order) and produced the quarantine list, discharging
  flags the corpus's own data resolves. Dynamic mode re-executes recipes under
  an alternative valid basis or a different FRST; its perturbations are proven
  to bite by an offline selftest.

### Pass 2 (commits `ac8bb4d..8b6e152`)

- **pm_corpus quarantine.** 7 of 10 graded a quantity the question never
  requests, in two flavors — three unrelated to the request (id 1 asks for a
  plot, grades an inequivalent-CY count; id 6 asks to *identify* a polytope,
  grades its automorphism order; id 8 asks for a distribution, grades the
  global minimum) and four genuinely open-ended (ids 0/2/3/5 grade an unnamed
  Pearson r to 3dp; for id 2 the truth is r = 0.103, so "no clear trend" is
  right and scores zero). `pm_corpus build` now rebuilds quarantined ids into
  the quarantine file so a rebuild cannot silently restore them.
- **Closed experiment removed.** `eval/nodes.py`, the `--nodes/--rnodes/
  --watch` arms, and the `CAPTURE_RESULT_NODES` hooks threaded through
  `run_python` — the grounding-pointer arc the DESIGN_LOG had already closed.
  `run_python` behavior verified unchanged.
- **`get_cy_info` caveat.** The "ONE CY from a default triangulation" warning
  was a `print`, reaching the model only inside `run_python`; it is now a field
  on the returned record.
- **Env-flag split-brain.** `CYTOOLS_MAP_TOOLS=""` read as enabled in
  `prompt.py` and disabled in `mapping.py`, so the prompt could advertise
  absent tools. Fixed and pinned by a test comparing both readers.
- **Replay header.** `save_history`'s hardcoded import list had gone stale by
  six tools; it now derives from the session's live tool set.
- **`eval_claude --no-code`.** Unsandboxed code execution stays on by default
  (withholding it would give this arm a different tool set than the local agent
  and confound the comparison) but is now announced and opt-outable.
- **`ms_corpus` provenance.** Its build script was never committed *and* its
  rows carry no reproduction code, so its answers cannot be verified even with
  database access. Annotated as a frozen smoke corpus rather than deleted.
- **Docs synced.** README tool table, sandbox warning scope, stale DESIGN_LOG
  branch note.

## Corpus state

| Corpus | n | Author | State |
|---|---|---|---|
| `corpus.jsonl` | 92 | agent, from notebooks | Active, 42 concepts. A regression suite, not a capability number, until the no-tools baseline runs. |
| `corpus_quarantined.jsonl` | 25 | agent | Out of scoring; awaiting pinning or a fuzz proof of stability. |
| `corpus_removed.jsonl` | 14 | agent | Duplicates, retained for provenance and paraphrase mining. |
| `pm_corpus.jsonl` | 3 | agent, during dev | 7 of the original 10 quarantined. |
| `pm_corpus_quarantined.jsonl` | 7 | agent | Questions that don't determine their graded answer. |
| `ms_corpus.jsonl` | 10 | script (never committed) | Frozen/unverified; smoke corpus only. |
| `ladder.jsonl` | 6 | hand/agent | Fit for its diagnostic purpose. |
| `heldout.jsonl` | 25 | **human** (frozen) | The load-bearing asset — and still never run. Never let an agent author or edit it. |

## Literature: what this project already gets right

- **Executable, code-derived ground truth** — the CORE-Bench discipline
  (arXiv:2409.11363), and the exact mitigation for the hand-extraction errors
  that dominated FrontierMath's audited 33–42% flaw rate (arXiv:2411.04872).
- **Typed tools over free codegen at 8B** — open models reference nonexistent
  packages in 21.7% of generations vs 5.2% commercial (arXiv:2406.10279). At
  ~15 tools, in-prompt listing is right; selection degrades past ~50
  (arXiv:2503.01763).
- **Harness-side iteration routes around a measured cliff** — BFCL v3 puts
  7–8B models at 70–90% single-turn but ~5–15% multi-turn/stateful, so
  `compute_for_each` is cliff-avoidance with published justification.
- **Glossary-RAG for long-tail knowledge** — Kandpal (arXiv:2211.08411) on
  long-tail facts resisting scale, DocPrompting (arXiv:2207.05987) on
  unseen-function gains; curated self-contained entries beat naive chunking
  (arXiv:2312.10997).
- **The ladder's design** — fixed model and questions, one layer varied per
  rung, is the paired-comparison structure that maximizes statistical power
  (Miller, arXiv:2411.00640).
- **A human-written held-out corpus at all** — GAIA (arXiv:2311.12983) and the
  holdout doctrine in "AI Agents That Matter" (arXiv:2407.01502).

**Novelty:** the survey found no prior LLM agent driving CYTools or any
Kreuzer–Skarke tooling. Nearest neighbor is CYTransformer (arXiv:2507.03732),
which is generative rather than agentic. The closest methodological twin is a
SageMath-augmented LLM agent eval (arXiv:2607.06820) showing +9.7pp from CAS
access — implying the headline number here should be the **tool-access delta**,
agent-with-CYTools vs model-alone.

## Literature: why the AI-written corpora failed the way they did

- **LLM-authored items skew easy and templated** — AutoBencher (arXiv:2407.08351)
  found naive generator-written questions solved at 97%+ by the generator's own
  family; diversity, not volume, is what matters (arXiv:2410.15226).
- **Uncurated synthetic items carry high defect rates** — Self-Instruct's own
  audit found 54% fully valid (arXiv:2212.10560); SWE-bench Verified filtered
  68% of sampled tasks; MMLU carries 6.5% label errors (arXiv:2406.04127);
  even FrontierMath's expert items ran 33–42% flawed. Human audit **with a
  published agreement rate** is the mandatory artifact, per Perez et al.
  (arXiv:2212.09251).
- **Contamination** — notebooks are public pretraining material
  (arXiv:2310.18018); small open models are the most memorization-prone family
  (GSM1k, arXiv:2405.00332); SWE-Bench Illusion (arXiv:2506.12286).
- **Small-n statistics** — CLT intervals are miscalibrated below a few hundred
  items (arXiv:2503.01747); rankings reshuffle under task subsets
  (arXiv:2107.07002); 8B scores swing several points on seed alone
  (arXiv:2504.07086); pass^k over pass@1 (τ-bench, arXiv:2406.12045).
- **Judge bias, if a rubric path is ever added** — position/verbosity/
  self-enhancement bias (arXiv:2306.05685), self-preference tied to
  self-recognition (arXiv:2404.13076), and ChemCrow's finding that an LLM judge
  could not replace expert assessment in a specialist domain (arXiv:2304.05376).
- **Safe generation pattern** — TaskBench back-instruct: sample the verified
  ground-truth call chain first, generate the question from it, grade tool
  selection / arguments / answer separately (arXiv:2311.18760).
- **RAG caveat** — Gorilla found BM25 retrieval scoring *below* zero-shot
  (17.0% vs 71.7%, arXiv:2305.15334), so live retrieval precision must be
  measured against no-retrieval and oracle-entry baselines; small models
  degrade most from irrelevant context (arXiv:2310.01558).

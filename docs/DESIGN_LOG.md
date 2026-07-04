# Verbose Design Log

The repo is small on purpose. I tried a bunch of different things that ended up not working, so I cut them to keep the repo trim. Unfortunately I didn't keep the best record of these experiments (other than my git logs) so I recreated some of the history here

**Original goal**: free, local AI agent that's competitive with frontier models for helping with CYTools computations.

No way I'm beating frontier models more generally. Both in their engineering (I'm just one man...) and in the resources needed to run them. These models seem to have poor knowledge of CYTools, so maybe I can help the user-base by building a *very specialized* harness for driving those computations.

In large part, this project was actually to learn how harnesses work.

# Early Stages (L2)

Much of the early stages were learning the basics like how to drive a model locally (Ollama), how to give these models access to tools, how these models learn to call tools, what an MCP server is etc. Just learning. ALmost immediately I was confronted with the engineering problem of writing a good system prompt and collection of tools.

I took much of this early guidance from what I've heard about frontier models: lean system prompt and minimal tools. Now I know that's a bit biased since these frontier models are general-purpose, but I still think that's the right decision. I justify this with my dumb framing of 'empathy' for the algorithm (please trust me I'm not a crackpot lol). I could give a super long and detailed system prompt, but that's somewhat analogous to asking a new grad student to read a textbook and then asking them to solve research problems. The information dump is hard to parse, contains a lot of noise (details on irrelevant computations to the current task), and difficult to recall.

In favor of this, I aimed to give a lean system prompt that'd be generally valuable to a model. This makes it kinda basic, but it's the information that each agent needs to know (you work on physics, you use CYTools via toolcalls and writing programs, etc.). The detailed information about string theory/CYTools still needs to be given, so that responsibility shifts to tools, error messages, and specialized injection of context.

## Tools

Again following guidance from frontier labs, I aimed to give a thin collection of multi-purpose tools. Instead of "compute the volume of this polytope" I favored "compute general information about the polytope". In this way, by wrapping up N tools into 1, I could reduce the tool surface N-fold. This assumes the model can read its desired data from the output, but that seems to work generally.

No matter how well I engineered the tools, the models still seemed to mess them up. These mistakes were very valuable though because, in nearly all cases, they were human parsable. From dumb things like wanting to read data 'h11' sometimes as `data.h11` and sometimes as `data['h11']` to struggling in switching between derived objects (our pipeline is Polytope->Triangulation->(ToricVariety->; kinda skippable...)CalabiYauManifold->physics). In every case, I still tried to show 'empathy': if the model's call path is clear and unambiguous to me, regardless of whether it is the 'right' path, I generalized the tools to support it.

Concretely this is scattered all through the tool layer. A fetched polytope id will answer `id['id']` or `id.ks_ind` by just handing itself back, so a model that treats an id like a record doesn't hit an opaque crash. Keyword arguments get matched to their real names when the model misremembers them. Ask a result dict for a field that isn't there and the error hands back the fields that *are* ("not a field; available keys: ...") instead of a bare KeyError it has to guess its way out of. The one place I don't bend is correctness: a value that's supposed to be an integer gets an integrality check and errors rather than silently rounding, since a quietly-wrong integer is worse than a failure.

When the model was doing something truly crazy, this also gave opportunity to provide very directed feedback. In contrast to the system prompt, which is served to all agents regardless of their goal, I have very specific implicit information about what the model is trying to do if it errors-out in any particular tool call. I can infer its goal, understand its error, and provide a very *directed* nudge. This is wher I provide much of the feedback that'd otherwise go in the system prompt, directed nudges when the model struggles. This maintains the connection to how humans best learn: a tutor correcting our mistakes and helping us get beyond them is immeasurably more valuable than a textbook.

## Context

Toolcalling is good, but there is still are a lot of weird technical terms that we use in our group 'divisor volume', 'favorable', 'NTFE', 'automorphisms'. Even if I was provided a gentle/guiding tutor, without knowing the context, I'd be dead in the water. The basic idea was to have an encyclopedia defining certain terms, explaining what they mean, and (in some cases) giving recipes on how to compute them. I did this via regex but now I understand that I was building a rudimentary form of RAG.

In practice it's a small glossary: each term gets a definition and, where it helps, a short recipe for computing it in CYTools, and I keyword-match the question to decide which entries to inject. The keyword matching is the crude part, and swapping it for real embedding-based retrieval is the obvious next step I haven't taken.

## End-Product for Early Stages

At the end of this early stage, I had a harness (toolcalling, context injection) that enabled the model to actually drive CYTools and even make some relevant plots. I tested it on some simple questions that I didn't know the answer to like "For a given N, what is the highest h11 of a polytope such that all of its 2-faces have <= N points" which are relevant to certain downstream applications. These are simple things, but still impressive for a model to be able to do

I'll later work on a 'ladder' of the harness for which this will be rung 'L2'. L1 is reserved for a bare-bones model.

# Orchestration mistake

Partially inspired by the separation of roles in work like https://arxiv.org/html/2605.22763v1, I wanted to try splitting the AI agent into two: a project manager (PM) and an engineer (later renamed to conductor and executor). I had the parallel motivation of wanting to increase trust in the system.

In more detail, I observed models floundering even on some semi-simple multi-stage requests, and they'd hallucinate/lie to give me an answer they thought I wanted. The goal was: if I separate tasks, then one agent (the PM) can just focus on splitting the tasks up into bite-sized chunks while the other agent (the engineer) can approach the smaller problems more easily.

Simultaneous with this split, I introduced a truth ledger. The agents needed to communicate and I was worried (from initial experiments) that the engineer would lie to the PM. If the PM just had to take the engineers word, then the results would be highly untrustworthy. The idea was, since the engineer primarily operates by writing/running code, we could just have this code itself be the message to the PM. Simply capture the written code and the outputs (and optionally some intent/motivation by the engineer) and pass this data to the PM. In this way, one makes it harder (not impossible) for the engineer to lie.

This orchestration made up L3/L4 of the ladder. L3 was the orchestration, L4 was that with a voting scheme. I spent a LOT of time trying to make this work, but (very surprisingly to me), L3&L4 were consistently worse than the simpler L2, despite my models being very simple (e.g., Qwen3:8b). (I did check this wasn't just my own bugs: a few of the early L3/L4 losses turned out to be artifacts, like a solver/environment conflict, a dropped final-answer emission, or a plan-gate that misread a real tool result as "nothing produced". I fixed those, which recovered the specific questions but didn't flip the overall result.) I even had cute engineering ideas in this like using AST to ensure the engineer isn't just faking code outputs (something I observed in initial stages), but ultimately I am abandoning this idea.

The AST check is a decent example of the whole trap. It catches the lazy fake, `print(15)` where 15 is just the number the model wanted, because a printed bare constant clearly wasn't computed. But a model that wants 15 badly enough writes `x = 15; print(x)`, or a plausible-looking computation rigged to land on 15, and now the ledger honestly records a real execution that's still a lie. I can force the ledger to reflect what actually ran; I can't force what ran to be the right computation. That's the wall the trust idea hit.

As I view it, the major issue is that I added so much beaurocracy that calls became slower and there were more locations to have errors. Even dumb things like "fetch 5 polytopes and tell me how many of them are N-favorable" (conceivable a 2-call answer) faild at the fetching stage. I really tried to force this approach by even adding hand-written deterministic pipelines like "iterate over X, applying Y to it, returning the REDUCTION" which helped reliability in some cases, but this was still spotty (in the models appropriately knowing to call them and giving the right arguments).

The clearest illustration of the tax: ask for Kahler moduli where the divisor volumes are something impossible (say all negative). Plain L2 calls the solver and gets the whole result back, including a loud "did not converge, don't trust these numbers" flag, and correctly says it can't be done. The orchestrated version maps the computation over the ids and pulls just the one volume field out of the result, so the "did not converge" flag never rides along with it, and it happily reports a garbage number. L2 wins here for a dumb reason: it sees the entire tool output, while the pipeline's neat one-field extraction is exactly what throws the warning away.

# Grading

A frequent issue in my debugging of harness failures was that many of them weren't actually failures in the model/harness, but in the grader. Obviously I can't manually grade all of the responses, so I made a regex method, but that was incredibly brittle. If it was trying to extract a number and there was any other number in the string, it was incredibly difficult to extract the right number. The nastiest version was the answer's own scaffolding leaking a matching digit: a polytope named `h11-3_h21-43` literally contains a 3 and a 43, so a question whose answer was 3 could score correct off the name alone, even when the model never actually computed anything. I built a lot of safeguards but those felt very local - just protecting against certain observed issues but not making the larger system more resilient.

I briefly toyed around with using a separately LLM as a judge (throwback to https://arxiv.org/html/2605.22763v1) which actually was much better at extracting the output, but this was a bit unsettling since I don't really trust the LLM. I converged, ultimately, upon enforcing a more-structured/typed output field that is used for grading. Concretely, each answer ends with a little tagged block, `<final>{"kind": "...", "value": ...}`, and the grader just compares that typed value to the truth (exact for integers, a tolerance band for floats, order-insensitive for lists), with no string-scraping at all. I mean this is used for the math HW of millions of students, so why shouldn't it be good here. That did help significantly.

One thing I made myself do before deleting the old regex grader: re-grade every stored answer with both graders and read every case where they disagreed. All of them were the typed grader being right: one where regex wrongly failed a correct answer, a few where it wrongly passed a wrong one off a coincidental digit. Since the switch only ever fixed mistakes, I felt fine ripping the regex path out.

# AI notes

I will continue a good amount of this development via Claude Code. The move to RAG via SentenceTransformers, the pruning/iteration that I've already been doing, etc. I am a bit forgetful in recording these notes as I go (it's another thing to track on top of the commits, planning next steps) so I'll direct the agents to record my notes. Below will be AI-written notes, formatted as `## (DATE) SUBJECT`.

For AI agents: these notes must be MINIMAL, backed by FACTS/EXPERIMENTS, limit notes to things that are DEMONSTRATED/PROVEN and indicate the level of EVIDENCE that you have for it. Be BRIEF and mimic the style above.

## (2026-07-02) Typed grader is now the only grader

- grade_typed (the `<final>` typed block) grades every eval path: the ladder, eval.py, eval_claude, eval_orch, eval_single_pm, all routed through one emission helper (eval/emit.py). [evidence: shipped, commit f3277c4; test_answer 11/11 pass]
- checked before deleting regex: re-graded stored answers with both graders, 4 disagreements out of 75 scored, all 4 the typed grader being right (1 regex false-negative, 3 false-positives off coincidental digits). [evidence: A/B on stored envelopes, n=75]
- also removed the LLM judge (eval/judge.py) and the evidence_grade regex cross-check, both dead/superseded. [evidence: no live callers]

## (2026-07-02) Why L3/L4 lose to L2 (root causes + branch fixes)

- id4 (favorable): orchestrator hits a KeyError on get_polytope_info(...)['favorable'] (real key is favorable_N) and commits "impossible" instead of recovering; happens at both L3 and L4. [evidence: session logs, repeated]
- id121 (impossible target): reports a volume from a non-converged solve because compute_for_each pulls ['cy_volume'] and drops the converged=False flag; plain L2 sees the whole result and abstains. [evidence: session logs/ledger]
- fixes parked on branch fix/l3-failures (unmerged): key-hint on missing dict fields, demote non-converged solver fields under `last_iterate`, finalizer maps an execution error to "none" not "impossible". [evidence: written + behavior unit-checked on _InfoDict; NOT validated to change pass rate]
- prelim full-corpus ladder run started then killed early: at n~10 (~2.5%), L2 >= L3 >= L4. [evidence: WEAK, not significant]

## (2026-07-02) Starting repo cleanup: removing the orchestration layer

- removing the two-agent orchestrator (cytools_agent/orchestrator/, the L3/L4 rungs, eval_orch.py): worse than the simpler L2, and adds bulk and latency. rationale and failure modes are logged above; the code goes, the lessons stay. [evidence: prior testing summarized above; the clean re-run was only n~10 but reproduced the loss modes]
- keeping the tool layer, context injection, typed grader, and the L0-L2 ladder.
- done: deleted cytools_agent/orchestrator/ (+ viewer.py, eval_orch.py), dropped L3/L4 from the ladder, swept stale refs, and repointed README + diagnostics to the L0-L2 story and the Agent loop. ~3.75k lines removed. [evidence: `import cytools_agent` ok, eval modules compile, test_answer 11/11, no code imports the orchestrator]
- still open: notebooks/demo.ipynb still uses the orchestrator (separate rework); branch fix/l3-failures (tool fixes) still unmerged.

## (2026-07-02) Demo notebook off the orchestrator

- notebooks/demo.ipynb: removed the 7 orchestrator cells (run_session, voting, OrchestratorChat, viewer); added the map/plot/search + glossary tools to the Agent's tool list and one "iterate and plot" agent.chat demo. [evidence: valid JSON, 14 cells, no orchestrator refs; the 14-tool wiring builds schemas cleanly in the env]
- not yet run: the new plot cell has no output; the notebook needs an end-to-end run to repopulate, and the qwen3:8b plot query should be eyeballed (small models were the weak spot for multi-step plots).

## (2026-07-02) Two small fixes: setup.sh restart, cleaner agent trace

- setup.sh: `systemctl enable --now` does not restart an already-running ollama, so re-running setup never applied the OLLAMA_CONTEXT_LENGTH=16384 drop-in -- the live process predated it and kept the ~4096 default, front-truncating the system prompt (model reverts to generic chatter). Changed to enable + restart. [evidence: drop-in dated 07-02 16:47, ollama process up since 07-01 17:00; the agent's truncation probe still fired]
- agent.py verbose logging (v>=2): print each tool call as one line `-> name(args)` instead of dumping the raw ChatCompletionMessage repr (which carried the full reasoning field), and put a blank line before the returned answer so the response is separated from the trace. [evidence: compiles, imports, formatter checked]

## (2026-07-02) Demo quirk: notebook agent was missing glossary injection

- the notebook built Agent() without message_hook, so the encyclopedia auto-injection (the README's knowledge pillar) was off in the demo -- the model hallucinated "NTFE = Normal Triangulations with Fixed Edges" (real: Not-2-face-equivalent). Added message_hook=glossary.glossary_context, matching eval/_harness.make_agent. [evidence: glossary_context now returns the real NTFE definition for that question]
- left as-is (plain-loop behavior, per the L2 notes above): the agent guesses get_cy_info height vectors instead of threading get_heights output, and sometimes explains instead of computing.

## (2026-07-02) Demo failure: confabulated favorability, and the keyword injector missed the fix

- on qwen3:8b: for a polytope with 2 NTFEs the agent returned only 1 CY volume, then invented a false justification ("only 1 NTFE is favorable", "favorability is per-triangulation", "1 valid, 1 invalid non-regular/star") and defended it across three corrections before recomputing both (33.604, 37.160). favorability is a property of the polytope, and NTFEs are FRST by construction, so both are valid. [evidence: demo cells fbd01ceb..1e43a5d3]
- contributing cause (not root): the keyword glossary injection never surfaced the `favorable` definition that would have countered it. `favorable` is in _SCAN_SKIP (an over-fire guard) and the token matcher misses "Favorability"; also glossary_context only scans the user message, so it cannot catch a term the model raises unprompted (which is where the confabulation started). a semantic retriever would likely have injected the definition on "Favorability is a property of the polytope". [evidence: glossary_context returned "" for that message; cy_glossary('favorability') resolves to the favorable entry]
- root cause is the weak model confabulating and defending. bumped the demo to qwen3:14b; cells fbd01ceb..1e43a5d3 presuppose the 8b error and likely need removing once 14b answers the CY-volume question in one turn. [evidence: not yet re-run on 14b]

## (2026-07-03) Recall benchmark for the glossary-injection A/B (before building RAG)

- eval/retrieval_bench.py: scores any `retrieve(message, k) -> {glossary keys}` on 128 labeled cases -- 15 hand probes, 105 real corpus questions labeled from their `kind` via KIND_TO_TERM, and 8 command/follow-up negatives. Reports recall on positives and false-fire on negatives. Built before the RAG so "does it beat keyword" is measured, not assumed.
- keyword baseline: recall 95/120 = 79% (exact 5/5 = 100%, paraphrase 2/10 = 20%, corpus 88/105 = 84%); false-fire 0/8 = 0%. Misses cluster on the `favorable` family (_SCAN_SKIP), h11/h21 -> hodge numbers, and paraphrases -- those are the RAG targets. The bar RAG must hold: 100% on exact, 0% over-fire (the negatives include h11-filter probes, where a semantic retriever risks over-firing). [evidence: python -m eval.retrieval_bench]
- caveat: labels are author-curated and a few are arguable (ntfe vs frst on "how many FRSTs"; h21 as a filter vs the asked quantity); real keyword recall is a bit above 79%. KIND_TO_TERM needs domain review before the number is trusted.

## (2026-07-03) Starting the dense-retrieval (RAG) experiment

- installed sentence-transformers 5.6.0 + torch 2.12.1+cu130 into cytools-agent (GPU build; retrieval will default to device="cpu", GPU kept for possible embedder fine-tuning later). numpy 2.4.6 / numba / cytools unaffected. added sentence-transformers to environment.yml. [evidence: imports + a CPU embed both OK; test_answer still 11/11]
- plan, measured on retrieval_bench.py: a dense retriever (bge-small-en-v1.5) behind the same retrieve(message, k) -> keys interface, A/B vs keyword; then calibrate a cosine threshold (the off-switch); then hybrid = keyword union dense-above-threshold. bar to beat/hold: keyword recall 79% (exact 100%, paraphrase 20%), false-fire 0%.
- early signal only, NOT a result: cos("how symmetric is the polytope?", "automorphism group order") = 0.634 with bge-small, i.e. the embedder does connect a paraphrase keyword misses. still unproven that dense/hybrid beats keyword without regressing exact-match or over-firing on the h11-filter negatives. [evidence: one cosine pair, not the bench]

## (2026-07-03) RAG result: hybrid wins, transformer alone does not

Three retrievers now share the retrieve(message, k) -> keys shape in retrieval_bench.py: regex (the string matcher, = glossary_context's existing selection), transformer (bge-small-en-v1.5 on CPU, cosine to each entry's "term: definition", keep entries with sim >= threshold, top-k), and hybrid (regex union transformer, regex first, capped at k). Threshold is a similarity bar: higher = stricter (fewer fire), not looser. Added a --sweep mode that encodes each query once and re-applies each cutoff to the cached scores.

Threshold sweep (transformer / hybrid), regex reference = recall 79%, paraphrase 20%, false-fire 0%:
- transformer alone: at any cutoff loose enough to hold recall (0.40-0.65) it false-fires 25-100%; the cutoffs that reach 0% false-fire (0.75+) crater recall to 48% then 18%. No good operating point on its own.
- hybrid: recall stays ~82-87% across 0.40-0.75 because regex is a floor; raising the bar only sheds the transformer's marginal guesses. 0.75 is the lowest bar with 0% false-fire.

Head-to-head at the calibrated 0.75 (all three at 0% false-fire, so apples-to-apples):
- regex:       exact 100%, paraphrase 20%, corpus 84%, overall 79%
- transformer: exact  80%, paraphrase 40%, corpus 48%, overall 48%
- hybrid:      exact 100%, paraphrase 50%, corpus 84%, overall 82%

Verdict: hybrid is strictly regex-or-better on every bucket -- it keeps regex's exact/corpus and adds a paraphrase lift (20 -> 50%) with no over-fire. The transformer's only real contribution is as a paraphrase booster bolted onto regex, not a replacement (alone it even misses an exact term). So retrieval helps, but modestly and only as hybrid. Set DENSE_THRESHOLD = 0.75 in the bench.

Caveats: 0.75 is calibrated on these 128 cases (only 10 paraphrases), so it is approximate, not a hard constant; a held-out set / more negatives would firm it up before wiring hybrid into the product glossary_context. The 0.70 vs 0.75 knob: 0.70 buys +3 recall and +10 paraphrase for 1 of 8 negatives firing (a false-fire here just injects one irrelevant definition, not a wrong answer). [evidence: python -m eval.retrieval_bench, python -m eval.retrieval_bench --sweep]

## (2026-07-04) Experiment A: answers as pointers to nodes (grounding the reported value)

Context: after removing the orchestration layer, nothing at runtime forces the final answer to be the number the code actually computed -- the model can transcribe or confabulate it (the id121 failure). Framed the fix as two obligations: A) prove the reported value equals what the code returned, B) prove the code is non-pathological. B is undecidable over arbitrary code (Rice's theorem), so it is punted; the tractable win is A, done not by verifying a claim but by removing the model from the value-reporting path.

Mechanism (minimal): the model leaves its answer in a run_python variable and points at it instead of retyping it:
    <final>{"kind": "int", "node": "answer"}</final>
The grader reads the real Python object out of the persistent run_python namespace (_code._NS) and grades THAT with the same deterministic check() as before. So the reported value cannot disagree with the computed one -- it is the computed one. Nodes = namespace variables (chosen over tool-call-ids: reads the genuine object, not a stringified result). eval/nodes.py holds the contract (NODE_FINAL_INSTRUCTION), the node-aware parser, a numpy->plain coercion, and resolve_and_grade (dangling pointer -> FAIL; impossible/none and a literal "value" still grade as-is; a reply with no block at all falls back to the same blind finalizer the value arm uses, so the A/B isolates the node mechanism, not the backstop). Wired as `--nodes` on eval_single_pm so before/after is one script.

Not addressed by A (honest scope): a wrong-but-consistent computation still passes (that is B), and bool/impossible answers are awkward to point at (accepted). Backstop asymmetry: only the no-block case backstops in both arms; when the model DOES commit, the value arm trusts its typed number and the node arm reads the variable -- that gap is exactly what we are measuring.

First attempt (inconclusive by design): before (value) vs after (nodes), qwen3:14b, single rep, 10 pm_corpus cases -> 1/10 both arms. The 3 flips were all run-to-run variance, not the mechanism: id4 FAIL->PASS because the *after* run happened to solve a case the *before* run gave up on; id9 PASS->FAIL because the after run computed a different (wrong) number and even ignored the node contract (typed a value). Two lessons: (1) independent before/after runs at default temperature conflate model variance with the mechanism -- a single rep measures noise; (2) pm_corpus is too hard (the model rarely computes right), so there is almost no computed-right-but-reported-wrong population for A to rescue. [evidence: scratch/A_before_pm.json, scratch/A_after_pm.json]

Corrected design (running): two independent arms, each in its natural mode (value arm types the number; node arm points) -- NOT one prompt asking for both, which biases the typing arm into copying the variable and hides real fabrication. Multiple trials per arm so temperature averages out. Run on a model-solvable subset (22 single-step corpus.jsonl cases stratified over int/float/list/bool), where the model succeeds often enough that a reporting error can actually surface as a flip. qwen3:14b, 3 reps/arm. Read A's effect as the pass-rate delta; per-case agreement says rescue vs mechanism-cost. [evidence to come: scratch/A_before_corpus.json, scratch/A_after_corpus.json]

Interim observation (value arm ~half done, 25/29 pass): every failure so far is a computation/capability error, not a reporting error -- id9 committed 3->0 (computed the wrong count and reported it faithfully), id13 returned "impossible" on all 3 reps (never produced an answer). None is "computed right, typed wrong," which is the only class A can fix. So this model does not fabricate on this corpus, and the before/after will be a near-wash.

Key conclusion (why the benchmark cannot distinguish A here, and why that is fine): A's measurable accuracy benefit is proportional to the fabrication rate, which is ~0 for a capable model on single-step questions. That makes A look like a weak-model crutch when measured as a pass-rate delta. But that is the wrong lens. A's real value is a STRUCTURAL GUARANTEE: the reported value cannot differ from the computed value because it IS the computed value (correctness-by-construction), which is the frontier-transferable framing (ground the answer in tool output) rather than a crutch. Decision: adopt A as a guarantee, not as an accuracy booster; a ~0 delta on this set is expected and acceptable. Where the delta WOULD show (and where A earns its keep as insurance): weaker models, long multi-step chains, and aggregations (the id121 compute_for_each flag-drop).

Next (deferred, not yet run): confirm the wash on this run, then rerun the same 22 cases with a weaker model (qwen3:4b or ministral-3b). If A's delta grows as capability drops, that confirms the reading -- A is a crutch as a booster but a cheap guarantee worth keeping regardless.

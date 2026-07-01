"""Local LLM judge for the system ladder.

Grades a rung's final answer against the known-correct value using a LOCAL
model (the same ollama-backed OpenAI-compatible client the rungs use). This is
the "option B" grader, but run locally rather than via a frontier API.

Protocol -- CLOSED BOOK. The judge sees only three things:
    (1) the QUESTION,
    (2) the known CORRECT ANSWER (ground truth from the corpus),
    (3) the CANDIDATE final-answer string produced by a rung.
It does NOT see the tool-call transcript or captured stdout, so it cannot
verify that a number was actually computed (vs hallucinated) -- it only decides
whether the candidate COMMITS the correct value as its answer, rather than
merely mentioning it incidentally. That distinction is the whole point: it is
where the regex heuristic fails (matching a digit inside a polytope name like
`h11-3_h21-43`, or inside a list the model itself disavows).

Returns a Verdict(status, reason, raw); status is PASS / FAIL, or ERROR if the
judge's output could not be parsed.
"""
import re
from collections import namedtuple

Verdict = namedtuple("Verdict", "status reason raw")

DEFAULT_JUDGE_MODEL = "qwen3-14b-16k"

# --- behavior-defining prompt (surface this for review before trusting it) ---
SYSTEM = (
    "You grade answers to math/computation questions about lattice polytopes. "
    "You are given the QUESTION, the known CORRECT ANSWER, and a CANDIDATE "
    "final answer produced by another system. Your only job is to decide "
    "whether the candidate commits the correct value AS its answer.\n"
    "Rules:\n"
    "- PASS only if the candidate clearly asserts the correct value as its "
    "answer to the question.\n"
    "- FAIL if it refuses, says it could not compute, gives a different value, "
    "or the correct value appears only incidentally -- for example inside the "
    "question's own wording or a polytope name (like 'h11-3_h21-43'), in "
    "intermediate or listed data the candidate does not endorse, or in a set "
    "of numbers it explicitly disavows (e.g. 'do not trust them').\n"
    "- Compare numbers only; ignore wording, units, and extra prose. Integers "
    "must match exactly. For a decimal correct answer, PASS if the candidate's "
    "committed value rounds to the correct answer's stated precision (correct "
    "1.46 accepts 1.46 or 1.4583).\n"
    "- Do NOT use outside knowledge to re-derive the answer. Trust the given "
    "CORRECT ANSWER as ground truth; you are only checking agreement.\n"
    "- Some candidates end with parenthetical orchestrator tags describing "
    "their OWN result; honor these over any raw printed numbers:\n"
    "  * '(self-consistency: N/M runs agreed on X)' means the committed "
    "answer is X. If X does not equal the correct answer, FAIL -- even if a "
    "printed list or vector elsewhere happens to contain the correct value.\n"
    "  * '(LOW CONFIDENCE: ... runs disagreed ...)' means the candidate is "
    "unsure; PASS only if its stated final value equals the correct answer.\n"
    "  * 'do not trust them' / 'treat with care' are generic cautions and do "
    "NOT by themselves force FAIL if the stated value is correct.\n"
    "Output exactly two lines and nothing else:\n"
    "REASON: <one short sentence>\n"
    "VERDICT: PASS   (or)   VERDICT: FAIL"
)


def judge_messages(question, truth, answer):
    user = (f"QUESTION:\n{question}\n\n"
            f"CORRECT ANSWER: {truth}\n\n"
            f"CANDIDATE FINAL ANSWER:\n{answer}\n\n"
            "Grade it now. /no_think")
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}]


_VERDICT = re.compile(r"VERDICT:\s*(PASS|FAIL)", re.I)
_REASON = re.compile(r"REASON:\s*(.+)", re.I)


def llm_judge(question, truth, answer, model=DEFAULT_JUDGE_MODEL, timeout=180):
    """Grade one candidate answer. Deterministic (temperature 0)."""
    from eval._harness import client
    resp = client.chat.completions.create(
        model=model, timeout=timeout, temperature=0,
        messages=judge_messages(question, truth, answer))
    raw = (resp.choices[0].message.content or "").strip()
    # a reasoning model may wrap its scratch work in <think>...</think>;
    # strip it so the parse keys on the actual verdict lines.
    body = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    hits = list(_VERDICT.finditer(body))
    status = hits[-1].group(1).upper() if hits else "ERROR"
    rm = _REASON.search(body)
    reason = rm.group(1).strip() if rm else ""
    return Verdict(status, reason, raw)

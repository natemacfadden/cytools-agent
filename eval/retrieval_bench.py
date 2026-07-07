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
# Description:  A/B benchmark for glossary context retrieval, so swapping the
#               retriever (keyword matcher -> embedding RAG) is measured, not
#               assumed. Scores any function
#
#                   retrieve(message, k) -> set[str]   # glossary keys to inject
#
#               against labeled messages on two axes that trade off:
#                 - recall on positives: did the relevant term get injected
#                   (keyword's weak spot is paraphrases/variants);
#                 - false-fire on negatives: did it inject on a message with no
#                   term to define (over-firing is the RAG risk; keyword guards
#                   it with _SCAN_SKIP).
#
#               Cases come from three places:
#                 - hand:   targeted exact/paraphrase/negative probes;
#                 - corpus: real corpus questions, labeled from each `kind` via
#                           KIND_TO_TERM;
#                 - negatives: realistic tool-driving commands / follow-ups.
#
#               Caveat: KIND_TO_TERM and hand labels are author-curated and want
#               domain-expert review (a few arguable, flagged inline).
#
#     python -m eval.retrieval_bench            # score every retriever
#     python -m eval.retrieval_bench --detail   # + per-case hits/misses
#     python -m eval.retrieval_bench --misses   # only the misses/false-fires
#
# All functions here are human-read (developer tooling).
# -----------------------------------------------------------------------------

import json
import os
import sys

from cytools_agent.tools import glossary as _g

_CORPUS = os.path.join(os.path.dirname(__file__), "corpus.jsonl")

# Targeted probes: (message, {expected keys}, bucket).
HAND_CASES = [
    # exact / synonym present: keyword's home turf
    ("How many NTFE triangulations does it have?", {"ntfe"}, "exact"),
    ("What is the second Chern class of the CY?", {"second chern class"}, "exact"),
    ("Compute the Gopakumar-Vafa invariants.", {"gopakumar-vafa invariants"}, "exact"),
    ("Give me the Mori cone rays.", {"mori cone"}, "exact"),
    ("What is the D3 tadpole charge?", {"d3 tadpole charge"}, "exact"),
    # paraphrase / variant / unlisted phrasing: the RAG target
    ("Favorability is a property of the polytope, not the triangulation", {"favorable"}, "paraphrase"),
    ("Is this geometry favorable with respect to the N lattice?", {"favorable"}, "paraphrase"),
    ("How symmetric is the polytope?", {"automorphisms"}, "paraphrase"),
    ("What is the volume of the Calabi-Yau at that point?", {"cy volume"}, "paraphrase"),
    ("How big is the space of Kahler parameters?", {"kahler moduli space"}, "paraphrase"),
    ("Which toric curve has the smallest volume?", {"toric curve volume"}, "paraphrase"),
    ("Give me the canonical hash that identifies this polytope.", {"content id"}, "paraphrase"),
    ("What is the sum of the genera of the two-dimensional faces?", {"2-face genus"}, "paraphrase"),
    ("Is the manifold's ambient space missing any divisors from the polytope?",
     {"non-toric divisors"}, "paraphrase"),
    ("Which pairs of divisors are never allowed to intersect?", {"stanley-reisner ideal"}, "paraphrase"),
]

# Realistic non-glossary messages: commands / follow-ups. Expected = nothing.
# (A few mention an h11/h21 filter, which should not trigger the hodge-numbers
# definition: a good over-fire probe for a semantic retriever.)
NEGATIVES = [
    "Fetch the first 10 polytopes at h11=3.",
    "Now do the same at h11=4.",
    "Save the session as a standalone script.",
    "Re-run that with a larger sample.",
    "Show me those ids again.",
    "Plot the results and save the figure.",
    "Thanks, that is exactly what I needed.",
    "Actually, use the third one instead.",
]

# corpus `kind` -> glossary key. Curated; skips kinds with no clean glossary term
# (vertex counts, no 'vertices' entry, and the plot_* tasks). A few are arguable
# and want review: triangulation_count/n_ntfe_frsts -> ntfe, facet/edge/2-face
# counts -> face count, mirror_h11 -> dual polytope.
KIND_TO_TERM = {
    "2-face-genera": "2-face genus",
    "automorphism-order": "automorphisms",
    "chi": "euler characteristic",
    "cy_phase_count": "distinct calabi-yaus",
    "cy_volume": "cy volume",
    "dimension": "polytope dimension",
    "distinct-cy-count": "distinct calabi-yaus",
    "divisor_volumes": "divisor volume",
    "dual-point-count": "dual point count",
    "euler-characteristic": "euler characteristic",
    "euler_characteristic": "euler characteristic",
    "face-count": "face count",
    "facet-count": "face count",
    "fanroots_volume_finder": "kahler parameters for target divisor volumes",
    "favorability": "favorable",
    "favorable": "favorable",
    "favorable-count": "favorable",
    "favorable_M": "favorable",
    "favorable_N": "favorable",
    "glsm-charge-matrix": "glsm charge matrix",
    "glsm_shape": "glsm charge matrix",
    "gv_invariant": "gopakumar-vafa invariants",
    "h11": "hodge numbers",
    "h21": "hodge numbers",
    "hodge-number": "hodge numbers",
    "hodge_numbers": "hodge numbers",
    "impossible_inverse": "kahler parameters for target divisor volumes",
    "impossible_obvious": "kahler parameters for target divisor volumes",
    "induced-2face-triangulation": "induced 2-face triangulation",
    "intersection_numbers": "triple intersection numbers",
    "is_trilayer": "trilayer",
    "kahler_cone": "kahler cone",
    "ks-count": "polytope count",
    "min_toric_curve_volume": "toric curve volume",
    "mirror_h11": "dual polytope",
    "mori_cone": "mori cone",
    "mori_cone_cap_rays": "mori cone cap",
    "n_2d_reflexive_subpolys": "2d reflexive subpolytopes",
    "n_2faces": "face count",
    "n_edges": "face count",
    "n_facets": "face count",
    "n_ntfe_frsts": "ntfe",
    "n_points": "lattice points",
    "n_points_interior_to_facets": "points interior to facets",
    "n_rigid_divisors": "rigid divisors",
    "ntfe_count": "ntfe",
    "point-count": "lattice points",
    "points-interior-to-facets": "points interior to facets",
    "polytope_points": "lattice points",
    "prime_toric_divisors": "prime toric divisors",
    "second_chern_class": "second chern class",
    "stretched_cone_tip": "stretched cone tip",
    "tadpole_Q0": "d3 tadpole charge",
    "total_ntfe_frsts": "ntfe",
    "triangulation_count": "ntfe",
    "trilayer": "trilayer",
    "trilayer-count": "trilayer",
    "trilayer_count": "trilayer",
    "triple-intersection-count": "triple intersection numbers",
    "triple-intersection-numbers": "triple intersection numbers",
}


def corpus_cases(max_per_kind=3):
    """Real questions from the corpus, labeled from `kind` via KIND_TO_TERM."""
    seen = {}
    out = []
    for line in open(_CORPUS):
        r = json.loads(line)
        term = KIND_TO_TERM.get(r.get("kind"))
        if not term:
            continue
        n = seen.get(r["kind"], 0)
        if n >= max_per_kind:
            continue
        seen[r["kind"]] = n + 1
        out.append((r["question"], {term}, "corpus"))
    return out


def all_cases():
    return (HAND_CASES
            + corpus_cases()
            + [(m, set(), "negative") for m in NEGATIVES])


# The three retrievers are the shipped ones (cytools_agent.tools.glossary), so
# this bench measures exactly what glossary_context uses: no drift between the
# validated numbers and production. glossary owns the model, threshold (0.75),
# and the keyword/dense/hybrid logic; here we just wrap them for scoring.
DENSE_THRESHOLD = _g.DENSE_THRESHOLD
_BGE_QUERY_PREFIX = _g._BGE_QUERY_PREFIX


def _ensure_index():
    _g._ensure_dense()


def keyword_retrieve(message, k=3):
    """Keyword layer (regex phrase match, longest first, capped at k)."""
    return set(_g._regex_keys(message, k))


def dense_retrieve(message, k=3, threshold=DENSE_THRESHOLD):
    """Semantic layer (bge-small cosine >= threshold, top-k)."""
    return _g._dense_keys(message, k, threshold)


def hybrid_retrieve(message, k=3, threshold=DENSE_THRESHOLD):
    """Both: keyword first, then dense above threshold, capped at k."""
    return set(_g.retrieve_keys(message, k, threshold))


def _validate_labels(cases):
    keys = set(_g._GLOSSARY)
    bad = {t for _, exp, _ in cases for t in exp if t not in keys}
    if bad:
        raise SystemExit(f"labeled term(s) not in glossary: {sorted(bad)}")


def evaluate(retrieve, cases=None, k=3, show="none"):
    """Score a retriever. show in {none, all, misses}. Returns a dict of rates."""
    cases = cases or all_cases()
    _validate_labels(cases)
    buckets = ("exact", "paraphrase", "corpus")
    tally = {b: [0, 0] for b in buckets}          # bucket -> [hits, total]
    neg_fires = neg_total = 0
    for msg, exp, bucket in cases:
        got = retrieve(msg, k)
        if bucket == "negative":
            neg_total += 1
            fired = bool(got)
            neg_fires += fired
            if show == "all" or (show == "misses" and fired):
                print(f"  [{'FIRE' if fired else 'ok  '}] negative   got={sorted(got)}  |  {msg}")
            continue
        hit = exp <= got
        tally[bucket][0] += hit
        tally[bucket][1] += 1
        if show == "all" or (show == "misses" and not hit):
            print(f"  [{'ok  ' if hit else 'MISS'}] {bucket:10} exp={sorted(exp)} got={sorted(got)}  |  {msg[:80]}")

    ph = sum(t[0] for t in tally.values())
    pt = sum(t[1] for t in tally.values())
    print(f"\n  recall (positives injected): {ph}/{pt} = {ph / pt:.0%}")
    for b in buckets:
        h, n = tally[b]
        if n:
            print(f"    {b:11}: {h}/{n} = {h / n:.0%}")
    print(f"  false-fire (negatives):      {neg_fires}/{neg_total} = "
          f"{neg_fires / neg_total:.0%}")
    return {"recall": ph / pt, "false_fire": neg_fires / neg_total,
            "by_bucket": {b: tally[b] for b in buckets}}


# retrievers under test; all share the retrieve(message, k) shape.
# "regex" = the string/keyword matcher; "transformer" = the embedding retriever.
RETRIEVERS = {"regex": keyword_retrieve,
              "transformer": dense_retrieve,
              "hybrid": hybrid_retrieve}


def sweep(thresholds=None, k=3):
    """Sweep the transformer's cosine cutoff and show, at each threshold, the
    recall / paraphrase-recall / false-fire for the transformer alone and for
    hybrid (regex union transformer). Encodes every query once, then re-applies
    each threshold to the cached similarity scores, so the sweep is cheap."""
    _ensure_index()
    thresholds = thresholds or [round(0.40 + 0.05 * i, 2) for i in range(9)]  # .40-.80
    cases = all_cases()
    _validate_labels(cases)

    precomp = []                       # (expected, bucket, ranked[(sim,key)], regex_set)
    for msg, exp, bucket in cases:
        q = _g._embed_model.encode([_BGE_QUERY_PREFIX + msg],
                                   normalize_embeddings=True)[0]
        ranked = sorted(zip((_g._entry_vecs @ q).tolist(), _g._entry_keys),
                        reverse=True)
        precomp.append((exp, bucket, ranked, keyword_retrieve(msg, k)))

    def score(make_set):
        pos = hits = para_h = para_t = fires = neg = 0
        for exp, bucket, ranked, kw in precomp:
            got = make_set(ranked, kw)
            if bucket == "negative":
                neg += 1
                fires += bool(got)
                continue
            pos += 1
            hits += (exp <= got)
            if bucket == "paraphrase":
                para_t += 1
                para_h += (exp <= got)
        return hits / pos, para_h / para_t, fires / neg

    reg = score(lambda ranked, kw: kw)
    print(f"regex-only reference:  recall {reg[0]:.0%}   paraphrase {reg[1]:.0%}   "
          f"false-fire {reg[2]:.0%}\n")
    print("          transformer alone           hybrid (regex + transformer)")
    print("thresh  recall  paraphr  false-fire   recall  paraphr  false-fire")
    for t in thresholds:
        tr = score(lambda ranked, kw, t=t: {key for sim, key in ranked[:k] if sim >= t})

        def hybrid_set(ranked, kw, t=t):
            dn = [key for sim, key in ranked[:k] if sim >= t]
            return set((list(kw) + [x for x in dn if x not in kw])[:k])
        hy = score(hybrid_set)
        print(f"{t:>5.2f}   {tr[0]:>5.0%}   {tr[1]:>5.0%}    {tr[2]:>6.0%}     "
              f"{hy[0]:>5.0%}   {hy[1]:>5.0%}    {hy[2]:>6.0%}")


def main():
    if "--sweep" in sys.argv:
        sweep()
        return
    show = "all" if "--detail" in sys.argv else ("misses" if "--misses" in sys.argv else "none")
    for name, fn in RETRIEVERS.items():
        print(f"###### retriever: {name}  ({len(all_cases())} cases) ######")
        evaluate(fn, show=show)
        print()


if __name__ == "__main__":
    main()

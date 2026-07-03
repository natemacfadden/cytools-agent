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
# Description:  A/B benchmark for glossary context RETRIEVAL, so a change of
#               retriever (the keyword matcher -> an embedding RAG) is measured,
#               not assumed. Scores any function
#
#                   retrieve(message, k) -> set[str]   # glossary keys to inject
#
#               against labeled messages, on two axes that trade off:
#                 - recall on POSITIVES: did the relevant term get injected
#                   (keyword's weak spot is paraphrases/variants);
#                 - false-fire on NEGATIVES: did it inject on a message with no
#                   term to define (over-firing is the RAG risk; keyword guards
#                   it with _SCAN_SKIP).
#
#               Cases come from three places:
#                 - hand:   targeted exact/paraphrase/negative probes;
#                 - corpus: REAL questions from eval/corpus.jsonl, labeled from
#                           each question's `kind` via KIND_TO_TERM (realistic
#                           phrasing, grounded label);
#                 - negatives: realistic tool-driving commands / follow-ups.
#
#               Caveat: KIND_TO_TERM and the hand labels are author-curated and
#               need domain-expert review (a few are arguable, flagged inline).
#               Label quality is what makes the numbers trustworthy.
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
    # exact / synonym present -- keyword's home turf
    ("How many NTFE triangulations does it have?", {"ntfe"}, "exact"),
    ("What is the second Chern class of the CY?", {"second chern class"}, "exact"),
    ("Compute the Gopakumar-Vafa invariants.", {"gopakumar-vafa invariants"}, "exact"),
    ("Give me the Mori cone rays.", {"mori cone"}, "exact"),
    ("What is the D3 tadpole charge?", {"d3 tadpole charge"}, "exact"),
    # paraphrase / variant / unlisted phrasing -- the RAG target
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
# (A few mention an h11/h21 FILTER, which should NOT trigger the hodge-numbers
# definition -- a good over-fire probe for a semantic retriever.)
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
# (vertex counts -- no 'vertices' entry -- and the plot_* tasks). A few are
# arguable and want review: triangulation_count/n_ntfe_frsts -> ntfe,
# facet/edge/2-face counts -> face count, mirror_h11 -> dual polytope.
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


def keyword_retrieve(message, k=3):
    """Current production retriever: glossary_context's selection -- matched
    keys, longest phrase first, capped at k."""
    return set(sorted(_g._matched_keys(message), key=lambda s: -len(s))[:k])


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


# retrievers under test -- add the RAG one here (same signature) when it exists
RETRIEVERS = {"keyword": keyword_retrieve}


def main():
    show = "all" if "--detail" in sys.argv else ("misses" if "--misses" in sys.argv else "none")
    for name, fn in RETRIEVERS.items():
        print(f"###### retriever: {name}  ({len(all_cases())} cases) ######")
        evaluate(fn, show=show)
        print()


if __name__ == "__main__":
    main()

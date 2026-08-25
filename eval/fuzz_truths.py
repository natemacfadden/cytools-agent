# =============================================================================
#    Copyright (C) 2026  Nate MacFadden for the Liam McAllister Group
#    (GPL-3.0-or-later; see eval/answer.py header.)
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  Truth-stability audit for corpus.jsonl. A stored truth is only
#               a truth if the question determines it; this tool finds rows
#               whose answers silently depend on a convention the question
#               does not pin down.
#
#   static (default, no cytools/network): classify every row's risk --
#     basis:         the answer is coordinates in an unpinned divisor basis
#     triangulation: the answer depends on which FRST .triangulate() returns,
#                    discharged when the corpus itself records that the
#                    polytope has a unique FRST/NTFE
#     fetch_order:   the polytope identity is "first at h11=X", which assumes
#                    the KS server returns a stable order (h21-pinned rows are
#                    weaker versions of the same assumption)
#     python -m eval.fuzz_truths
#
#   dynamic (needs cytools + KS access or a warm cache): re-execute flagged
#   rows under perturbations -- a permuted divisor basis, and a different FRST
#   than the default -- and report which stored answers actually change.
#     python -m eval.fuzz_truths --dynamic [--ids 31,32]
#
#   selftest (needs cytools, no network): prove the perturbation patches bite,
#   using a polytope built from eval/case1_verts.json vertices.
#     python -m eval.fuzz_truths --selftest
#
# All functions here are human-read (developer tooling).
# -----------------------------------------------------------------------------

# external imports
import json
import os
import re
import subprocess
import sys

PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(__file__))
CORPUS = os.path.join(os.path.dirname(__file__), "corpus.jsonl")

# kinds whose value is unchanged across the FRSTs of one polytope (fine star
# triangulations of a reflexive polytope give one deformation class per NTFE,
# and these are invariants of the polytope/deformation class, not the phase)
TRI_INVARIANT = {
    "euler_characteristic", "hodge_numbers", "h11", "h21", "h11_M",
    "mirror_h11", "tadpole_Q0", "n_frsts", "n_ntfes", "n_distinct_cys",
    "favorable_N", "favorable_M", "ks-count", "glsm_shape",
    "prime_toric_divisors",
}

# kinds whose answer is written in divisor-basis coordinates (vectors/tensors)
BASIS_COORDS = {"second_chern_class", "intersection_numbers", "mori_cone",
                "kahler_cone", "gv_invariant"}
# kinds whose answer is a count that still depends on the basis choice
BASIS_COUNTS = {"n_intersection_numbers"}


def _rows():
    """Corpus rows PLUS quarantined ill-posed rows: the quarantine file is
    this tool's primary worklist (rows return to the corpus once their
    question pins the convention or a dynamic run proves stability)."""
    rows = [json.loads(line) for line in open(CORPUS)]
    qpath = os.path.join(os.path.dirname(__file__),
                         "corpus_quarantined.jsonl")
    if os.path.exists(qpath):
        rows += [json.loads(line) for line in open(qpath)]
    return rows


def _identity(code):
    """(h11, h21, favorable, index) of the polytope a row's code fetches, with
    the corpus-proven equivalences folded in (first-at-h11 is favorable for
    h11 in {2,3,4}: ids 35/4/27; first h11=2 has h21=29: id 90; first h11=3
    has h21=43: id 89)."""
    m = re.search(r"fetch_polytopes\(([^)]*)\)", code)
    if not m:
        return None
    a = m.group(1)
    g = lambda pat: (re.search(pat, a) or [None, None])[1]
    h11 = g(r"h11\s*=\s*(\d+)") or g(r"^\s*\d+\s*,\s*(\d+)")
    h21 = g(r"h21\s*=\s*(\d+)")
    fav = bool(re.search(r"favorable\s*=\s*True", a))
    idx = g(r"ids\[(\d+)\]") if "ids[" in code else None
    key = (int(h11) if h11 else None, int(h21) if h21 else None, fav,
           int(idx) if idx else 0 if "ids[0]" in code or "[0]" in code else None)
    # proven-identical first polytopes collapse to one canonical identity
    canon = {(2, 29, False, 0): (2, None, False, 0),
             (2, None, True, 0): (2, None, False, 0),
             (3, 43, False, 0): (3, None, False, 0),
             (3, None, True, 0): (3, None, False, 0),
             (4, None, True, 0): (4, None, False, 0)}
    return canon.get(key, key)


def _uses_default_triangulation(code):
    """True when the row builds a CY/triangulation without pinning heights:
    p.triangulate() with no argument, or the first entry of get_heights."""
    if re.search(r"\.triangulate\(\s*\)", code):
        return True
    if re.search(r"get_heights\([^)]*\)\s*(\[0\]|\['heights'\]\[0\])", code):
        return True
    return False


def classify():
    """Static risk classification. Returns {id: [flags]}."""
    rows = _rows()

    # identities the corpus itself proves have a unique FRST or unique NTFE,
    # which discharges the triangulation flag for CY-level quantities
    unique_tri = set()
    for r in rows:
        if r["kind"] in ("n_frsts", "n_ntfes") and r["answer"] == 1:
            ident = _identity(r["code"])
            if ident:
                unique_tri.add(ident)

    flags = {}
    for r in rows:
        f = []
        code, kind = r["code"], r["kind"]
        ident = _identity(code)

        if kind in BASIS_COORDS and isinstance(r["answer"], list):
            f.append("basis: answer is divisor-basis coordinates; the "
                     "question does not pin the basis")
        if kind in BASIS_COUNTS or (
                "in_basis=True" in code and isinstance(r["answer"], int)
                and kind in BASIS_COORDS | BASIS_COUNTS):
            f.append("basis-count: count of nonzero in-basis entries depends "
                     "on the basis choice")

        if (_uses_default_triangulation(code) and kind not in TRI_INVARIANT):
            if ident in unique_tri:
                pass   # corpus proves this polytope has a unique FRST/NTFE
            else:
                f.append("triangulation: quantity depends on which FRST "
                         ".triangulate()/get_heights[0] returns; heights not "
                         "pinned and uniqueness not established")

        if "fetch_polytopes" in code:
            if ident and ident[1] is None and not ident[2]:
                f.append("fetch-order: identity is positional in the "
                         "unfiltered KS stream")
        if f:
            flags[r["id"]] = f
    return flags


def report():
    rows = {r["id"]: r for r in _rows()}
    flags = classify()
    by_cat = {}
    for i, fs in sorted(flags.items()):
        for f in fs:
            by_cat.setdefault(f.split(":")[0], []).append(i)
    print(f"{len(flags)}/{len(rows)} rows carry stability flags\n")
    for cat, ids in sorted(by_cat.items()):
        print(f"  {cat:14s} {len(ids):3d} rows: {ids}")
    print()
    for i, fs in sorted(flags.items()):
        r = rows[i]
        print(f"[{i}] {r['kind']}")
        print(f"    Q: {r['question'][:90]}")
        for f in fs:
            print(f"    - {f}")
    return flags


# dynamic: re-execute flagged rows under perturbations
# ----------------------------------------------------
# Each perturbation is a preamble prepended to the row's own code in a fresh
# interpreter; if the printed output no longer matches the stored answer, the
# answer depended on the perturbed convention.
PERTURB = {
    # {n} is the alternative-basis index: the n-th accepted single-index swap.
    # One swap can land on a linearly-equivalent divisor and leave the answer
    # unchanged, so the dynamic runner tries several n per row.
    "basis": r"""
import cytools
from cytools.triangulation import Triangulation as _Tri
_SWAP_N = {n}
_orig_get_cy = _Tri.get_cy
def _patched_get_cy(self, *a, **k):
    # CYTools canonicalizes index-list bases (a permutation is a no-op), so a
    # genuinely different basis means a different index SET: deterministically
    # swap one default-basis index for a non-basis index that still forms a
    # valid basis, taking the _SWAP_N-th accepted swap. If fewer exist, the
    # CY is returned unchanged.
    cy = _orig_get_cy(self, *a, **k)
    b = list(map(int, cy.divisor_basis()))
    npts = len(cy.second_chern_class(in_basis=False))
    seen = 0
    for e in (i for i in range(npts) if i not in b):
        for j in range(len(b)):
            cand = sorted(b[:j] + b[j+1:] + [e])
            try:
                cy.set_divisor_basis(cand)
            except Exception:
                continue
            seen += 1
            if seen == _SWAP_N:
                return cy
    cy.set_divisor_basis(b)
    return cy
_Tri.get_cy = _patched_get_cy
""",
    "triangulation": r"""
import itertools
import cytools
_orig_triangulate = cytools.Polytope.triangulate  # Polytope is top-level
def _patched_triangulate(self, *a, **k):
    if a or k.get("heights") is not None:
        return _orig_triangulate(self, *a, **k)
    # lazily take the second enumerated FRST (never enumerate them all: the
    # count can be astronomical); with only one, this reduces to the default
    tris = list(itertools.islice(
        self.all_triangulations(only_fine=True, only_regular=True,
                                only_star=True), 2))
    return tris[-1] if tris else _orig_triangulate(self, *a, **k)
cytools.Polytope.triangulate = _patched_triangulate
""",
}


def dynamic(only_ids=None):
    from eval.corpus import _eq
    rows = {r["id"]: r for r in _rows()}
    flags = classify()
    results = []
    for i, fs in sorted(flags.items()):
        if only_ids and i not in only_ids:
            continue
        r = rows[i]
        cats = {f.split(":")[0].replace("-count", "") for f in fs}
        for cat in sorted(cats & set(PERTURB)):
            # one swap can hit a linearly-equivalent divisor; try a few
            variants = ([PERTURB["basis"].format(n=n) for n in (1, 2, 3)]
                        if cat == "basis" else [PERTURB[cat]])
            verdict, detail = "STABLE", ""
            for pre in variants:
                code = pre + "\n" + r["code"]
                try:
                    p = subprocess.run([PY, "-c", code], capture_output=True,
                                       text=True, cwd=ROOT, timeout=600)
                except subprocess.TimeoutExpired:
                    verdict = "TIMEOUT"
                    break
                if p.returncode != 0:
                    verdict, detail = "ERROR", p.stderr.strip()[-80:]
                    break
                if not _eq(p.stdout, r["answer"]):
                    verdict = "SENSITIVE"
                    detail = (f"got {p.stdout.strip()[:60]} "
                              f"want {r['answer']}")
                    break
            results.append((i, cat, verdict, detail))
    for i, cat, verdict, detail in results:
        print(f"[{i}] {cat:14s} {verdict}  {detail}")
    n_sens = sum(v == "SENSITIVE" for _, _, v, _ in results)
    print(f"\n{n_sens} SENSITIVE / {len(results)} perturbation runs")
    return results


# selftest fixtures, both offline (vertices inline, no KS fetch):
#  - basis bite needs inequivalent divisor classes entering the basis under a
#    single index swap; highly symmetric small polytopes (quintic, products,
#    X18) have equivalent alternates that leave c2 unchanged, so this uses a
#    generic h11=20 KS polytope (from eval/case1_verts.json). Everything it
#    needs is fast at h11=20 -- what is NOT fast is FRST enumeration, so the
#    triangulation patch is exercised on X18 in P(1,1,1,6,9) instead.
_BASIS_FIXTURE = [[1, 0, 0, 0], [-1, -2, 2, 0], [0, 0, 1, 0],
                  [-7, -2, -4, -2], [-6, -2, -3, 0], [0, 0, 0, 1],
                  [0, 1, 0, 0]]
_TRI_FIXTURE = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                [-1, -1, -6, -9]]


def selftest():
    """Offline proof that the perturbation preambles actually bite: some
    alternative divisor basis must change c2 coordinates while chi stays
    fixed; the triangulation patch must return a valid fine star
    triangulation and preserve chi. (Its ability to pick a genuinely
    different FRST is exercised by dynamic runs on corpus polytopes with
    recorded multi-FRST counts.)"""
    prog_tmpl = "import json\nimport cytools\n{patch}\n" \
        "p = cytools.Polytope({verts!r})\n" \
        "t = p.triangulate()\ncy = t.get_cy()\n" \
        "print(json.dumps({{'c2': cy.second_chern_class(in_basis=True)" \
        ".tolist(), 'chi': int(cy.chi()), 'fine': bool(t.is_fine()), " \
        "'star': bool(t.is_star())}}))"

    def run(patch, verts):
        prog = prog_tmpl.format(patch=patch, verts=verts)
        p = subprocess.run([PY, "-c", prog], capture_output=True, text=True,
                           cwd=ROOT, timeout=600)
        assert p.returncode == 0, p.stderr[-400:]
        return json.loads(p.stdout)

    base = run("", _BASIS_FIXTURE)
    alts = [run(PERTURB["basis"].format(n=n), _BASIS_FIXTURE)
            for n in (1, 2, 3)]
    assert all(a["chi"] == base["chi"] for a in alts), \
        "chi must be basis-independent"
    assert any(a["c2"] != base["c2"] for a in alts), \
        "no alternative basis changed c2 coords"
    tbase = run("", _TRI_FIXTURE)
    tri = run(PERTURB["triangulation"], _TRI_FIXTURE)
    assert tri["fine"] and tri["star"], "triangulation patch not fine+star"
    assert tri["chi"] == tbase["chi"], "chi must be triangulation-independent"
    print("selftest ok: an alternative divisor basis changes c2 coords and "
          "preserves chi; triangulation patch yields a fine star "
          "triangulation and preserves chi")


def main():
    args = sys.argv[1:]
    if "--selftest" in args:
        selftest()
    elif "--dynamic" in args:
        ids = None
        if "--ids" in args:
            ids = {int(x) for x in args[args.index("--ids") + 1].split(",")}
        dynamic(ids)
    else:
        report()


if __name__ == "__main__":
    main()

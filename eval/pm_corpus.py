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
# Description:  The multi-step corpus: big research problems, each spanning many
#               polytopes, a per-item computation, and an aggregate/plot
#               deliverable, so it exercises planning + decomposition (not a
#               single tool call).
#
#               Unlike eval/corpus.jsonl (single-fact questions), the deliverable
#               here is usually a figure plus an analysis, so there is no single
#               "the answer". Each problem carries a reference `code` that prints
#               one checkable summary statistic (a correlation, a count, a mean),
#               a deterministic anchor a stronger model can verify the run
#               against. The question text stays general (it doesn't dictate the
#               method or the exact summary); the code is just one valid solution.
#
#   build:  run each problem's reference code, record the printed summary as its
#           answer, and (re)write eval/pm_corpus.jsonl.
#               python -m eval.pm_corpus build [--timeout S] [--ids 0,3,5]
#   verify: re-run each entry's code and confirm it still prints the answer.
#               python -m eval.pm_corpus verify
#
# All functions here are human-read (developer tooling).
# -----------------------------------------------------------------------------

# external imports
import json
import os
import subprocess
import sys

from eval.corpus import _eq   # shared standalone-output comparison

PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(__file__))
CORPUS = os.path.join(os.path.dirname(__file__), "pm_corpus.jsonl")

_PRELUDE = ("import warnings; warnings.filterwarnings('ignore')\n"
            "import numpy as np\n"
            "from cytools_agent.tools import polytope, triangulation, cy\n")


# the problems: (question, kind, body). `body` is appended to _PRELUDE and must
# print exactly the checkable summary. The first two are the user's seeds.
# ----------------------------------------------------------------------------
PROBLEMS = [
    # --- seed 1 -------------------------------------------------------------
    ("For the first 150 favorable h11=4 polytopes (first NTFE triangulation "
     "each), how does the largest toric curve volume at the stretched Kahler "
     "cone tip track the D3 tadpole? Scatter the results and flag any outliers.",
     "curve-volume-vs-tadpole",
     "ids = polytope.fetch_polytopes(limit=150, h11=4, favorable=True)\n"
     "x, y = [], []\n"
     "for ks in ids:\n"
     "    info = polytope.get_polytope_info(ks)\n"
     "    h = triangulation.get_heights(ks)['heights'][0]\n"
     "    cv = cy.get_cy_info(ks, h, t='tip', cone='toric')['curve_volumes']\n"
     "    x.append((2 + info['h11'] + info['h21']) / 2)\n"
     "    y.append(max(cv))\n"
     "print(round(float(np.corrcoef(x, y)[0, 1]), 3))"),

    # --- seed 2 -------------------------------------------------------------
    ("Fetch a polytope at h11=5 and at each h21 (many polytopes of the form "
     "(h11,h21)=(5,X)). For each, construct the inequivalent CYs. Make a "
     "scatter plot of their CY volume at the tip of the stretched Kahler cone "
     "vs h21 across all of these polytopes.",
     "cy-volume-vs-h21",
     "ids = [polytope.fetch_polytopes(1, 5, h21)[0]\n"
     "       for h21 in polytope.ks_stats(5)['h21_values']]\n"
     "n = sum(len(triangulation.get_heights(ks)['heights']) for ks in ids)\n"
     "print(n)"),

    # --- 8 more in the same spirit -----------------------------------------
    ("Across the first 80 favorable polytopes at h11=3, how does the "
     "Calabi-Yau volume at the stretched Kahler cone tip (first NTFE "
     "triangulation) vary with h21? Make a scatter plot and report the trend.",
     "cyvol-vs-h21-trend",
     "ids = polytope.fetch_polytopes(limit=80, h11=3, favorable=True)\n"
     "x, y = [], []\n"
     "for ks in ids:\n"
     "    info = polytope.get_polytope_info(ks)\n"
     "    h = triangulation.get_heights(ks)['heights'][0]\n"
     "    v = cy.get_cy_info(ks, h, t='tip', cone='toric')['cy_volume']\n"
     "    x.append(info['h21']); y.append(v)\n"
     "print(round(float(np.corrcoef(x, y)[0, 1]), 3))"),

    ("For the first 50 polytopes at h11=4, compare the largest entry of the "
     "Calabi-Yau's second Chern class against its Euler characteristic (first "
     "NTFE triangulation). Scatter them and note any relationship.",
     "chern-vs-euler",
     "ids = polytope.fetch_polytopes(limit=50, h11=4)\n"
     "x, y = [], []\n"
     "for ks in ids:\n"
     "    h = triangulation.get_heights(ks)['heights'][0]\n"
     "    info = cy.get_cy_info(ks, h)\n"
     "    x.append(max(info['second_chern_class']))\n"
     "    y.append(info['euler_characteristic'])\n"
     "print(round(float(np.corrcoef(x, y)[0, 1]), 3))"),

    ("Among the first 100 polytopes at h11=3, plot the distribution of the "
     "number of NTFE triangulations, and report the mean.",
     "ntfe-count-distribution",
     "ids = polytope.fetch_polytopes(limit=100, h11=3)\n"
     "counts = [triangulation.get_heights(ks)['shape'][0] for ks in ids]\n"
     "print(round(float(np.mean(counts)), 3))"),

    ("For the first 40 favorable polytopes at h11=5, examine how the largest "
     "divisor volume at the stretched Kahler cone tip (first NTFE) relates to "
     "the D3 tadpole charge. Scatter and flag outliers.",
     "divvol-vs-tadpole",
     "ids = polytope.fetch_polytopes(limit=40, h11=5, favorable=True)\n"
     "x, y = [], []\n"
     "for ks in ids:\n"
     "    info = polytope.get_polytope_info(ks)\n"
     "    h = triangulation.get_heights(ks)['heights'][0]\n"
     "    dv = cy.get_cy_info(ks, h, t='tip', cone='toric')['divisor_volumes']\n"
     "    x.append((2 + info['h11'] + info['h21']) / 2)\n"
     "    y.append(max(dv))\n"
     "print(round(float(np.corrcoef(x, y)[0, 1]), 3))"),

    ("Across the first 60 polytopes at h11=4, how does the automorphism group "
     "order relate to the number of lattice points of the polytope? Make a "
     "scatter plot and identify the most symmetric polytope.",
     "automorphism-vs-points",
     "ids = polytope.fetch_polytopes(limit=60, h11=4)\n"
     "orders = [polytope.get_polytope_info(ks)['automorphism_order']\n"
     "          for ks in ids]\n"
     "print(int(max(orders)))"),

    ("For the first polytope at each h11 from 2 through 7, compute the "
     "Calabi-Yau volume at the stretched Kahler cone tip (first NTFE) and plot "
     "it against h11.",
     "cyvol-vs-h11",
     "vols = []\n"
     "for h11 in range(2, 8):\n"
     "    ks = polytope.fetch_polytopes(1, h11)[0]\n"
     "    h = triangulation.get_heights(ks)['heights'][0]\n"
     "    vols.append(round(cy.get_cy_info(ks, h, t='tip',\n"
     "                                     cone='toric')['cy_volume'], 2))\n"
     "print(vols)"),

    ("For the first 50 polytopes at h11=4, compute for each the minimum toric "
     "curve volume at the stretched Kahler cone tip over all its NTFE "
     "triangulations, and plot the distribution of these minima.",
     "min-curve-volume-distribution",
     "ids = polytope.fetch_polytopes(limit=50, h11=4)\n"
     "mins = []\n"
     "for ks in ids:\n"
     "    hs = triangulation.get_heights(ks)['heights']\n"
     "    res = cy.get_cy_info(ks, hs, t='tip', cone='toric')\n"
     "    mins.append(min(min(r['curve_volumes']) for r in res))\n"
     "print(round(float(min(mins)), 3))"),

    ("Among the first 100 polytopes at h11=4, plot the distribution of the "
     "maximum 2-face genus, and report how many have a maximum 2-face genus "
     "of zero.",
     "max-genus-distribution",
     "ids = polytope.fetch_polytopes(limit=100, h11=4)\n"
     "maxg = [max(polytope.get_polytope_info(ks)['genera_2face'])\n"
     "        for ks in ids]\n"
     "print(int(sum(1 for g in maxg if g == 0)))"),
]


def _code(body):
    return _PRELUDE + body


def _run(code, timeout):
    return subprocess.run([PY, "-c", code], capture_output=True, text=True,
                          cwd=ROOT, timeout=timeout)


def build(timeout=1200, only=None):
    """Run each problem's reference code, capture its printed summary as the
    answer, and write pm_corpus.jsonl. `only` is an optional id whitelist."""
    rows = []
    for i, (q, kind, body) in enumerate(PROBLEMS):
        code = _code(body)
        ans = None
        if only is None or i in only:
            try:
                p = _run(code, timeout)
                if p.returncode == 0:
                    out = p.stdout.strip()
                    try:
                        ans = json.loads(out) if out.startswith("[") else \
                            float(out) if "." in out else int(out)
                    except ValueError:
                        ans = out
                    print(f"[{i}] {kind}: {ans}", flush=True)
                else:
                    print(f"[{i}] {kind}: ERROR {p.stderr.strip()[-120:]}",
                          flush=True)
            except subprocess.TimeoutExpired:
                print(f"[{i}] {kind}: TIMEOUT (>{timeout}s)", flush=True)
        rows.append({"id": i, "question": q, "answer": ans, "kind": kind,
                     "code": code, "source": "hand-authored", "agent": "coordinator"})
    with open(CORPUS, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} problems -> {CORPUS}")


def verify(timeout=1200):
    """Re-run each entry's code and confirm it reproduces the stored answer."""
    rows = [json.loads(line) for line in open(CORPUS)]
    ok = bad = skip = 0
    for r in rows:
        if r["answer"] is None:
            skip += 1
            continue
        p = _run(r["code"], timeout)
        if p.returncode == 0 and _eq(p.stdout, r["answer"]):
            ok += 1
        else:
            bad += 1
            print(f"  MISMATCH [{r['id']}] {r['kind']}: got "
                  f"{p.stdout.strip()[:40]!r} want {r['answer']}")
    print(f"verified {ok}, mismatch {bad}, skipped (no answer) {skip} "
          f"/ {len(rows)}")


def main():
    args = sys.argv[1:]
    cmd = args[0] if args else ""
    timeout = int(args[args.index("--timeout") + 1]) \
        if "--timeout" in args else 1200
    only = {int(x) for x in args[args.index("--ids") + 1].split(",")} \
        if "--ids" in args else None
    if cmd == "build":
        build(timeout, only)
    elif cmd == "verify":
        verify(timeout)
    else:
        print("usage: python -m eval.pm_corpus build|verify "
              "[--timeout S] [--ids 0,3,5]")
        sys.exit(0 if cmd in ("-h", "--help") else 1)


if __name__ == "__main__":
    main()

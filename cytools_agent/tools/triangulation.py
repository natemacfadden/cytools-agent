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
# Description:  Triangulation tools for AI agents. Focuses on generation of
#               triangulations by heights and NTFE triangulations.
# -----------------------------------------------------------------------------

# external imports
import cytools

# local imports
from cytools_agent.tools import costs
from cytools_agent.tools.polytope import get_polytope, _InfoDict, _h11_of
from cytools_agent.tools.cy import _as_heights
from cytools_agent.tools._synonyms import forgive_kwargs

# human-read
_GNN_OK = None


def _gnn_available() -> bool:
    """Whether the optional dualgnn package (GNN sampler) is importable.
    Cached after the first check since importing it loads PyTorch."""
    global _GNN_OK
    if _GNN_OK is None:
        try:
            import dualgnn  # noqa: F401
            _GNN_OK = True
        except Exception:
            _GNN_OK = False
    return _GNN_OK


_GNN_NOTE = ("near-uniform GNN sample of NTFE triangulations (dualGNN, "
             "arXiv:2605.27770) -- a sample, NOT the full census; tested "
             "envelope h11 <~ 128")


# human-read
def _triangulation_difficulty(poly: cytools.Polytope,
                              kind: str = "NTFE") -> tuple[float, str]:
    """
    Estimate how hard enumerating this polytope's triangulations is, from its
    number of lattice points. Returns a difficulty in [0, 1] and a short
    message.

    NTFE bands recalibrated 2026-06 against cytools' fast NTFE code (measured:
    <=0.1s through 25 points, >120s at 37). Point count only loosely predicts
    NTFE cost -- the count of NTFEs drives it, and that varies wildly mid-band
    (an h11=15 polytope had 11k NTFEs / 3s while an h11=20 had 81 / 0.1s) --
    so the mid bands are conservative. FRST enumeration was NOT sped up and
    keeps the old, stricter bands (measured: 26s already at 13 points).
    """
    n = len(poly.points())
    if kind == "FRST":
        bands = [(9, 0.4, "easy"), (12, 0.5, "doable"), (14, 0.9, "tough")]
    else:
        bands = [(17, 0, "easy"), (21, 0.1, "doable"), (25, 0.5, "tough"),
                 (30, 0.9, "VERY tough")]
    for cap, difficulty, msg in bands:
        if n <= cap:
            return difficulty, msg
    return 1, "too large"

# human-read
class _HeightsDict(dict):
    """The get_heights result: {"shape", "heights"} that ALSO answers integer
    indexing/iteration like the underlying list -- result[0] is the first
    height vector (the model's unambiguous intent; older corpus code too), and
    a missing string key gets a pointed error naming the real keys."""

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return self["heights"][key]
        return super().__getitem__(key)

    def __missing__(self, key):
        raise KeyError(
            f"'{key}' is not a field of the get_heights result. It has "
            f"'shape' ([n_triangulations, n_points]) and 'heights' (the list "
            f"of height vectors); result[i] also gives heights[i].")


# human-read
def _shaped(heights: list[list[float]]) -> dict:
    """Wrap a list of height vectors with its shape, so callers read the count
    from `shape[0]` instead of counting the vectors."""
    return _HeightsDict({
        "shape": [len(heights), len(heights[0]) if heights else 0],
        "heights": heights,
    })

# model-read
@forgive_kwargs
def get_heights(ks_ind: str, n: int | None = None, kind: str = "NTFE",
                effort: float = 0.5,
                seed: int | None = None, sampler: str = "auto") -> dict:
    """
    Triangulations of a polytope, as a dict {"shape": [n_triangulations,
    n_points], "heights": [...]}. "shape"[0] is HOW MANY triangulations there
    are; "heights" is the list of height vectors (one per triangulation), so
    heights[i] selects the i-th triangulation (e.g. for get_cy_info).

    Modes:
    - n given: a random sample of up to `n` triangulations. Works at any size.
      The `sampler` decides how: "gnn" is a near-uniform sample of NTFE
      triangulations (dualGNN, arXiv:2605.27770; needs the optional dualgnn
      package); "fast" draws heights around the Delaunay heights (quick but
      NOT a fair sample); "auto" (default) picks "gnn" for large polytopes
      when available, else "fast". A GNN result carries a "note" field
      stating it is a sample, not the census.
    - n omitted, kind="NTFE" (default): ALL inequivalent triangulations - FRSTs
      modulo equivalent restrictions to 2D faces. This is what distinguishes
      Calabi-Yaus, and is usually few.
    - n omitted, kind="FRST": ALL fine regular star triangulations. Many more
      than NTFE, and blows up much faster with size, so it is guarded more
      strictly (refuses at smaller polytopes).

    The exhaustive modes blow up with size, so they are guarded by `effort`
    (raise when harder than allowed; FRST more aggressively). When exhaustive
    enumeration is infeasible, sample instead (sampler="gnn" if fairness
    matters).

    Parameters
    ----------
    ks_ind : str
        The id of a polytope, of the form "h11-X_h21-Y_ind-Z".
    n : int, optional
        Sample size for the random sampling mode. Omit for an exhaustive mode.
    kind : str, optional
        For the exhaustive mode: "NTFE" (default) or "FRST".
    effort : float, optional
        How hard to try. >0 easiest only, >0.1 moderate, >0.5 tough, >0.9 very
        tough, >1 any case.
    seed : int, optional
        Random seed for the sampling (n given) mode.
    sampler : str, optional
        For the sampling mode: "auto" (default), "gnn", or "fast".

    Returns
    -------
    dict
        {"shape": [n_triangulations, n_points], "heights": list of height
        vectors}. Read the count from shape[0].
    """
    p = get_polytope(ks_ind)

    if n is not None:
        if sampler not in ("auto", "gnn", "fast"):
            raise ValueError(
                f"sampler must be 'auto', 'gnn', or 'fast', got {sampler!r}")
        if sampler == "gnn" and not _gnn_available():
            raise ValueError(
                "sampler='gnn' needs the optional dualgnn package "
                "(pip install dualgnn); it is not importable here. "
                "Use sampler='fast' instead.")
        if sampler == "auto":
            # prefer the fair sampler exactly where exhaustive enumeration
            # refuses, so big-polytope questions get sample-based answers
            # with honest near-uniform statistics
            sampler = ("gnn" if _gnn_available()
                       and _triangulation_difficulty(p)[0] >= 1 else "fast")
        if sampler == "gnn":
            with costs.timed("get_heights_gnn", h11=_h11_of(ks_ind),
                             n_points=len(p.points())):
                hts = p.random_triangulations_gnn(
                    N=n, make_star=True, as_heights=True, seed=seed)
            hts = [[float(x) for x in h] for h in hts]
            # note BEFORE heights: ledger rows truncate long results, and the
            # provenance must survive into the evidence
            return _HeightsDict({
                "shape": [len(hts), len(hts[0]) if hts else 0],
                "note": _GNN_NOTE,
                "heights": hts,
            })
        with costs.timed("get_heights_sample", h11=_h11_of(ks_ind),
                         n_points=len(p.points())):
            tris = p.random_triangulations_fast(
                N=n, max_retries=5, make_star=True, as_list=True,
                progress_bar=False, seed=seed,
            )
        return _shaped([t.heights().tolist() for t in tris])

    difficulty, msg = _triangulation_difficulty(p, kind)
    if difficulty > effort:
        est = costs.estimate(f"get_heights_{kind}", h11=_h11_of(ks_ind),
                             strict=True)
        measured = (f" Measured cost at this size: median "
                    f"{est['median_s']}s, p90 {est['p90_s']}s "
                    f"(n={est['n']})." if est else "")
        gnn_tip = (f" For a near-uniform sample (fair statistics), call "
                   f"get_heights({ks_ind!r}, n=<count>, sampler='gnn')."
                   if _gnn_available() else "")
        raise ValueError(
            f"enumerating ALL {kind} of {ks_ind} is difficulty {difficulty} "
            f"but effort {effort} ('{msg}').{measured} For a large polytope, "
            f"sample triangulations instead: call "
            f"get_heights({ks_ind!r}, n=<count>).{gnn_tip}"
        )

    with costs.timed(f"get_heights_{kind}", h11=_h11_of(ks_ind),
                     n_points=len(p.points())):
        if kind == "NTFE":
            return _shaped([h.tolist()
                            for h in p.ntfe_frsts(heights_only=True)])
        if kind == "FRST":
            return _shaped([t.heights().tolist()
                            for t in p.all_triangulations(
                only_fine=True, only_regular=True, only_star=True,
                as_list=True, include_points_interior_to_facets=True)])
    raise ValueError(f"kind must be 'NTFE' or 'FRST', got {kind!r}")

# model-read
@forgive_kwargs
def get_triangulation_info(ks_ind: str, heights: list[float]) -> dict:
    """
    Get info about the triangulation defined by input heights.

    Parameters
    ----------
    ks_ind : str
        The id of the polytope, of the form "h11-X_h21-Y_ind-Z".
    heights : list[float]
        The list of heights, with length equal to n_points_interior_to_facets.
        Pass a whole list of height vectors to get one result per triangulation.

    Returns
    -------
    dict
        is_valid, is_fine, is_regular, is_star, hash, n_simplices. A list of
        these if multiple height vectors were passed.
    """
    heights = _as_heights(heights)
    if heights and isinstance(heights[0], (list, tuple)):
        return [get_triangulation_info(ks_ind, h) for h in heights]
    p = get_polytope(ks_ind)
    t = p.triangulate(heights=heights, make_star=True)

    return _InfoDict({
        "is_valid": t.is_valid(),
        "is_fine": t.is_fine(),
        "is_regular": t.is_regular(),
        "is_star": t.is_star(),
        "hash": hash(t),
        "n_simplices": len(t.simplices())
    })

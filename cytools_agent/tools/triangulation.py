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
from cytools_agent.tools.polytope import get_polytope
from cytools_agent.tools.cy import _as_heights
from cytools_agent.tools._synonyms import forgive_kwargs

# human-read
def _triangulation_difficulty(poly: cytools.Polytope) -> tuple[float, str]:
    """
    Estimate how hard enumerating this polytope's triangulations is, from its
    number of lattice points (NTFE/FRST counts blow up with the point count).

    Returns a difficulty in [0, 1] and a short message. Calibrated so that
    h11~10-12 polytopes (~15+ points) land in the 'VERY tough' band.
    """
    n = len(poly.points())

    if n <= 9:            # h11 <~ 4
        return 0, "easy"
    elif n <= 12:         # h11 ~ 5-7
        return 0.1, "doable"
    elif n <= 14:         # h11 ~ 8
        return 0.5, "tough"
    elif n <= 17:         # h11 ~ 10-12
        return 0.9, "VERY tough"
    else:                 # h11 >~ 13
        return 1, "too large"

# human-read
def _shaped(heights: list[list[float]]) -> dict:
    """Wrap a list of height vectors with its shape, so callers read the count
    from `shape[0]` instead of counting the vectors."""
    return {
        "shape": [len(heights), len(heights[0]) if heights else 0],
        "heights": heights,
    }

# model-read
@forgive_kwargs
def get_heights(ks_ind: str, n: int | None = None, kind: str = "NTFE",
                effort: float = 0.5,
                seed: int | None = None) -> dict:
    """
    Triangulations of a polytope, as a dict {"shape": [n_triangulations,
    n_points], "heights": [...]}. "shape"[0] is HOW MANY triangulations there
    are; "heights" is the list of height vectors (one per triangulation), so
    heights[i] selects the i-th triangulation (e.g. for get_cy_info).

    Modes:
    - n given: a fast pseudorandom sample of up to `n` triangulations (heights
      drawn around the Delaunay heights; NOT a fair sample). Works at any size.
    - n omitted, kind="NTFE" (default): ALL inequivalent triangulations - FRSTs
      modulo equivalent restrictions to 2D faces. This is what distinguishes
      Calabi-Yaus, and is usually few.
    - n omitted, kind="FRST": ALL fine regular star triangulations. Many more
      than NTFE, and blows up much faster with size, so it is guarded more
      strictly (refuses at smaller polytopes).

    The exhaustive modes blow up with size, so they are guarded by `effort`
    (raise when harder than allowed; FRST more aggressively).

    Parameters
    ----------
    ks_ind : str
        The id of a polytope, of the form "h11-X_h21-Y_ind-Z".
    n : int, optional
        Sample size for the fast random mode. Omit for an exhaustive mode.
    kind : str, optional
        For the exhaustive mode: "NTFE" (default) or "FRST".
    effort : float, optional
        How hard to try. >0 easiest only, >0.1 moderate, >0.5 tough, >0.9 very
        tough, >1 any case.
    seed : int, optional
        Random seed for the sampling (n given) mode.

    Returns
    -------
    dict
        {"shape": [n_triangulations, n_points], "heights": list of height
        vectors}. Read the count from shape[0].
    """
    p = get_polytope(ks_ind)

    if n is not None:
        tris = p.random_triangulations_fast(
            N=n, max_retries=5, make_star=True, as_list=True,
            progress_bar=False, seed=seed,
        )
        return _shaped([t.heights().tolist() for t in tris])

    difficulty, msg = _triangulation_difficulty(p)
    if kind == "FRST":
        difficulty = min(1.0, difficulty + 0.4)   # FRSTs blow up much faster
    if difficulty > effort:
        raise ValueError(
            f"enumerating ALL {kind} of {ks_ind} is difficulty {difficulty} "
            f"but effort {effort} ('{msg}'). For a large polytope, sample "
            f"triangulations instead: call get_heights({ks_ind!r}, n=<count>)."
        )

    if kind == "NTFE":
        return _shaped([h.tolist() for h in p.ntfe_frsts(heights_only=True)])
    if kind == "FRST":
        return _shaped([t.heights().tolist() for t in p.all_triangulations(
            only_fine=True, only_regular=True, only_star=True, as_list=True,
            include_points_interior_to_facets=True)])
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

    return {
        "is_valid": t.is_valid(),
        "is_fine": t.is_fine(),
        "is_regular": t.is_regular(),
        "is_star": t.is_star(),
        "hash": hash(t),
        "n_simplices": len(t.simplices())
    }

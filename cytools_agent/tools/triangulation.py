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
from cytools_agent.tools.history import logged
from cytools_agent.tools.polytope import get_polytope

# non-model-facing
# ----------------
def _guard_ntfe_call(poly: cytools.Polytope) -> tuple[float, str]:
    """
    Prevent the model from grabbing NTFEs of too large of a polytope.

    Returns an effort level (0 is maximally easy; 1 is maximally hard)
    as well as a brief message about the difficulty.
    """
    max_n_pts = max([len(f.points()) for f in poly.faces(2)])

    if max_n_pts <= 12:
        return 0, "easy"
    elif max_n_pts <= 15:
        return 0.1, "doable"
    elif max_n_pts <= 17:
        return 0.5, "tough"
    elif max_n_pts <= 21:
        return 0.9, "VERY tough"
    else:
        return 1, "too large"

# model-facing
# ------------
@logged
def get_heights(ks_ind: str, n: int | None = None, effort: float = 0.5,
                seed: int | None = None) -> list[list[float]]:
    """
    Height vectors (each selecting a triangulation) for a polytope.

    Two modes:
    - n is None: return ALL inequivalent triangulations - one height vector per
      fine regular star triangulation (FRST), modulo equal restrictions to 2D
      faces (an NTFE). Exact, but blows up with size, so it is guarded by
      `effort` and raises for cases harder than that allows.
    - n given: return a fast pseudorandom sample of up to `n` triangulations
      (heights drawn around the Delaunay heights). Works at any size but is NOT
      a fair sample, and may return fewer than `n`.

    Parameters
    ----------
    ks_ind : str
        The id of a polytope, of the form "h11-X_h21-Y_ind-Z".
    n : int, optional
        How many random triangulations to sample. If omitted, returns ALL
        inequivalent triangulations instead.
    effort : float, optional
        For the exhaustive (n=None) mode: how hard to try. >0 easiest cases
        only, >0.1 moderate, >0.5 tough, >0.9 very tough, >1 any case.
    seed : int, optional
        For the sampling (n given) mode: random seed, for reproducibility.

    Returns
    -------
    list of list of float
        One height vector per triangulation.
    """
    p = get_polytope(ks_ind)

    if n is None:
        difficulty, msg = _guard_ntfe_call(p)
        if difficulty > effort:
            raise ValueError(
                f"polytope {ks_ind} has difficulty level {difficulty} but "
                f"effort level {effort}. Case seems too hard. Guard message "
                f"'{msg}'."
            )
        return [h.tolist() for h in p.ntfe_frsts(heights_only=True)]

    tris = p.random_triangulations_fast(
        N=n, max_retries=5, make_star=True, as_list=True,
        progress_bar=False, seed=seed,
    )
    return [t.heights().tolist() for t in tris]

@logged
def get_triangulation_info(ks_ind: str, heights: list[float]) -> dict:
    """
    Get info about the triangulation defined by input heights.

    Parameters
    ----------
    ks_ind : str
        The id of the polytope, of the form "h11-X_h21-Y_ind-Z".
    heights : list[float]
        The list of heights, with length equal to n_points_interior_to_facets.

    Returns
    -------
    dict
        is_valid, is_fine, is_regular, is_star, hash, n_simplices
    """
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

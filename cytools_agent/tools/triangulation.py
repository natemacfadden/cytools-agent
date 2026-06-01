# =============================================================================
# This file is part of CYTools-agent.
#
# CYTools-agent is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# CYTools-agent is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# CYTools-agent. If not, see <https://www.gnu.org/licenses/>.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  Triangulation tools for AI agents. Focuses on generation of
#               triangulations by heights and NTFE triangulations.
# -----------------------------------------------------------------------------

# CYTools import
import cytools

# cytools-agent imports
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
def all_inequiv_heights(ks_ind: str, effort: float = 0.5) -> list[list[float]]:
    """
    Enumerate height vectors, one per fine regular star triangulation (FRST) of
    the polytope (mod triangulations with equivalent restrictions to 2D faces).
    Each distinct triangulation is called an NTFE (non-two-face-equivalent).

    Cheap for small polytopes (h11<~17) but blows up with size, so double check
    for calls with h11>~21.

    Parameters
    ----------
    ks_ind : str
        The id of a polytope, of the form "h11-X_h21-Y_ind-Z".
    effort : float, optional
        How hard to try for the heights. >0 means only go for the easiest cases.
        >0.1 means go for moderate cases. >0.5 means go for tough cases. >0.9
        means go for very tough cases. >1 means go for any case.

    Returns
    -------
    list of list of float
        One height vector per inequivalent FRST.
    """
    p = get_polytope(ks_ind)

    # guard the call
    difficulty, msg = _guard_ntfe_call(p)
    if difficulty > effort:
        raise ValueError(
            f"polytope {ks_ind} has difficulty level {difficulty} but effort "
            f"level {effort}. Case seems too hard. Guard message '{msg}'."
        )

    heights = p.ntfe_frsts(heights_only=True)
    return [h.tolist() for h in heights]

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

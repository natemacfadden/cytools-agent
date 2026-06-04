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
# Description:  Calabi-Yau layer. A CY follows trivially from a triangulation,
#               so it is rebuilt on demand from (ks_ind, heights). Raw tensors
#               (intersection numbers, c2) are reached in code via get_cy(...)
#               in run_python; get_cy_info_at_point returns the kappa
#               contractions at a point in Kahler moduli space.
# -----------------------------------------------------------------------------

# external imports
import numpy as np

# local imports
from cytools_agent.tools.history import logged
from cytools_agent.tools.polytope import get_polytope

# non-model-facing
# ----------------
def get_cy(ks_ind, heights):
    """The Calabi-Yau from triangulating `ks_ind` with `heights`."""
    poly = get_polytope(ks_ind)
    if not poly.is_favorable(lattice="N"):
        raise ValueError(
            f"{ks_ind} is non-favorable; its CY needs experimental "
            "CYTools features, so skip it"
        )
    # make_star: get_heights gives FRST heights, but triangulate won't
    # force the star unless asked, and get_cy needs a star triangulation
    tri = poly.triangulate(heights=heights, make_star=True)
    return tri.get_cy()


def _mori_cone(cy, which):
    """
    The Mori cone in basis; its dual is the Kahler cone. 'Kcup' caps the Mori
    cone (more accurate Kahler cone); 'toric' uses the toric Mori cone (the
    tutorial's).
    """
    if which == "Kcup":
        return cy.mori_cone_cap(in_basis=True)
    if which == "toric":
        return cy.toric_mori_cone(in_basis=True)
    raise ValueError(f"cone must be 'Kcup' or 'toric', got {which!r}")

# model-facing
# ------------
@logged
def get_cy_info_at_point(ks_ind: str, heights: list[float],
                         t: list[float] | None = None,
                         cone: str = "Kcup") -> dict:
    """
    Intersection-number contractions at a point in Kahler moduli space.

    For the CY of `ks_ind` triangulated by `heights`, picks a point `t`, checks
    it lies in the Kahler cone, then contracts the in-basis intersection
    numbers kappa with `t`: A = kappa@t (a matrix), divisor volumes =
    0.5*kappa@t@t (a vector), and CY volume = (1/6)*kappa@t@t@t (a scalar).

    Parameters
    ----------
    ks_ind : str
        The id of a fetched polytope, of the form "h11-X_h21-Y_ind-Z".
    heights : list[float]
        Heights selecting the triangulation (from get_heights).
    t : list[float], optional
        A point in the basis Kahler moduli space (length h11). LEAVE THIS OUT
        unless you have a specific point: by default it uses the tip of the
        stretched Kahler cone, a canonical interior point. Do NOT hardcode an
        arbitrary point like [1, 0, ...] - it almost never lies in the cone.
    cone : str, optional
        Which Kahler cone to use: "Kcup" (more accurate, the default) or
        "toric" (cheaper, and much faster at large h11).

    Returns
    -------
    dict
        cone (which cone was used), t (the point used), A (= kappa@t),
        divisor_volumes (= 0.5*kappa@t@t), and cy_volume (= kappa@t@t@t / 6).
    """
    cy = get_cy(ks_ind, heights)
    K = _mori_cone(cy, cone).dual()

    if t is None:
        t = K.tip_of_stretched_cone(1)   # canonical interior point
        if t is None:
            raise ValueError("could not find a stretched-cone tip; pass t")
    t = np.asarray(t, dtype=float)

    if not K.contains(t):
        raise ValueError(f"point t is not in the {cone} Kahler cone")

    kappa = cy.intersection_numbers(in_basis=True, format="dense")
    A = np.tensordot(kappa, t, axes=([2], [0]))   # kappa @ t
    ktt = A @ t                                    # kappa @ t @ t
    return {
        "cone": cone,
        "t": t.tolist(),
        "A": A.tolist(),
        "divisor_volumes": (0.5 * ktt).tolist(),
        "cy_volume": float(ktt @ t / 6),
    }


@logged
def get_cy_cones(ks_ind: str, heights: list[float],
                 cone: str = "Kcup") -> dict:
    """
    The Mori cone rays of the CY, in the basis of divisors.

    These rays generate the Mori (effective-curve) cone and are exactly the
    hyperplane normals of the dual Kahler cone, so this one array describes
    both cones (Kahler cone = {t : ray . t >= 0 for every ray}).

    Parameters
    ----------
    ks_ind : str
        The id of a fetched polytope, of the form "h11-X_h21-Y_ind-Z".
    heights : list[float]
        Heights selecting the triangulation (from get_heights).
    cone : str, optional
        Which Mori cone: "Kcup" (the capped/accurate one, the default) or
        "toric" (the toric Mori cone, as in the tutorial). Kcup gets very
        expensive at large h11 (huge ray count) -- use "toric" there.

    Returns
    -------
    dict
        cone (which cone was used) and mori_rays (its generating rays, which
        are also the dual Kahler cone's hyperplane normals).
    """
    mori = _mori_cone(get_cy(ks_ind, heights), cone)
    return {"cone": cone, "mori_rays": mori.rays().tolist()}

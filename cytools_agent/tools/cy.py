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
#               so it is rebuilt on demand from (ks_ind, heights). get_cy_info
#               returns CY invariants (optionally volumes at a Kahler point);
#               get_cy_cones returns the Mori/Kahler cone; raw objects are
#               reached in code via get_cy(...) in run_python.
# -----------------------------------------------------------------------------

# external imports
import numpy as np

# local imports
from cytools_agent.tools.polytope import get_polytope, _InfoDict
from cytools_agent.tools._synonyms import forgive_kwargs


def _exact_int(v, tol=1e-6):
    """Cast an expected-integer CYTools value to int, raising if it is not
    integral within tol. Guards against a flint/cytools regression that would
    otherwise be silently rounded to a wrong integer."""
    r = round(v)
    if abs(v - r) > tol:
        raise ValueError(f"non-integer where integer expected: {v!r} "
                         f"(|v - round(v)| = {abs(v - r)})")
    return int(r)


# human-read
class _ResultList(list):
    """A list of per-triangulation result dicts. get_cy_info / get_cy_cones
    ALWAYS return one of these (length 1 for a single triangulation), so the
    SAME code works for one CY or many: iterate it (for r in result) or
    aggregate (min(r['curve_volumes'] for r in result)). As a convenience, when
    it holds exactly ONE result you may also use it like that dict directly
    (result['cy_volume'], 'cy_volume' in result, result.keys()). Dict-style
    access on a MULTI result raises a clear message instead of Python's opaque
    'list indices must be integers'."""

    def _single(self, op):
        if len(self) == 1:
            return self[0]
        raise TypeError(
            f"this holds {len(self)} results (one per triangulation), not one "
            f"dict, so {op} is ambiguous. Index one (result[0]) or aggregate "
            "over all (e.g. min(r['curve_volumes'] for r in result)).")

    def __getitem__(self, key):
        if isinstance(key, str):              # dict-style access: ok iff single
            return self._single(f"result[{key!r}]")[key]
        return super().__getitem__(key)       # int / slice: normal list access

    def __contains__(self, key):
        if isinstance(key, str) and len(self) == 1:
            return key in self[0]
        return super().__contains__(key)

    def keys(self):
        return self._single("result.keys()").keys()

    def values(self):
        return self._single("result.values()").values()

    def items(self):
        return self._single("result.items()").items()

    def get(self, key, default=None):
        return self._single(f"result.get({key!r})").get(key, default)


# human-read
def _as_heights(heights):
    """Accept the get_heights(...) result directly: if given that dict (or any
    dict with a 'heights' key), use its list of height vectors. Otherwise pass
    through (a single height vector, or a list of them)."""
    if isinstance(heights, dict) and "heights" in heights:
        return heights["heights"]
    # a bare scalar (e.g. the model passed a Hodge number as `heights`) would
    # otherwise hit `heights[0]` downstream and raise a cryptic "'int' object is
    # not subscriptable". Catch it here with a message that names the mistake.
    if heights is not None and not isinstance(heights, (list, tuple, np.ndarray)):
        raise TypeError(
            f"heights must be a height vector (a list of numbers), a list of "
            f"them, or the get_heights(ks_ind) result -- got "
            f"{type(heights).__name__} {heights!r}. Omit heights for the default "
            f"triangulation, or pass get_heights(ks_ind)['heights'][0]. (A bare "
            f"number here is often a mistaken Hodge number -- those are fetch "
            f"filters, e.g. fetch_polytopes(h21=...).)")
    return heights


# model-read (exposed in the run_python namespace)
@forgive_kwargs
def get_cy(ks_ind: str, heights: list[float] | dict | None = None):
    """The Calabi-Yau(s) from triangulating `ks_ind`. With no heights it uses a
    default star triangulation (one CY). Pass ONE height vector for that
    triangulation's CY, or many (e.g. the get_heights(ks_ind) result) to get a LIST
    of CYs, one per triangulation."""
    poly = get_polytope(ks_ind)
    if not poly.is_favorable(lattice="N"):
        raise ValueError(
            f"{ks_ind} is non-favorable, so its CY is unsupported here. "
            "Fetch a favorable one: fetch_polytopes(..., favorable=True)."
        )
    heights = _as_heights(heights)
    if heights and isinstance(heights[0], (list, tuple)):
        # many height vectors -> a CY each (like get_cy_info / get_cy_cones)
        return _ResultList(get_cy(ks_ind, h) for h in heights)
    # make_star: get_heights gives FRST heights, but triangulate won't force
    # the star unless asked, and get_cy needs a star triangulation. With no
    # heights, triangulate picks default (Delaunay) heights.
    if heights is None:
        # flag the confusable case: this is ONE CY, not necessarily the only
        print("[note: get_cy(ks_ind) built ONE CY from a default triangulation; "
              "this polytope may admit several inequivalent CYs -- enumerate "
              "them with get_heights(ks_ind)['heights'], e.g. get_cy_info(ks_ind, "
              "get_heights(ks_ind))]")
        tri = poly.triangulate(make_star=True)
    else:
        try:
            tri = poly.triangulate(heights=heights, make_star=True)
        except Exception as e:
            m = str(e).lower()
            if "height" in m and "point" in m:
                # the common confusion: VolumeFinder's heights are a
                # vector-config representation, not triangulation heights, so
                # feeding them back here mismatches the point count.
                raise ValueError(
                    f"{e} -- a height vector needs one entry per triangulation "
                    "point. If these heights came from "
                    "find_kahler_for_divisor_volumes (VolumeFinder), they are "
                    "not triangulation heights; do not pass them to get_cy. "
                    "The volume at that point is already in that tool's "
                    "result -- read result['cy_volume'].")
            raise
    return tri.get_cy()


# human-read
def _mori_cone(cy, which):
    """
    A Mori cone of the CY in basis (its dual is the matching Kahler cone).
    `which` picks the dual pair:
      "Kcup"  -> Mcap (cy.mori_cone_cap): the outer approximation to the true
                 Mori cone. Its dual is Kcup, the inner approximation to the
                 true Kahler cone (the union of the toric Kahler cones of all
                 2-face-equivalent triangulations). Kcup != Mcap; they are
                 duals (rays <-> hyperplanes). Most accurate, costly at large
                 h11.
      "toric" -> the toric Mori cone of THIS triangulation
                 (cy.toric_mori_cone). Its dual is the toric Kahler cone.
                 Cheaper, much faster at large h11.
    """
    if which == "Kcup":
        return cy.mori_cone_cap(in_basis=True)
    if which == "toric":
        return cy.toric_mori_cone(in_basis=True)
    raise ValueError(f"cone must be 'Kcup' or 'toric', got {which!r}")

# human-read
def _cy_info_one(ks_ind, heights, t, cone):
    """CY invariants for ONE triangulation (heights is one vector or None);
    get_cy_info wraps this in a _ResultList. See get_cy_info's docstring."""
    import time as _time
    from cytools_agent.tools import costs
    from cytools_agent.tools.polytope import _h11_of
    _t0 = _time.monotonic()
    cy = get_cy(ks_ind, heights)
    dok = cy.intersection_numbers(in_basis=True, format="dok")
    info = _InfoDict({
        "h11": int(cy.h11()),
        "h21": int(cy.h21()),
        "euler_characteristic": int(2 * (cy.h11() - cy.h21())),
        "second_chern_class": cy.second_chern_class(in_basis=True).tolist(),
        "intersection_numbers": [[*map(int, k), _exact_int(v)]
                                 for k, v in dok.items()],
        "n_prime_toric_divisors": len(cy.prime_toric_divisors()),
    })
    if t is None:
        costs.record("get_cy_info", _time.monotonic() - _t0,
                     h11=_h11_of(ks_ind), with_t=False)
        return info

    mori = _mori_cone(cy, cone)
    K = mori.dual()
    if t == "tip":
        t = K.tip_of_stretched_cone(1)   # canonical interior point
        if t is None:
            raise ValueError("could not find a stretched-cone tip; pass t")
    t = np.asarray(t, dtype=float)
    if not K.contains(t):
        raise ValueError(
            f"point t is not in the {cone} Kahler cone; omit t (or pass "
            "t='tip') to use the stretched-cone tip."
        )

    kappa = cy.intersection_numbers(in_basis=True, format="dense")
    ktt = np.tensordot(kappa, t, axes=([2], [0])) @ t   # kappa @ t @ t
    curve_vols = np.asarray(mori.rays()) @ t   # Mori-ray . t = curve volumes
    info["cone"] = cone
    info["t"] = t.tolist()
    info["divisor_volumes"] = (0.5 * ktt).tolist()
    info["cy_volume"] = float(ktt @ t / 6)
    info["curve_volumes"] = curve_vols.tolist()   # reduce as needed: min/max
    costs.record("get_cy_info", _time.monotonic() - _t0,
                 h11=_h11_of(ks_ind), with_t=True, cone=cone)
    return info


# model-read
@forgive_kwargs
def get_cy_info(ks_ind: str, heights: list[float] | dict | None = None,
                t: list[float] | str | None = None,
                cone: str = "Kcup") -> "_ResultList":
    """
    Invariants of the Calabi-Yau, optionally evaluated at a Kahler point.

    Always returns the point-INDEPENDENT invariants (cheap; no cone needed):
    Hodge numbers, Euler characteristic, second Chern class, the nonzero
    in-basis triple intersection numbers, and the number of prime toric
    divisors. If `t` is given, ALSO checks `t` is in the Kahler cone and adds
    the divisor volumes (0.5*kappa@t@t), CY volume (kappa@t@t@t / 6), and curve
    volumes (Mori-cone ray . t) there.

    Parameters
    ----------
    ks_ind : str
        The id of a fetched polytope, of the form "h11-X_h21-Y_ind-Z".
    heights : list[float]
        Heights selecting the triangulation (a get_heights(...)["heights"][i]).
        Pass a whole list of height vectors -- or the get_heights(...) result
        itself -- to get one result per (inequivalent) triangulation.
    t : list[float] or "tip", optional
        Leave unset to return only the invariants (see Returns); pass a point
        to also get the divisor/CY volumes there. If the user gave a specific
        point, use it (a length-h11 vector); otherwise pass "tip" for a
        canonical interior point (the stretched-cone tip). Do not invent an
        arbitrary point -- it is usually outside the cone.
    cone : str, optional
        Which Kahler cone the point t must lie in -- the dual of the Mori cone
        named here:
        "Kcup" (default) is Kcup, the inner approximation to the true Kahler
        cone (the union of the toric Kahler cones of all 2-face-equivalent
        triangulations; dual to Mcap = cy.mori_cone_cap). Most accurate, costly
        at large h11.
        "toric" is the toric Kahler cone of this single triangulation (dual to
        the toric Mori cone). Cheaper, much faster at large h11.

    Returns
    -------
    _ResultList (a list of result dicts, one per triangulation)
        Each dict has h11, h21, euler_characteristic, second_chern_class,
        intersection_numbers (nonzero, in-basis, as [i, j, k, value]), and
        n_prime_toric_divisors; plus, if t is given, cone, t, divisor_volumes,
        cy_volume, and curve_volumes (reduce as needed, e.g. min/max).
        Aggregate with `for r in result` / min(r['curve_volumes'] for r in
        result). For ONE triangulation it has length 1 and also acts like that
        dict directly (result['cy_volume']).
    """
    heights = _as_heights(heights)
    if heights and isinstance(heights[0], (list, tuple)):
        return _ResultList(_cy_info_one(ks_ind, h, t, cone) for h in heights)
    return _ResultList([_cy_info_one(ks_ind, heights, t, cone)])


# model-read
@forgive_kwargs
def get_cy_cones(ks_ind: str, heights: list[float] | dict | None = None,
                 cone: str = "Kcup") -> "_ResultList":
    """
    The cone data of the CY, in the basis of divisors. The same vectors play
    two dual roles, so they are returned under BOTH labels: as the generating
    RAYS of the Mori (effective-curve) cone, and as the HYPERPLANE normals
    bounding the dual Kahler cone (Kahler cone = {t : ray . t >= 0 for every
    ray}). So the number of hyperplanes bounding the Kahler cone is the length
    of either array.

    Parameters
    ----------
    ks_ind : str
        The id of a fetched polytope, of the form "h11-X_h21-Y_ind-Z".
    heights : list[float]
        Heights selecting the triangulation (a get_heights(...)["heights"][i]).
        Pass a whole list of height vectors to get one result per triangulation.
    cone : str, optional
        Which dual cone pair to return (mori_rays and the dual
        kahler_cone_hyperplanes):
        "Kcup" (default) returns Mcap = cy.mori_cone_cap as mori_rays; its dual
        is Kcup, the inner approximation to the true Kahler cone (Kcup != Mcap;
        they are duals). Most accurate, expensive at large h11.
        "toric" returns the toric Mori cone of this triangulation; its dual is
        the toric Kahler cone. Cheaper at large h11.

    Returns
    -------
    _ResultList (a list of result dicts, one per triangulation)
        Each dict has cone (which cone was used), mori_rays (the Mori cone's
        generating rays), and kahler_cone_hyperplanes (the SAME vectors, as the
        dual Kahler cone's bounding hyperplane normals). Iterate with `for r in
        result`. For ONE triangulation it has length 1 and also acts like that
        dict directly (result['mori_rays']).
    """
    heights = _as_heights(heights)
    if heights and isinstance(heights[0], (list, tuple)):
        return _ResultList(_cy_cones_one(ks_ind, h, cone) for h in heights)
    return _ResultList([_cy_cones_one(ks_ind, heights, cone)])


# human-read
def _cy_cones_one(ks_ind, heights, cone):
    """Cone data for ONE triangulation; get_cy_cones wraps this."""
    mori = _mori_cone(get_cy(ks_ind, heights), cone)
    rays = mori.rays().tolist()
    return _InfoDict({"cone": cone, "mori_rays": rays,
                      "kahler_cone_hyperplanes": rays})


# model-read (exposed in the run_python namespace)
@forgive_kwargs
def find_kahler_for_divisor_volumes(ks_ind: str,
                                    target: list[float]) -> "_InfoDict":
    """Find Kahler parameters (heights) of the CY from `ks_ind` at which its
    basis divisor volumes equal `target`, using fanroots' VolumeFinder
    root-finder. `target` is a list of divisor volumes, one per basis divisor
    (length = h11).

    Returns a dict with `converged` (bool) and `status` (a message that flags
    non-convergence). When it converged, the answer fields are at the top level:
    `heights`/`t` (the Kahler point), `divisor_volumes` (achieved; ~= target),
    and `cy_volume` (the CY volume at `t`).

    When it did NOT converge the target was likely unreachable / outside the
    Kahler cone, so there is no solution to report. The same fields are then
    demoted under `last_iterate` (the stalled point, kept for inspection) and
    are absent at the top level -- so a bare `result['cy_volume']` raises a
    pointed error instead of silently handing back a non-solution as if it were
    the answer. Reach into `['last_iterate']` only if you deliberately want the
    stalled point; do not report it as the answer.
    """
    from cytools_agent.tools.polytope import _h11_of
    tgt = np.asarray(target, dtype=float)
    if tgt.ndim != 1:
        raise TypeError(
            "target must be a flat list of divisor volumes (one per basis "
            f"divisor); got an array of shape {tgt.shape}.")
    h11 = _h11_of(ks_ind)
    if len(tgt) != h11:
        raise ValueError(
            f"target must have {h11} entries -- one per basis divisor of this "
            f"h11={h11} polytope -- but got {len(tgt)}.")
    poly = get_polytope(ks_ind)
    if not poly.is_favorable(lattice="N"):
        raise ValueError(
            f"{ks_ind} is non-favorable, so its CY is unsupported here. "
            "Fetch a favorable one: fetch_polytopes(..., favorable=True).")
    try:
        from fanroots.applications.volume_finder import VolumeFinder
    except ImportError as e:
        raise ImportError(
            "fanroots (VolumeFinder) is not importable in this "
            f"environment: {e}")
    vc = poly.vc()
    vf = VolumeFinder(target=tgt, vc=vc, history_level=0, verbosity=0)
    vf.optimize()
    converged = vf.finished_reason == "converged"
    h = np.asarray(vf.heights, dtype=float)
    t = np.asarray(vc.proj(h), dtype=float)
    dv = np.asarray(vf.div_vols(h), dtype=float)
    point = {"heights": h.tolist(), "t": t.tolist(),
             "divisor_volumes": dv.tolist(), "cy_volume": float(t @ dv / 3.0)}
    if converged:
        return _InfoDict({"converged": True, "status": "converged", **point})
    # not converged: the target is likely unreachable / outside the Kahler cone,
    # so `point` is not a solution. Demote it under `last_iterate` so grabbing a
    # bare answer field (e.g. compute_for_each's `...['cy_volume']`) raises a
    # pointed error rather than committing the stalled value as the answer.
    return _InfoDict({
        "converged": False,
        "status": (
            f"did not converge (finished_reason={vf.finished_reason!r}): the "
            "target divisor volumes are unreachable / outside the Kahler cone, "
            "so no such Kahler point exists -- the task is infeasible. The "
            "stalled last iterate is under 'last_iterate' for inspection only; "
            "do not report it as the answer."),
        "last_iterate": _InfoDict(point),
    })

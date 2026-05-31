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
# Description:  Polytope tools for AI agents. Polytopes are fetched from the
#               Kreuzer-Skarke database and cached (as vertices, to keep memory
#               low) in a module-level dict keyed by (h11, h21, ind) ids.
# -----------------------------------------------------------------------------

# CYTools import
import cytools

# cytools-agent imports
from cytools_agent.tools.history import logged

# module-level cache
# ------------------
_CACHE   = {} # ks_ind -> vertices (list[list[int]])
_FETCHED = {} # (h11, h21) -> {"count": int, "complete": bool}; how much of each
              # query is known as a contiguous prefix (from index 0) in the
              # cache

# non-model-facing
# ----------------
# cache management
def _get_polytope(ks_ind: str) -> cytools.Polytope:
    """
    Reconstruct the cached Polytope associated with a ks_ind.
    """
    return cytools.Polytope(_CACHE[ks_ind])

def _cache_can_serve(h11: int, h21: int | None, limit: int) -> bool:
    """
    Whether the cache already holds the first `limit` polytopes of this query.
    """
    exact = _FETCHED.get((h11, h21))
    if exact and (exact["complete"] or exact["count"] >= limit):
        return True

    # a fully-exhausted (h11-only) fetch determines every (h11, h21) subquery
    # (conservative: this can be done even if broad["complete"]==False, but
    #  that requires more careful tracking of #polys per h21)
    if h21 is not None:
        broad = _FETCHED.get((h11, None))
        if broad and broad["complete"]:
            return True

    return False

def _get_cached_ks_inds(h11: int, h21: int | None) -> list[str]:
    """
    The cached ids matching (h11, h21), in canonical fetch order (h21 asc, then
    ind asc). h21=None matches every h21.
    """
    matches = []

    for ks_ind in _CACHE:
        _h11,_h21,ind = (int(part.split("-")[1]) for part in ks_ind.split("_"))

        if _h11 != h11 or (h21 is not None and _h21 != h21):
            continue
        else:
            # prepend h21 and ind for sorting...
            matches.append((_h21, ind, ks_ind))

    matches.sort()
    return [ks_ind for _, _, ks_ind in matches]

# model-facing
# ------------
@logged
def fetch_polytopes(limit: int, h11: int, h21: int | None = None) -> list[str]:
    """
    Fetch 4D reflexive polytopes from the Kreuzer-Skarke database.

    Each returned id has the form "h11-X_h21-Y_ind-Z", where Z is the position
    of the polytope within the (h11, h21) group.

    `limit` is REQUIRED. If the user has not said how many polytopes they want,
    ASK THEM before calling this tool - do NOT guess a limit. The KS database
    is large and a high limit can fetch thousands.

    Parameters
    ----------
    limit : int
        The maximum number of polytopes to fetch (required).
    h11 : int
        The Hodge number h11 of the desired polytopes.
    h21 : int, optional
        The Hodge number h21 of the desired polytopes.

    Returns
    -------
    list of str
        The ids of the matching polytopes.
    """
    # serve from the cache when the query is already fully known
    if _cache_can_serve(h11, h21, limit):
        return _get_cached_ks_inds(h11, h21)[:limit]

    # otherwise hit the database
    polys = cytools.fetch_polytopes(
        h11=h11, h21=h21, limit=limit, dim=4, lattice="N", as_list=True
    )

    # save into caches; polytopes arrive in lexicographic (h11, h21, ind) order,
    # so ind counts within each (h11, h21) group and resets when the group changes
    prev_group, ind = None, 0
    for p in polys:
        _h11, _h21 = int(p.h11(lattice="N")), int(p.h21(lattice="N"))
        ind = ind + 1 if (_h11, _h21) == prev_group else 0
        prev_group = (_h11, _h21)

        _CACHE[f"h11-{_h11}_h21-{_h21}_ind-{ind}"] = p.vertices().tolist()

    # record how much of this query is now fully known
    n = len(polys)
    prev = _FETCHED.get((h11, h21))
    _FETCHED[(h11, h21)] = {
        "count": max(n, prev["count"] if prev else 0),
        "complete": bool(prev and prev["complete"]) or (n < limit),
    }
    return _get_cached_ks_inds(h11, h21)[:limit]

@logged
def get_polytope_info(ks_ind: str) -> dict:
    """
    Return geometric information about a cached polytope.

    Parameters
    ----------
    ks_ind : str
        The id of a fetched polytope, of the form "h11-X_h21-Y_ind-Z".

    Returns
    -------
    dict
        h11, h21, favorable_N, favorable_M, is_trilayer, n_points,
        n_points_interior_to_facets, n_vertices, and facedim_to_facepts (a dict
        from face dimension to the list of point-counts of the faces of that
        dimension).
    """
    p = _get_polytope(ks_ind)
    return {
        "h11": int(p.h11(lattice="N")),
        "h21": int(p.h21(lattice="N")),
        "favorable_N": bool(p.is_favorable(lattice="N")),
        "favorable_M": bool(p.is_favorable(lattice="M")),
        "is_trilayer": bool(p.is_trilayer()),
        "n_points": len(p.points()),
        "n_points_interior_to_facets": len(p.points_interior_to_facets()),
        "n_vertices": len(p.vertices()),
        "facedim_to_facepts": {
            d: [len(f.points()) for f in p.faces(d)] for d in range(p.dim() + 1)
        },
    }

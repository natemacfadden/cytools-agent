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
# Description:  Polytope tools for AI agents. Polytopes are fetched from the
#               Kreuzer-Skarke database and cached (as vertices, to keep memory
#               low) in a module-level dict keyed by (h11, h21, ind) ids.
# -----------------------------------------------------------------------------

# external imports
import json
import os

import cytools

# local imports
from cytools_agent.tools.history import logged

# module-level cache
# ------------------
_CACHE   = {} # ks_ind -> vertices (list[list[int]])
_FETCHED = {} # (h11, h21) -> {"count": int, "complete": bool}; how much of each
              # query is known as a contiguous prefix (from index 0) in the
              # cache

# Kreuzer-Skarke polytope counts for the full 4d database
# (calabi-yau-data/polytopes-4d on HuggingFace; 473,800,776 polytopes total).
# ks_counts.json holds both by_pair (h11, h21) and by_h11 (h21-agnostic) counts.
# ---------------------------------------------------------------------------
with open(os.path.join(os.path.dirname(__file__), "ks_counts.json")) as _f:
    _KS = json.load(_f)
_KS_PAIR = {tuple(int(x) for x in k.split(",")): v
            for k, v in _KS["by_pair"].items()}
_KS_H11 = {int(k): v for k, v in _KS["by_h11"].items()}

# non-model-facing
# ----------------
# cache management
def get_polytope(ks_ind: str) -> cytools.Polytope:
    """Reconstruct the cached Polytope associated with a ks_ind."""
    return cytools.Polytope(_CACHE[ks_ind])

def _cache_can_serve(h11: int, h21: int | None, limit: int) -> bool:
    """True if the cache already holds the first `limit` of this query."""
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

def _filter_favorable(ks_inds: list[str], favorable: bool | None) -> list[str]:
    """Keep ids whose N-favorability matches `favorable` (None keeps all)."""
    if favorable is None:
        return ks_inds
    return [i for i in ks_inds
            if get_polytope(i).is_favorable(lattice="N") == favorable]

def _ensure_cached(h11: int, h21: int | None, limit: int) -> None:
    """Ensure the first `limit` of this query are cached (fetch if not)."""
    if _cache_can_serve(h11, h21, limit):
        return

    polys = cytools.fetch_polytopes(
        h11=h11, h21=h21, limit=limit, dim=4, lattice="N", as_list=True
    )

    # polytopes arrive in lexicographic (h11, h21, ind) order, so ind counts
    # within each (h11, h21) group, reset when the group changes
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

# model-facing
# ------------
@logged
def fetch_polytopes(limit: int, h11: int, h21: int | None = None,
                    favorable: bool | None = None) -> list[str]:
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
    favorable : bool, optional
        If set, return `limit` polytopes with this N-favorability, scanning
        deeper into the list as needed (you only get fewer if the DB runs out).

    Returns
    -------
    list of str
        The ids of the matching polytopes.
    """
    if favorable is None:
        _ensure_cached(h11, h21, limit)
        return _get_cached_ks_inds(h11, h21)[:limit]

    # favorable: scan deeper into the (h11, h21) list until `limit` matches are
    # found (the first favorable one may be well past index 0), or the DB ends
    scan = max(limit, 10)
    while True:
        _ensure_cached(h11, h21, scan)
        ids = _get_cached_ks_inds(h11, h21)
        fav = _filter_favorable(ids[:scan], favorable)
        if len(fav) >= limit or len(ids) < scan:
            return fav[:limit]
        scan *= 2

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
    p = get_polytope(ks_ind)
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

@logged
def ks_stats(h11: int, h21: int | None = None) -> dict:
    """
    Polytope counts in the Kreuzer-Skarke database of 4d reflexive polytopes.

    Use this to CHECK whether polytopes exist (and how many) at given Hodge
    numbers instead of guessing - the database spans h11 from 1 to 491.

    Parameters
    ----------
    h11 : int
        The Hodge number h11.
    h21 : int, optional
        The Hodge number h21. If omitted, counts over all h21 at this h11.

    Returns
    -------
    dict
        count and exists - for the exact (h11, h21) when h21 is given, else the
        h21-agnostic total at this h11.
    """
    if h21 is not None:
        n = _KS_PAIR.get((h11, h21), 0)
        return {"h11": h11, "h21": h21, "count": n, "exists": n > 0}

    n = _KS_H11.get(h11, 0)
    return {"h11": h11, "count": n, "exists": n > 0}

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
import time

import cytools

# local imports
from cytools_agent.tools._synonyms import forgive_kwargs

# keys/attrs that mean "the id" -- so a model that treats a fetched id as a
# record (polytopes[0]['id'], .ks_ind) gets the id back instead of an opaque
# "string indices must be integers".
_ID_KEYS = {"id", "ks_ind", "ks", "polytope_id", "poly_id", "pid", "name"}


# human-read
class _PolytopeId(str):
    """A polytope id string that also answers dict-/attribute-style id access by
    returning ITSELF -- fetch_polytopes returns these, so polytopes[0]['id'] (or
    .ks_ind) just works. It is a str everywhere else; character indexing and
    slicing are unchanged. A non-id string key points the model at the data tool."""

    def __getitem__(self, key):
        if isinstance(key, str):
            if key.lower() in _ID_KEYS:
                return self
            raise KeyError(
                f"a fetched element IS the polytope id string itself, not a "
                f"record -- ['{key}'] is not available. For polytope data call "
                f"get_polytope_info(<id>)['{key}'] (or get_cy_info(...)).")
        return str.__getitem__(self, key)     # int / slice: normal str indexing

    def __getattr__(self, name):
        if name in _ID_KEYS:
            return self
        raise AttributeError(name)


def _ids(seq, h11=None, h21=None, favorable=None):
    """Wrap fetched ids so dict-/attr-style id access is forgiven. If the query
    matched NOTHING, raise a pointed error (instead of returning [] that the
    caller then explodes on with polytope_ids[0] -> IndexError)."""
    seq = list(seq)
    if not seq:
        cond = "h11=%s" % h11 + (", h21=%s" % h21 if h21 is not None else "") \
            + (", favorable=%s" % favorable if favorable is not None else "")
        raise ValueError(
            f"fetch_polytopes found NO polytopes for {cond}. Check ks_stats(h11) "
            f"for which (h11, h21) exist and their counts -- h11 must be >=1 and "
            f"present in the database, and over-constraining h21/favorable can "
            f"also yield none.")
    return [_PolytopeId(s) for s in seq]


# human-read
class _InfoDict(dict):
    """A result dict that, on a missing key, says which keys ARE available --
    so a model guessing a field (info['vertices']) gets a pointed error naming
    the real keys instead of a bare KeyError it has to .keys() its way out of."""

    def __missing__(self, key):
        raise KeyError(
            f"'{key}' is not a field in this result. Available keys: "
            f"{', '.join(self.keys())}.")


_CACHE   = {} # ks_ind -> vertices (list[list[int]])
_CIDS    = {} # ks_ind -> content id: sha256 of the affine normal form, so the
              # name is invariant under GL(n,Z) x Z^n relabeling -- a durable,
              # database-independent identity. Computed lazily (~100 ms each,
              # PALP-bound) and memoized here + persisted with the cache.
_FETCHED = {} # (h11, h21) -> {"count": int, "complete": bool, "ids": [...]}.
              # "ids" is the DB-ORDER id list this query has fetched (a
              # contiguous prefix from index 0). Serving a query from the
              # cache MUST slice this list -- reconstructing the order by
              # sorting every cached id broke "the first N": an (h11, h21)-
              # specific or favorable fetch caches ids BEYOND the broad
              # prefix, and they sort into the middle (measured: pm_corpus
              # id9's "first 100 at h11=4" silently became a different 100
              # as the day's runs polluted the cache, flipping its answer).

# Optional on-disk persistence of the (real) fetched polytopes, so repeated
# runs do not re-hit the Kreuzer-Skarke database. A DEVELOPMENT feature,
# DISABLED by default: it grows without bound (measured: 33 MB in a day of
# eval work) and end users should not accumulate that silently. Opt in by
# setting CYTOOLS_AGENT_KS_CACHE to a file path -- the eval harnesses do
# (they re-run the same queries constantly and the savings are large). The
# in-process memory cache always works either way, and the politeness guards
# (_ks_guard) bound the database load of cacheless sessions.
_REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DISK = os.environ.get("CYTOOLS_AGENT_KS_CACHE", "")

# Optional READ-ONLY trusted base layer. Loaded first and NEVER written, so no
# run -- including an untrusted model-driven eval -- can poison the data the
# old corpus depends on (regression protection). The writable overlay (_DISK)
# then holds only NEWLY discovered polytopes; reads merge base+overlay, writes
# touch the overlay only. This is how we BOTH protect the trusted census AND
# keep exposing the ability to discover new polytopes.
_BASE = os.environ.get("CYTOOLS_AGENT_KS_BASE", "")
_BASE_KEYS = set()        # ks_inds loaded from the base -> excluded from writes
_BASE_FETCHED = set()     # (h11, h21) fetch keys loaded from the base


# Cache format version. Bumped when the id-assignment semantics change; the
# loader and the merge refuse data from any other version, so a long-running
# process with older code in memory (a stale Jupyter kernel, an old MCP
# server) can clobber the file but can never silently poison a new process
# -- measured: a stale writer relabeled ids onto different geometry and
# flipped 8 corpus answers. v3 adds content ids ("cids").
_FORMAT = 3


# human-read
def _read_cache_file(path):
    """Read a cache file; return its dict, or None if absent/unreadable or
    from a different format era (other-era data is never merged)."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    return d if d.get("format") == _FORMAT else None


# human-read
def _parse_fetched(d):
    """Yield (key, record) for each well-formed fetched entry in a cache dict.
    Drops legacy entries with no recorded id order (the next query refetches)."""
    for k, v in d.get("fetched", {}).items():
        try:
            h11s, h21s = k.split(",")
            if "ids" not in v:
                continue
            yield (int(h11s), int(h21s) if h21s else None), v
        except (ValueError, TypeError):
            continue


# human-read
def _spot_check_ok():
    """Recompute a few stored content ids from the stored geometry; False on
    any mismatch (the file is corrupt -- trust none of it). ~1 s at 10 samples.
    Catches the measured ~17%-corruption incident with ~84% probability per
    load; the merge-side conflict detector covers ALL shared keys on save."""
    import random as _random
    import hashlib
    import numpy as np
    keys = sorted(set(_CIDS) & set(_CACHE))
    for ks in _random.sample(keys, k=min(10, len(keys))):
        p = cytools.Polytope(_CACHE[ks])
        nf = np.ascontiguousarray(
            np.array(p.normal_form(affine_transform=True), dtype=np.int64))
        if hashlib.sha256(nf.tobytes()).hexdigest()[:12] != _CIDS[ks]:
            return False
    return True


# human-read
def _prune_dangling():
    """Drop any fetched-record whose id list references geometry the cache does
    not hold. A partial/corrupt write (observed: (2,74) listed ind-0 but only
    ind-1 was cached) otherwise makes _cache_can_serve answer True while
    get_polytope(ind-0) raises 'fewer than 1 ...'. Dropping the record forces a
    clean refetch instead of serving a dangling reference."""
    bad = [key for key, v in _FETCHED.items()
           if any(i not in _CACHE for i in v.get("ids", []))]
    for key in bad:
        del _FETCHED[key]


# human-read
def _load_disk_cache():
    # the read-only trusted base FIRST (its keys are excluded from writes), so
    # it can never be overwritten by the overlay or by any run
    base = _read_cache_file(_BASE)
    if base is not None:
        _CACHE.update(base.get("cache", {}))
        _CIDS.update(base.get("cids", {}))
        _BASE_KEYS.update(base.get("cache", {}))
        for key, v in _parse_fetched(base):
            _FETCHED[key] = v
            _BASE_FETCHED.add(key)
    # then the writable overlay -- it adds only what the base does not already
    # have (base entries are authoritative, so the trusted census stands)
    overlay = _read_cache_file(_DISK)
    if overlay is not None:
        _CACHE.update({k: v for k, v in overlay.get("cache", {}).items()
                       if k not in _CACHE})
        _CIDS.update({k: v for k, v in overlay.get("cids", {}).items()
                      if k not in _CIDS})
        for key, v in _parse_fetched(overlay):
            if key not in _FETCHED:
                _FETCHED[key] = v
    if not _spot_check_ok():
        print("WARNING: cached geometry fails its content-id check -- "
              "discarding the disk cache as corrupt.")
        _CACHE.clear()
        _CIDS.clear()
        _FETCHED.clear()
        _BASE_KEYS.clear()
        _BASE_FETCHED.clear()
        return
    _prune_dangling()


# human-read
def _save_disk_cache():
    """Persist the cache, MERGING with what is on disk first -- several
    processes (an agent session + a research script) share this file, and a
    blind rewrite would discard whichever fetches the other process made
    since our load. Keep whichever side knows a LONGER prefix per query."""
    if not _DISK:
        return
    try:
        if os.path.exists(_DISK):
            try:
                with open(_DISK) as f:
                    d = json.load(f)
                if d.get("format") != _FORMAT:
                    d = {}          # never merge other-era data
                # relabeling detector: the same ks_ind with different
                # geometry on the two sides means somebody's labels are
                # wrong -- drop that entry entirely (forces a clean refetch)
                # rather than guessing which side to trust
                disk_cache = d.get("cache", {})
                for k in set(disk_cache) & set(_CACHE):
                    if (sorted(map(tuple, disk_cache[k]))
                            != sorted(map(tuple, _CACHE[k]))):
                        print(f"WARNING: conflicting geometry for {k} "
                              f"between processes -- dropping it from the "
                              f"shared cache.")
                        del _CACHE[k]
                        del disk_cache[k]
                        _CIDS.pop(k, None)
                        d.get("cids", {}).pop(k, None)
                _CACHE.update({k: v for k, v in disk_cache.items()
                               if k not in _CACHE})
                _CIDS.update({k: v for k, v in d.get("cids", {}).items()
                              if k not in _CIDS})
                for k, v in d.get("fetched", {}).items():
                    try:
                        h11s, h21s = k.split(",")
                        key = (int(h11s), int(h21s) if h21s else None)
                    except (ValueError, TypeError):
                        continue
                    ours = _FETCHED.get(key)
                    if "ids" in v and (not ours or
                                       len(v["ids"]) > len(ours.get("ids", []))
                                       or (v.get("complete")
                                           and not ours.get("complete"))):
                        _FETCHED[key] = v
            except (OSError, ValueError):
                pass
        os.makedirs(os.path.dirname(_DISK), exist_ok=True)
        # write the OVERLAY only: never persist base-layer entries (they live
        # in the read-only base file and must not be duplicated/mutated here)
        cache_out = {k: v for k, v in _CACHE.items() if k not in _BASE_KEYS}
        cids_out = {k: v for k, v in _CIDS.items() if k not in _BASE_KEYS}
        fetched = {f"{h11},{'' if h21 is None else h21}": v
                   for (h11, h21), v in _FETCHED.items()
                   if (h11, h21) not in _BASE_FETCHED}
        with open(_DISK, "w") as f:
            json.dump({"format": _FORMAT, "cache": cache_out,
                       "cids": cids_out, "fetched": fetched}, f)
    except OSError:
        pass

# Kreuzer-Skarke polytope counts for the full 4d database
# (calabi-yau-data/polytopes-4d on HuggingFace; 473,800,776 polytopes total).
# ks_counts.json holds both by_pair (h11, h21) and by_h11 (h21-agnostic) counts.
# ---------------------------------------------------------------------------
with open(os.path.join(os.path.dirname(__file__), "ks_counts.json")) as _f:
    _KS = json.load(_f)
_KS_PAIR = {tuple(int(x) for x in k.split(",")): v
            for k, v in _KS["by_pair"].items()}
_KS_H11 = {int(k): v for k, v in _KS["by_h11"].items()}

_load_disk_cache()   # serve prior real fetches from disk, sparing the KS DB

# human-read
def _h11_of(ks):
    """h11 parsed from an id string, else None (cheap cost-model feature)."""
    try:
        return int(str(ks).split("_")[0].split("-")[1])
    except (IndexError, ValueError):
        return None


# human-read
def content_id(p_or_ks) -> str:
    """Durable content-addressed identity: the first 12 hex of sha256 over
    the polytope's AFFINE NORMAL FORM -- invariant under GL(n,Z) lattice
    changes and translations, so the same abstract polytope gets the same id
    on any machine, from any database, in any embedding. Memoized per ks_ind
    (the normal form costs ~100 ms, PALP-bound)."""
    import hashlib
    import numpy as np
    ks = p_or_ks if isinstance(p_or_ks, str) else None
    if ks is not None and ks in _CIDS:
        return _CIDS[ks]
    p = get_polytope(p_or_ks)
    nf = np.ascontiguousarray(
        np.array(p.normal_form(affine_transform=True), dtype=np.int64))
    cid = hashlib.sha256(nf.tobytes()).hexdigest()[:12]
    if ks is not None:
        _CIDS[ks] = cid
        # persist in batches -- a full-file save per cid would rewrite a
        # multi-MB file once per polytope in a sweep
        if len(_CIDS) % 25 == 0:
            _save_disk_cache()
    return cid


# model-read (exposed in the run_python namespace)
@forgive_kwargs
def get_polytope(ks_ind: str | cytools.Polytope) -> cytools.Polytope:
    """Reconstruct the Polytope for a ks_ind, fetching it on demand if the id
    is well-formed but not yet cached. An already-built Polytope passes
    through, so tools accept either an id or a Polytope."""
    if isinstance(ks_ind, cytools.Polytope):
        return ks_ind
    if not isinstance(ks_ind, str):
        raise TypeError(
            f"ks_ind must be a polytope id string of the form "
            f"'h11-X_h21-Y_ind-Z' (or a Polytope), not "
            f"{type(ks_ind).__name__} {ks_ind!r}. Get ids from "
            f"fetch_polytopes(limit, h11)."
        )
    if ks_ind not in _CACHE:
        _autofetch(ks_ind)
    return cytools.Polytope(_CACHE[ks_ind])

# human-read
def _cache_can_serve(h11: int, h21: int | None, limit: int) -> bool:
    """True if the cache already holds the first `limit` of this query (a
    recorded DB-order prefix long enough, or known-complete)."""
    exact = _FETCHED.get((h11, h21))
    if exact and (exact["complete"] or len(exact.get("ids", [])) >= limit):
        return True

    # a fully-exhausted (h11-only) fetch determines every (h11, h21) subquery
    # (conservative: this can be done even if broad["complete"]==False, but
    #  that requires more careful tracking of #polys per h21)
    if h21 is not None:
        broad = _FETCHED.get((h11, None))
        if broad and broad["complete"]:
            return True

    return False

# human-read
def _get_cached_ks_inds(h11: int, h21: int | None) -> list[str]:
    """Cached ids for this query IN DATABASE ORDER -- the recorded fetch
    prefix when one exists (immune to cache pollution from other queries),
    falling back to the exact-(h11, h21) group sorted by ind (within one
    group, ind IS the database order)."""
    exact = _FETCHED.get((h11, h21))
    if exact and exact.get("ids"):
        return list(exact["ids"])
    if h21 is None:
        broad = _FETCHED.get((h11, None))
        return list(broad["ids"]) if broad and broad.get("ids") else []
    matches = []
    for ks_ind in _CACHE:
        _h11, _h21, ind = (int(p.split("-")[1]) for p in ks_ind.split("_"))
        if _h11 == h11 and _h21 == h21:
            matches.append((ind, ks_ind))
    matches.sort()
    return [ks_ind for _, ks_ind in matches]

# human-read
def _filter_favorable(ks_inds: list[str], favorable: bool | None) -> list[str]:
    """Keep ids whose N-favorability matches `favorable` (None keeps all)."""
    if favorable is None:
        return ks_inds
    return [i for i in ks_inds
            if get_polytope(i).is_favorable(lattice="N") == favorable]

# Politeness guardrail for the shared Kreuzer-Skarke database. Cache-served
# queries are free; queries that REALLY hit the database are (a) spaced by a
# minimum delay, (b) capped in size, and (c) capped in number per process --
# so neither an agent loop nor a careless human script (observed: a 250-query
# descending h11 sweep) can hammer the upstream source. All env-tunable.
_KS_MIN_INTERVAL = float(os.environ.get("CYTOOLS_KS_MIN_INTERVAL", "1.5"))
_KS_BUDGET = int(os.environ.get("CYTOOLS_KS_BUDGET", "40"))
_KS_MAX_LIMIT = int(os.environ.get("CYTOOLS_KS_MAX_LIMIT", "5000"))
_ks_queries = 0      # real DB queries this process
_ks_last = 0.0


# human-read
def ks_query_count() -> int:
    """Real (non-cache) KS-database queries issued by this process."""
    return _ks_queries


# human-read
def _ks_guard(h11, h21, limit):
    """Enforce budget + size cap + spacing before a REAL database query."""
    global _ks_queries, _ks_last
    if limit > _KS_MAX_LIMIT:
        raise ValueError(
            f"refusing a single database query for {limit} polytopes (cap "
            f"{_KS_MAX_LIMIT}). Check ks_stats(h11[, h21]) and narrow the "
            f"request, or work on a sample.")
    if _ks_queries >= _KS_BUDGET:
        raise RuntimeError(
            f"KS-database query budget exhausted ({_KS_BUDGET} real queries "
            f"this session) -- the shared database must not be hammered. "
            f"Plan with ks_stats (free), reuse already-fetched polytopes "
            f"(cached queries are free), or batch several h11 into FEWER "
            f"queries. If genuinely needed, raise CYTOOLS_KS_BUDGET.")
    wait = _KS_MIN_INTERVAL - (time.monotonic() - _ks_last)
    if wait > 0:
        time.sleep(wait)
    _ks_queries += 1
    _ks_last = time.monotonic()


# human-read
def _ensure_cached(h11: int, h21: int | None, limit: int) -> None:
    """Ensure the first `limit` of this query are cached (fetch if not)."""
    if _cache_can_serve(h11, h21, limit):
        return

    _ks_guard(h11, h21, limit)
    from cytools_agent.tools import costs
    _t0 = time.monotonic()
    polys = cytools.fetch_polytopes(
        h11=h11, h21=h21, limit=limit, dim=4, lattice="N", as_list=True
    )
    costs.record("ks_fetch", time.monotonic() - _t0, h11=h11,
                 limit=limit, n_returned=len(polys))

    # polytopes arrive in lexicographic (h11, h21, ind) order, so ind counts
    # within each (h11, h21) group, reset when the group changes. The ordered
    # id list IS the database order for this query -- record it, so serving
    # from the cache never has to reconstruct (and corrupt) it by sorting.
    prev_group, ind = None, 0
    ordered = []
    for p in polys:
        _h11, _h21 = int(p.h11(lattice="N")), int(p.h21(lattice="N"))
        ind = ind + 1 if (_h11, _h21) == prev_group else 0
        prev_group = (_h11, _h21)
        ks = f"h11-{_h11}_h21-{_h21}_ind-{ind}"
        _CACHE[ks] = p.vertices().tolist()
        ordered.append(ks)

    n = len(polys)
    prev = _FETCHED.get((h11, h21))
    if prev and len(prev.get("ids", [])) >= n:
        ordered = prev["ids"]           # keep the longer known prefix
    _FETCHED[(h11, h21)] = {
        "count": max(n, prev["count"] if prev else 0),
        "complete": bool(prev and prev["complete"]) or (n < limit),
        "ids": ordered,
    }
    _save_disk_cache()   # persist the real fetch so reruns skip the KS DB

# human-read
def _autofetch(ks_ind: str) -> None:
    """Fetch a well-formed but uncached id on demand, so the model can refer to
    a polytope by id without a prior explicit fetch."""
    try:
        h11, h21, ind = (int(p.split("-")[1]) for p in ks_ind.split("_"))
    except (ValueError, IndexError):
        raise KeyError(
            f"{ks_ind!r} is not a valid polytope id "
            "(expected 'h11-X_h21-Y_ind-Z')"
        )
    _ensure_cached(h11, h21, ind + 1)
    if ks_ind not in _CACHE:
        raise KeyError(
            f"{ks_ind!r} not found: there are fewer than {ind + 1} polytopes "
            f"at h11={h11}, h21={h21}"
        )

# model-read
@forgive_kwargs
def fetch_polytopes(limit: int, h11: int, h21: int | None = None,
                    favorable: bool | None = None) -> list[str]:
    """
    Fetch 4D reflexive polytopes from the Kreuzer-Skarke database.

    Each returned id has the form "h11-X_h21-Y_ind-Z", where Z is the position
    of the polytope within the (h11, h21) group.

    `limit` is REQUIRED. Do NOT guess it: first call ks_stats(h11[, h21]) to
    see how many polytopes exist, then pass that count (the KS database is
    large, so an arbitrary high limit can fetch thousands). For all FAVORABLE
    ones, pass favorable=True with limit set to that count -- you get fewer
    only if the database runs out.

    The database is a SHARED academic resource: real (non-cached) queries are
    rate-limited and budgeted per session. Re-fetching already-fetched
    polytopes is free (cached); prefer FEW, well-planned queries over a query
    per loop iteration.

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
        return _ids(_get_cached_ks_inds(h11, h21)[:limit], h11, h21)

    # favorable: scan deeper into the (h11, h21) list until `limit` matches are
    # found (the first favorable one may be well past index 0), or the DB ends
    scan = max(limit, 10)
    while True:
        _ensure_cached(h11, h21, scan)
        ids = _get_cached_ks_inds(h11, h21)
        fav = _filter_favorable(ids[:scan], favorable)
        if len(fav) >= limit or len(ids) < scan:
            return _ids(fav[:limit], h11, h21, favorable)
        scan *= 2

# model-read
@forgive_kwargs
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
        h11, h21, euler_characteristic (= 2*(h11-h21)), favorable_N,
        favorable_M, is_trilayer, automorphism_order, n_points,
        n_points_interior_to_facets, n_vertices, n_rigid_divisors (prime toric
        divisors whose dual face has no interior points), genera_2face (the
        genus of each 2-face, sorted descending; sum/max are common asks), and
        facedim_to_nfaces (a dict from face dimension d to HOW MANY d-faces
        there are; in 4d the 3-faces are the facets). A field that is not
        defined for this polytope (e.g. Hodge numbers of a 2d subpolytope) is
        OMITTED rather than raising; dim is always present.
    """
    from cytools_agent.tools import costs
    p = get_polytope(ks_ind)
    _t0 = time.monotonic()

    # graceful degradation: the tool also accepts raw Polytope objects (e.g.
    # a 2d reflexive subpolytope), for which several fields are undefined --
    # Hodge numbers and the CY-flavored fields are 4d notions, automorphisms
    # need full dimension. Compute what is meaningful, omit the rest; the
    # _InfoDict missing-key error names what IS available.
    def opt(compute):
        try:
            return compute()
        except Exception:
            return None

    # dimension-generic fields: meaningful for any lattice polytope.
    # content_id appears only when already memoized -- computing it costs
    # ~100 ms (PALP), too much to add to every info call; ask explicitly via
    # content_id(ks_ind) when needed.
    fields = {
        "dim": int(p.dim()),
        "content_id": (_CIDS.get(ks_ind)
                       if isinstance(ks_ind, str) else None),
        "n_points": len(p.points()),
        "n_points_interior_to_facets": len(p.points_interior_to_facets()),
        "n_vertices": len(p.vertices()),
        "facedim_to_nfaces": {
            d: len(p.faces(d)) for d in range(p.dim() + 1)
        },
        "automorphism_order": opt(lambda: len(p.automorphisms())),
    }
    # CY-flavored fields: 4d notions ONLY. cytools evaluates some of these
    # off-domain without raising (a 2d polytope reports h11=0), so an
    # exception guard is not enough -- gate on the dimension explicitly.
    if p.dim() == 4:
        h11 = opt(lambda: int(p.h11(lattice="N")))
        h21 = opt(lambda: int(p.h21(lattice="N")))
        fields.update({
            "h11": h11,
            "h21": h21,
            "euler_characteristic": (2 * (h11 - h21)
                                     if h11 is not None and h21 is not None
                                     else None),
            "favorable_N": opt(lambda: bool(p.is_favorable(lattice="N"))),
            "favorable_M": opt(lambda: bool(p.is_favorable(lattice="M"))),
            "is_trilayer": opt(lambda: bool(p.is_trilayer())),
            "n_rigid_divisors": opt(lambda: _n_rigid_divisors(p)),
            "genera_2face": opt(lambda: _genera_2face(p)),
        })
    costs.record("get_polytope_info", time.monotonic() - _t0,
                 h11=_h11_of(ks_ind), n_points=fields["n_points"])
    return _InfoDict({k: v for k, v in fields.items() if v is not None})

# human-read
def _n_rigid_divisors(p: cytools.Polytope) -> int:
    """Count rigid prime toric divisors: points not interior to a facet (i.e.
    interior to a dim 0/1/2 face) whose dual face has no interior points."""
    return len([pt for d in (0, 1, 2) for f in p.faces(d)
                for pt in f.interior_points(as_indices=True)
                if len(f.dual_face().interior_points()) == 0])

# human-read
def _genera_2face(p: cytools.Polytope) -> list[int]:
    """Genus of each 2-face (= #interior points of the dual 1-face), sorted
    descending. The sum/max are common asks."""
    return sorted((len(f.interior_points()) for f in p.dual().faces(1)),
                  reverse=True)

# model-read
@forgive_kwargs
def ks_stats(h11: int | None = None, h21: int | None = None) -> dict:
    """
    Polytope counts in the Kreuzer-Skarke database of 4d reflexive polytopes.

    Use this to CHECK whether polytopes exist (and how many) at given Hodge
    numbers instead of guessing - the database spans h11 from 1 to 491.
    Call it with NO arguments for whole-database statistics (the total and
    the per-h11 census) -- this is local data, never a database query.

    Parameters
    ----------
    h11 : int, optional
        The Hodge number h11. Omit for whole-database statistics.
    h21 : int, optional
        The Hodge number h21. If omitted, counts over all h21 at this h11.

    Returns
    -------
    dict
        count and exists - for the exact (h11, h21) when h21 is given. When h21
        is omitted: count (h21-agnostic total at this h11) and h21_values (the
        sorted list of h21 that actually occur at this h11 -- iterate over this
        to visit every (h11, h21), do NOT assume a range). When BOTH are
        omitted: total, h11_min, h11_max, and count_by_h11 (the full
        {h11: count} census).
    """
    if h11 is None:
        return {"total": sum(_KS_H11.values()),
                "h11_min": min(_KS_H11), "h11_max": max(_KS_H11),
                "count_by_h11": {h: _KS_H11[h] for h in sorted(_KS_H11)}}
    if h21 is not None:
        n = _KS_PAIR.get((h11, h21), 0)
        return {"h11": h11, "h21": h21, "count": n, "exists": n > 0}

    n = _KS_H11.get(h11, 0)
    h21_values = sorted(b for (a, b) in _KS_PAIR if a == h11)
    return {"h11": h11, "count": n, "exists": n > 0,
            "h21_values": h21_values}

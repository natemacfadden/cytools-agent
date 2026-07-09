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
# Description:  Harness-side iteration and plotting: small models struggle to
#               write the paired-array loop and matplotlib code (FRICTION_LOG
#               capstone), so these tools absorb both. A/B gated on
#               CYTOOLS_MAP_TOOLS (import is side-effect-free without it).
# -----------------------------------------------------------------------------

# external imports
import difflib
import os
import signal
import time


# human-read
def env_flag(name: str, default: bool = True) -> bool:
    """Parse an on/off env flag: unset -> default; '0'/'false'/'no'/'off'
    (any case) -> False; anything else -> True."""
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in ("0", "false", "no", "off", "")

# local imports
from cytools_agent.tools import code as _code
from cytools_agent.tools._synonyms import forgive_kwargs

_PREVIEW = 5        # values shown per column in the returned preview
_MAX_ERRORS = 3     # distinct error messages reported

# kind synonyms the model reaches for -> the canonical plot kind
_KINDS = {
    "scatter": "scatter", "scatterplot": "scatter", "scatter plot": "scatter",
    "histogram": "histogram", "hist": "histogram", "distribution": "histogram",
    "line": "line", "plot": "line", "lineplot": "line",
    "bar": "bar", "barchart": "bar", "bar chart": "bar",
}


# human-read
def _analyze_plot(xv, yv, kind):
    """Facts about the plotted data the model cannot read off a figure:
    ranges, constant axes, correlation, outliers. Returned with the figure
    path so the answer can state the relationship, not just that a plot
    exists."""
    import math

    def stats1(vals, name):
        out = {f"{name}_min": min(vals), f"{name}_max": max(vals)}
        if len(set(vals)) == 1:
            out[f"{name}_is_constant"] = True
        return out

    a = {"n": len(xv), **stats1(xv, "x")}
    if yv is not None:
        a.update(stats1(yv, "y"))
        n = len(xv)
        if (n >= 3 and len(set(xv)) > 1 and len(set(yv)) > 1
                and kind in ("scatter", "line")):
            mx, my = sum(xv) / n, sum(yv) / n
            sxy = sum((x - mx) * (y - my) for x, y in zip(xv, yv))
            sxx = sum((x - mx) ** 2 for x in xv)
            syy = sum((y - my) ** 2 for y in yv)
            a["pearson_r"] = round(sxy / math.sqrt(sxx * syy), 3)
    # outliers on each axis: |z| > 3
    for name, vals in (("x", xv), ("y", yv or [])):
        if len(vals) >= 4 and len(set(vals)) > 1:
            m = sum(vals) / len(vals)
            sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
            outs = sorted({v for v in vals if sd and abs(v - m) / sd > 3})
            if outs:
                a[f"{name}_outliers"] = outs[:5]
    return a


# human-read
def _as_id_list(ks_inds):
    """Accept what the model actually passes: a list of ids (canonical), a
    single id string, or a dict holding the fetch result."""
    if isinstance(ks_inds, str):
        return [ks_inds]
    if isinstance(ks_inds, dict):           # e.g. someone passed {"ids": [...]}
        for v in ks_inds.values():
            if isinstance(v, (list, tuple)):
                return list(v)
    return list(ks_inds)


# human-read
def _as_named_exprs(expressions):
    """Normalize to {name: expression}: accept a dict (canonical), a single
    expression string (named 'values'), or a list of expressions (named
    expr_1, expr_2, ...)."""
    if isinstance(expressions, str):
        return {"values": expressions}
    if isinstance(expressions, dict):
        return {str(k): str(v) for k, v in expressions.items()}
    if isinstance(expressions, (list, tuple)):
        return {f"expr_{i + 1}": str(e) for i, e in enumerate(expressions)}
    raise TypeError(
        "expressions must be a dict {name: expression}, a single expression "
        "string, or a list of expressions.")


# model-read
@forgive_kwargs
def compute_for_each(ks_inds: list[str], expressions: dict | str) -> dict:
    """
    Evaluate one or more Python expressions FOR EACH polytope id and store the
    results as aligned lists in the run_python scratchpad. Use this instead of
    writing your own for-loop whenever a task needs a per-polytope quantity
    across many polytopes (counts, volumes, maxima, ...).

    Each expression is evaluated once per id with `ks_ind` bound to that id
    (all tools are callable inside it, e.g. get_polytope_info, get_heights,
    get_cy_info). An id where any expression raises is skipped entirely, so
    all stored lists stay aligned; errors are reported, not fatal.

    Example: paired data for "largest curve volume vs tadpole charge":
        compute_for_each(ids, {
          "tadpole": "(2 + get_polytope_info(ks_ind)['h11'] "
                     "+ get_polytope_info(ks_ind)['h21']) / 2",
          "max_curve_vol": "max(get_cy_info(ks_ind, "
                           "get_heights(ks_ind)['heights'][0], t='tip', "
                           "cone='toric')['curve_volumes'])"})
    then plot with make_plot(kind='scatter', x='tadpole', y='max_curve_vol'),
    or reduce in run_python (e.g. print(np.mean(max_curve_vol))).

    Parameters
    ----------
    ks_inds : list of str
        The polytope ids to iterate over (from fetch_polytopes).
    expressions : dict or str
        {name: expression} -- each `name` becomes a scratchpad list with one
        value per id. A single expression string is stored as "values".

    Returns
    -------
    dict
        n_requested, n_ok (ids that evaluated cleanly), stored (the scratchpad
        names now holding the aligned lists, including ok_ids), a short
        preview of each list, stats per numeric list (n/mean/min/max/sum --
        report THESE, do not recompute by eye), and any errors (first few,
        with a count).
    """
    ids = _as_id_list(ks_inds)
    n_requested = len(ids)
    exprs = _as_named_exprs(expressions)
    if not ids:
        raise ValueError("ks_inds is empty -- fetch ids first "
                         "(fetch_polytopes) and pass them in.")

    compiled = {}
    for name, expr in exprs.items():
        try:
            compiled[name] = compile(expr, f"<compute_for_each:{name}>", "eval")
        except SyntaxError as e:
            from cytools_agent.tools._examples import example as _ex
            raise ValueError(
                f"expression {name!r} is not a valid Python expression "
                f"({e}). Write ONE expression per name, e.g. "
                f"\"{_ex('expr_error')[1]}\".") from None

    cols = {name: [] for name in exprs}
    ok_ids, errors = [], []
    # stored aligned columns (numeric lists parallel to a stored ok_ids) are
    # also offered as id-keyed dicts, so an expression can reference a prior
    # turn's per-polytope value as e.g. ntfe_count[ks_ind] (impossible with a
    # bare list)
    prev_ids = _code._NS.get("ok_ids")
    by_id = {}
    if isinstance(prev_ids, (list, tuple)) and prev_ids:
        for name, val in _code._NS.items():
            if (name not in _code._PRELOADED and name != "ok_ids"
                    and isinstance(val, (list, tuple))
                    and len(val) == len(prev_ids)
                    and all(isinstance(e, (int, float)) for e in val)):
                by_id[name] = dict(zip(prev_ids, val))
    # stop before the run_python wall-clock alarm (or our own fallback budget)
    # would kill the whole call: partial aligned columns beat losing all work
    t0 = time.monotonic()
    alarm_left = signal.getitimer(signal.ITIMER_REAL)[0]
    budget = max((alarm_left - 10) if alarm_left
                 else float(os.environ.get("CYTOOLS_RUN_TIMEOUT", "150")
                            or 150) - 10, 5)
    partial_note = None
    for k, ks in enumerate(ids):
        if time.monotonic() - t0 > budget:
            partial_note = (
                f"TIME BUDGET HIT: computed the first {k} of {len(ids)} ids; "
                f"all stored lists are PARTIAL (aligned, ok_ids says which). "
                f"For full coverage pass fewer ids or cheaper expressions.")
            ids = ids[:k]
            break
        scope = dict(_code._NS)
        scope.update(by_id)                   # id-keyed view shadows the list
        scope.update(ks_ind=ks, ks=ks)        # forgive the `ks` shorthand
        try:
            row = {name: eval(c, scope) for name, c in compiled.items()}
        except Exception as e:
            if len(errors) < _MAX_ERRORS:
                msg = f"{ks}: {type(e).__name__}: {e}"
                # a NameError in the expression is usually the per-item variable
                # typed wrong (ks_inds for ks_ind) or a stored column misnamed;
                # point at the right name instead of the raw NameError
                if isinstance(e, NameError):
                    miss = getattr(e, "name", "") or ""
                    near = difflib.get_close_matches(
                        miss, ["ks_ind"] + [n for n in _code._NS
                                            if not n.startswith("_")],
                        n=1, cutoff=0.5)
                    msg += (" [in the expression the per-item id is `ks_ind` "
                            "(singular)"
                            + (f"; did you mean {near[0]!r}?" if near else "")
                            + "]")
                errors.append(msg)
            continue
        # keep the id object as-is: fetch ids are _PolytopeId (a str subclass
        # whose ['id']/.ks_ind access is forgiven); str(ks) stripped that
        ok_ids.append(ks if isinstance(ks, str) else str(ks))
        for name, val in row.items():
            cols[name].append(val)

    # store the aligned lists where run_python / make_plot can use them
    for name, vals in cols.items():
        _code._NS[name] = vals
    _code._NS["ok_ids"] = ok_ids

    n_err = len(ids) - len(ok_ids)
    out = {
        "n_requested": n_requested,
        "n_ok": len(ok_ids),
        "stored": list(cols) + ["ok_ids"],
        "preview": {name: vals[:_PREVIEW] for name, vals in cols.items()},
    }
    if partial_note:
        out["partial"] = partial_note
    # aggregates computed by the harness: the reductions a question usually
    # wants (mean/min/max/sum) arrive pre-computed and exactly right, so the
    # model reports them instead of eyeballing arithmetic from a preview
    stats = {}
    for name, vals in cols.items():
        if vals and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                        for v in vals):
            stats[name] = {
                "n": len(vals),
                "mean": round(sum(vals) / len(vals), 6),
                "min": min(vals),
                "max": max(vals),
                "sum": round(sum(vals), 6),
            }
    if stats:
        out["stats"] = stats
    if n_err:
        out["errors"] = errors + (
            [f"... {n_err - len(errors)} more"] if n_err > len(errors) else [])
        if not ok_ids:
            raise ValueError(
                "every id failed -- first errors: " + " | ".join(errors)
                + ". Fix the expression and re-call.")
    return out


# model-read
@forgive_kwargs
def make_plot(kind: str, x: str | list, y: str | list | None = None,
              color: str | list | None = None,
              xlabel: str = "", ylabel: str = "",
              title: str = "", bins: int | None = None,
              logx: bool = False, logy: bool = False) -> str:
    """
    Build and save a figure from data already in the scratchpad. Pass the NAME
    of a stored list (e.g. one made by compute_for_each) -- or a literal list
    -- for each axis; the figure is saved to disk automatically and its path
    returned. Use this instead of writing matplotlib code. Call it several
    times for several figures.

    Parameters
    ----------
    kind : str
        "scatter" (needs x and y), "histogram" (x only), "line", or "bar".
    x : str or list
        Scratchpad variable name (e.g. "tadpole") or a literal list of values.
    y : str or list, optional
        Second axis, as for x. Required for scatter/line/bar.
    color : str or list, optional
        A THIRD aligned column to color scatter points by (adds a colorbar)
        -- e.g. color='h11' on an x-y scatter shows how the relationship
        varies with h11. For a histogram, a CATEGORY column: one overlaid
        (legended) histogram per distinct value.
    xlabel, ylabel, title : str, optional
        Axis labels and title.
    bins : int, optional
        Histogram bin count.
    logx, logy : bool, optional
        Log-scale the axis (use when values span decades, e.g. volumes).

    Returns
    -------
    str
        Confirmation with the saved figure path, plus computed facts about
        the plotted data (ranges, constant axes, correlation, outliers,
        per-group counts) -- USE these to describe the relationship in your
        answer instead of guessing from the figure.
    """
    plt = _code.plt
    if plt is None:
        raise RuntimeError("matplotlib is not available here.")
    k = _KINDS.get(str(kind).strip().lower())
    if k is None:
        raise ValueError(f"kind must be one of scatter/histogram/line/bar, "
                         f"got {kind!r}.")
    # the model often passes color="none"/"None"/"" to mean "no color column";
    # treat those as no color rather than a (missing) stored-list name
    if isinstance(color, str) and color.strip().lower() in ("", "none"):
        color = None

    def resolve(v, axis):
        if isinstance(v, str):
            if v in _code._NS and isinstance(_code._NS[v], (list, tuple)):
                vals = list(_code._NS[v])
            else:
                lists = [n for n, val in _code._NS.items()
                         if n not in _code._PRELOADED
                         and isinstance(val, (list, tuple))]
                avail = ", ".join(lists) or "(none, run compute_for_each first)"
                raise ValueError(
                    f"{axis}={v!r} is not a stored list. List-valued scratchpad "
                    f"variables: {avail}.")
        else:
            vals = list(v)
        # a plot column must be one value per id; a nested column (the raw
        # per-item list stored by mistake) otherwise blows up later as an
        # opaque "unhashable type: list". Point at the fix: reduce to a scalar.
        bad = next((e for e in vals if isinstance(e, (list, tuple, dict))), None)
        if bad is not None:
            raise ValueError(
                f"{axis}={v!r} has a list per id (e.g. {bad!r}), not one value "
                f"per id. Reduce each id to a scalar in compute_for_each first "
                f"(e.g. min(...), max(...), len(...), sum(...)), then plot that.")
        return vals

    xv = resolve(x, "x")
    yv = resolve(y, "y") if y is not None else None
    cv = resolve(color, "color") if color is not None else None
    if k != "histogram" and yv is None:
        raise ValueError(f"a {k} plot needs both x and y.")
    for name, vals in (("y", yv), ("color", cv)):
        if vals is not None and len(xv) != len(vals):
            raise ValueError(
                f"x and {name} have different lengths ({len(xv)} vs "
                f"{len(vals)}) -- use lists stored by the SAME "
                f"compute_for_each call so they stay aligned.")

    clabel = color if isinstance(color, str) else "color"
    fig, ax = plt.subplots()
    if k == "scatter":
        if cv is not None:
            sc = ax.scatter(xv, yv, c=cv, cmap="viridis")
            fig.colorbar(sc, ax=ax, label=clabel)
        else:
            ax.scatter(xv, yv)
    elif k == "histogram":
        if cv is not None:           # category column -> overlaid histograms
            groups = sorted(set(cv), key=str)
            if len(groups) > 8:
                raise ValueError(
                    f"color={clabel!r} has {len(groups)} distinct values -- "
                    f"too many for overlaid histograms (max 8). Use a "
                    f"coarser category or drop color.")
            for g in groups:
                ax.hist([v for v, c in zip(xv, cv) if c == g],
                        bins=bins or "auto", alpha=0.6, label=str(g))
            ax.legend(title=clabel)
        else:
            ax.hist(xv, bins=bins or "auto")
        ax.set_ylabel(ylabel or "count")
    elif k == "line":
        ax.plot(xv, yv)
    elif k == "bar":
        ax.bar(range(len(yv)), yv)
        ax.set_xticks(range(len(xv)), [str(v) for v in xv])
    if logx:
        ax.set_xscale("log")
    if logy and k != "histogram":
        ax.set_yscale("log")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel and k != "histogram":
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    note = _code._save_open_figures()
    analysis = _analyze_plot(xv, yv, k)
    if cv is not None:
        groups = {}
        for c in cv:
            groups[c] = groups.get(c, 0) + 1
        analysis["color_groups"] = dict(sorted(groups.items(), key=str)[:8])
    return (f"figure built.{note}\n[data facts -- use these to DESCRIBE the "
            f"relationship in your answer: {analysis}]")


# model-read
@forgive_kwargs
def search_polytopes(condition: str, objective: str = "largest_h11",
                     helpers: dict | None = None,
                     h11_max: int | None = None, h11_min: int = 1,
                     per_level: int = 300, deep_per_level: int = 5000,
                     grid_step: int = 10, patience: int = 3,
                     max_queries: int = 20) -> dict:
    """
    Budget-aware SEARCH over h11 levels for polytopes satisfying a condition.
    Use this for questions like "the largest h11 such that some polytope
    satisfies X" -- do NOT write your own loop over h11 levels (the database
    is shared; this tool probes politely and stops early).

    Strategy: coarse probe along an h11 grid (checking a PREFIX of each
    level), then on the first hit a level-by-level refinement toward the
    objective until `patience` consecutive levels show nothing. Coverage is
    deliberately LOSSY (prefixes, not exhaustive levels) -- the result
    reports exactly what was checked, so treat absences at prefix-checked
    levels as evidence, not proof.

    Parameters
    ----------
    condition : str
        A Python expression evaluated once per polytope with `ks_ind` bound
        (tools callable inside, like compute_for_each), truthy when the
        polytope qualifies. May reference helper names (see `helpers`).
    objective : str, optional
        "largest_h11" (default), "smallest_h11", or "any" (first hit).
    helpers : dict, optional
        {name: expression} computed per polytope BEFORE the condition, each
        name then bound inside it -- e.g.
        helpers={"n_rigid": "get_polytope_info(ks_ind)['n_rigid_divisors']"},
        condition="n_rigid == 0".
    h11_max, h11_min : int, optional
        Search range (defaults: the database's full h11 range).
    per_level : int, optional
        Prefix size for coarse probing (default 300).
    deep_per_level : int, optional
        Prefix size during refinement (default 5000).
    grid_step : int, optional
        Coarse-probe spacing in h11 (default 10).
    patience : int, optional
        Stop refining after this many consecutive empty levels (default 3).
    max_queries : int, optional
        Soft cap on real database queries this call may spend (default 20).

    Returns
    -------
    dict
        found, best_h11, witness (a qualifying id), n_hits_at_best, coverage
        (per level: checked / total / hits / exhaustive flag), queries_used,
        and `note` spelling out the lossiness. Report the note's caveats.
    """
    from cytools_agent.tools import polytope as P
    try:
        cond = compile(condition, "<search_polytopes:condition>", "eval")
    except SyntaxError as e:
        raise ValueError(
            f"condition is not a valid Python expression ({e}). Write ONE "
            f"boolean expression over ks_ind.") from None
    helper_code = {}
    for name, expr in (helpers or {}).items():
        try:
            helper_code[name] = compile(str(expr),
                                        f"<search_polytopes:{name}>", "eval")
        except SyntaxError as e:
            raise ValueError(f"helper {name!r} is not a valid expression "
                             f"({e}).") from None
    if objective not in ("largest_h11", "smallest_h11", "any"):
        raise ValueError("objective must be largest_h11/smallest_h11/any, "
                         f"got {objective!r}")

    q0 = P.ks_query_count()
    all_h11 = sorted(h for h, n in P._KS_H11.items() if n > 0)
    lo = max(h11_min, all_h11[0])
    hi = min(h11_max or all_h11[-1], all_h11[-1])
    descending = objective in ("largest_h11", "any")

    coverage = {}

    def spent():
        return P.ks_query_count() - q0

    def check(h, cap):
        total = P._KS_H11.get(h, 0)
        if total == 0:
            return None
        ids = P.fetch_polytopes(min(total, cap), h)
        hits = []
        for ks in ids:
            scope = dict(_code._NS)
            scope.update(ks_ind=ks, ks=ks)
            try:
                for name, hc in helper_code.items():
                    scope[name] = eval(hc, scope)
                if eval(cond, scope):
                    hits.append(str(ks))
            except Exception as e:
                raise ValueError(
                    f"condition raised on {ks}: {type(e).__name__}: {e}. "
                    f"Fix the expression.") from None
        coverage[h] = {"checked": len(ids), "total": total,
                       "hits": len(hits),
                       "exhaustive": len(ids) >= total}
        return hits[0] if hits else None

    # phase 1: coarse probe
    grid = (range(hi, lo - 1, -grid_step) if descending
            else range(lo, hi + 1, grid_step))
    best, witness = None, None
    for h in grid:
        if spent() >= max_queries:
            break
        w = check(h, per_level)
        if w:
            best, witness = h, w
            break

    # phase 2: refine toward the objective (skip for "any")
    if best is not None and objective != "any":
        step = 1 if objective == "largest_h11" else -1
        h, empty = best + step, 0
        while empty < patience and lo <= h <= hi and spent() < max_queries:
            w = check(h, deep_per_level)
            if w:
                best, witness, empty = h, w, 0
            else:
                empty += 1
            h += step

    prefix_lvls = sorted(h for h, c in coverage.items()
                         if not c["exhaustive"] and c["hits"] == 0)
    note = ("Coverage is partial: each level was checked as a PREFIX in "
            "database order (lowest h21 first) unless marked exhaustive. "
            "Absence at prefix-checked levels "
            f"({prefix_lvls if prefix_lvls else 'none'}) is evidence, not "
            "proof. State the answer as 'confirmed' with these caveats.")
    return {
        "found": best is not None,
        "best_h11": best,
        "witness": witness,
        "n_hits_at_best": coverage.get(best, {}).get("hits"),
        "coverage": {h: coverage[h] for h in sorted(coverage)},
        "queries_used": spent(),
        "note": note,
    }


# default on: the map/iteration tools materially raised pass rates in early A/B
# testing. CYTOOLS_MAP_TOOLS=0 restores the baseline (no map tools).
MAP_TOOLS_ENABLED = env_flag("CYTOOLS_MAP_TOOLS", default=True)

# A/B gate: only when enabled do the tools enter the run_python namespace and
# the advertised tool list; the baseline arm stays byte-identical.
if MAP_TOOLS_ENABLED:
    for _fn in (compute_for_each, make_plot, search_polytopes):
        _code._NS[_fn.__name__] = _fn
        _code._PRELOADED.append(_fn.__name__)     # survive reset_namespace
        _code._TOOL_NAMES.append(_fn.__name__)    # named in error hints

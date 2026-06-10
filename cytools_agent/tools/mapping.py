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
# Description:  Harness-side iteration and plotting. The observed capability
#               wall for small models (FRICTION_LOG capstone) is authoring the
#               per-item loop that RETAINS paired arrays, and the matplotlib
#               code that consumes them. These tools absorb both into the
#               harness: the model supplies a ONE-ITEM expression
#               (compute_for_each) or names of stored arrays (make_plot); the
#               harness does the loop, error-skipping, alignment, and figure.
#
#               EXPERIMENTAL / A-B GATED: registered with the model only when
#               CYTOOLS_MAP_TOOLS is set, so eval arms differ only in tool
#               availability. Module import is side-effect-free without it.
# -----------------------------------------------------------------------------

# external imports
import os


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
    exprs = _as_named_exprs(expressions)
    if not ids:
        raise ValueError("ks_inds is empty -- fetch ids first "
                         "(fetch_polytopes) and pass them in.")

    compiled = {}
    for name, expr in exprs.items():
        try:
            compiled[name] = compile(expr, f"<compute_for_each:{name}>", "eval")
        except SyntaxError as e:
            raise ValueError(
                f"expression {name!r} is not a valid Python expression "
                f"({e}). Write ONE expression per name, e.g. "
                f"\"max(get_polytope_info(ks_ind)['genera_2face'])\".") from None

    cols = {name: [] for name in exprs}
    ok_ids, errors = [], []
    for ks in ids:
        scope = dict(_code._NS)
        scope.update(ks_ind=ks, ks=ks)        # forgive the `ks` shorthand
        try:
            row = {name: eval(c, scope) for name, c in compiled.items()}
        except Exception as e:
            if len(errors) < _MAX_ERRORS:
                errors.append(f"{ks}: {type(e).__name__}: {e}")
            continue
        ok_ids.append(str(ks))
        for name, val in row.items():
            cols[name].append(val)

    # store the aligned lists where run_python / make_plot can use them
    for name, vals in cols.items():
        _code._NS[name] = vals
    _code._NS["ok_ids"] = ok_ids

    n_err = len(ids) - len(ok_ids)
    out = {
        "n_requested": len(ids),
        "n_ok": len(ok_ids),
        "stored": list(cols) + ["ok_ids"],
        "preview": {name: vals[:_PREVIEW] for name, vals in cols.items()},
    }
    # aggregates computed BY THE HARNESS: the reductions a question usually
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
              xlabel: str = "", ylabel: str = "",
              title: str = "", bins: int | None = None) -> str:
    """
    Build and save a figure from data already in the scratchpad. Pass the NAME
    of a stored list (e.g. one made by compute_for_each) -- or a literal list
    -- for each axis; the figure is saved to disk automatically and its path
    returned. Use this instead of writing matplotlib code.

    Parameters
    ----------
    kind : str
        "scatter" (needs x and y), "histogram" (x only), "line", or "bar".
    x : str or list
        Scratchpad variable name (e.g. "tadpole") or a literal list of values.
    y : str or list, optional
        Second axis, as for x. Required for scatter/line/bar.
    xlabel, ylabel, title : str, optional
        Axis labels and title.
    bins : int, optional
        Histogram bin count.

    Returns
    -------
    str
        Confirmation with the saved figure path.
    """
    plt = _code.plt
    if plt is None:
        raise RuntimeError("matplotlib is not available here.")
    k = _KINDS.get(str(kind).strip().lower())
    if k is None:
        raise ValueError(f"kind must be one of scatter/histogram/line/bar, "
                         f"got {kind!r}.")

    def resolve(v, axis):
        if isinstance(v, str):
            if v in _code._NS and isinstance(_code._NS[v], (list, tuple)):
                return list(_code._NS[v])
            lists = [n for n, val in _code._NS.items()
                     if n not in _code._PRELOADED
                     and isinstance(val, (list, tuple))]
            avail = ", ".join(lists) or "(none -- run compute_for_each first)"
            raise ValueError(
                f"{axis}={v!r} is not a stored list. List-valued scratchpad "
                f"variables: {avail}.")
        return list(v)

    xv = resolve(x, "x")
    yv = resolve(y, "y") if y is not None else None
    if k != "histogram" and yv is None:
        raise ValueError(f"a {k} plot needs both x and y.")
    if yv is not None and len(xv) != len(yv):
        raise ValueError(f"x and y have different lengths ({len(xv)} vs "
                         f"{len(yv)}) -- use lists stored by the SAME "
                         f"compute_for_each call so they stay aligned.")

    fig, ax = plt.subplots()
    if k == "scatter":
        ax.scatter(xv, yv)
    elif k == "histogram":
        ax.hist(xv, bins=bins or "auto")
        ax.set_ylabel(ylabel or "count")
    elif k == "line":
        ax.plot(xv, yv)
    elif k == "bar":
        ax.bar(range(len(yv)), yv)
        ax.set_xticks(range(len(xv)), [str(v) for v in xv])
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel and k != "histogram":
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    note = _code._save_open_figures()
    return "figure built." + note


# DEFAULT ON since the 2026-06-10 A/B (orchestrator 0/12 -> 4-6/12; the only
# passing configuration). CYTOOLS_MAP_TOOLS=0 restores the baseline arm.
MAP_TOOLS_ENABLED = env_flag("CYTOOLS_MAP_TOOLS", default=True)

# A/B gate: only when enabled do the tools enter the run_python namespace and
# the advertised tool list -- the baseline arm stays byte-identical.
if MAP_TOOLS_ENABLED:
    for _fn in (compute_for_each, make_plot):
        _code._NS[_fn.__name__] = _fn
        _code._PRELOADED.append(_fn.__name__)     # survive reset_namespace
        _code._TOOL_NAMES.append(_fn.__name__)    # named in error hints

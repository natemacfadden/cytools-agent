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
# Description:  The typed pipeline: most research questions here are
#               fetch -> per-item map -> reduce -> (plot). Instead of walking
#               a weak engineer through those steps in free-form code, ONE
#               schema-constrained compile call asks the model to fill the
#               template's slots (filters, one-item expressions, reduce ops,
#               plot axes); the HARNESS executes the pipeline deterministically
#               and composes the answer from computed values. Anything that
#               does not fit (fits=false, validation failure, runtime failure)
#               falls back to the normal PM walk -- the pipeline is a fast
#               path, never a constraint on what can be asked.
#
#               A/B gated: CYTOOLS_PIPELINE (default off until validated).
#               Local-model scaffolding by design; the MCP / capable-model
#               path is untouched.
# -----------------------------------------------------------------------------

# external imports
import ast
import re

# local imports
from cytools_agent.tools import polytope
from cytools_agent.tools.glossary import (ALL_MARKERS, expected_by_term,
                                          glossary_context)
from cytools_agent.tools.mapping import (compute_for_each, env_flag,
                                         make_plot)
from cytools_agent.orchestrator.evidence import emit

# DEFAULT ON since the round-3 A/B (arm J2: best computed-correctness,
# 10/12 evidence; misfits fall back to the free-form walk harmlessly);
# CYTOOLS_PIPELINE=0 disables.
PIPELINE = env_flag("CYTOOLS_PIPELINE", default=True)

_OPS = ("mean", "min", "max", "sum", "count", "argmax", "argmin",
        "ids_where_positive")

# the compile call's decoding schema: the model can only emit a filled template
PIPELINE_FORMAT = {
    "type": "object",
    "properties": {
        "fits": {"type": "boolean"},
        "fetch": {"type": "object",
                  "properties": {"h11": {"type": ["integer", "array", "null"],
                                         "items": {"type": "integer"}},
                                 "h21": {"type": ["integer", "null"]},
                                 "limit": {"type": ["integer", "null"]},
                                 "favorable": {"type": ["boolean", "null"]},
                                 "use_stored": {"type": ["string", "null"]}},
                  "required": []},
        "map": {"type": "object",
                "minProperties": 1, "maxProperties": 3,
                "additionalProperties": {"type": "string"}},
        "reduce": {"type": "array", "maxItems": 4,
                   "items": {"type": "object",
                             "properties": {
                                 "name": {"type": "string"},
                                 "op": {"enum": list(_OPS)},
                                 "of": {"type": "string"}},
                             "required": ["name", "op", "of"]}},
        "plot": {"type": ["array", "null"], "maxItems": 4,
                 "items": {"type": "object",
                           "properties": {
                               "kind": {"enum": ["scatter", "histogram",
                                                 "line", "bar"]},
                               "x": {"type": "string"},
                               "y": {"type": ["string", "null"]},
                               "color": {"type": ["string", "null"]},
                               "logx": {"type": "boolean"},
                               "logy": {"type": "boolean"}},
                           "required": ["kind", "x"]}},
    },
    "required": ["fits", "fetch", "map", "reduce", "plot"],
}

_COMPILE_INSTRUCTIONS = (
    "Decide whether this request fits the pipeline 'fetch polytopes -> "
    "compute expression(s) once per polytope -> reduce -> optional plot'. "
    "If not, reply fits=false (everything else may be minimal). If it fits: "
    "fetch = the Hodge-number filters and how many polytopes -- `limit` is "
    "PER h11, and for 'at each h11 in a range/list' pass h11 as the LIST of "
    "every value (e.g. [2,3,4,5,6,7,8,9,10]; never just the first); map = 1-3 "
    "named one-item Python EXPRESSIONS, each evaluated once per polytope "
    "with `ks_ind` bound to its id (the tools below are callable in the "
    "expression); reduce = the requested aggregations over the named map "
    "columns (op argmax/argmin returns the polytope ID at the extreme); "
    "plot = a LIST of the requested figures over named columns (one entry "
    "per requested figure -- several are fine; a scatter may set color to a "
    "third column, a histogram may set color to a category column for "
    "overlays, logx/logy for values spanning decades), or null if none "
    "asked. "
    "Use the glossary recipes verbatim where they match the asked quantity. "
    "For 'how many polytopes have X' make an INDICATOR map column (an "
    "expression that is 1 when the condition holds else 0, e.g. "
    "\"1 if max(...) == 0 else 0\") and reduce it with op=sum; for 'WHICH "
    "polytopes have X' also reduce the same indicator with "
    "op=ids_where_positive (returns their ids). A stored column's "
    "per-polytope value can be used inside a map expression as "
    "column_name[ks_ind]. "
    "If the request refers to polytopes ALREADY FETCHED in this conversation "
    "(e.g. 'those', 'the same polytopes', 'them') and a stored id list is "
    "shown in the context, set fetch = {\"use_stored\": \"<that variable "
    "name>\"} instead of Hodge-number filters; plot and reduce may then also "
    "name columns STORED from previous turns (shown in the context) -- map "
    "only what is not already stored."
)


def _stored_numeric_lists():
    """Numeric list-valued scratchpad vars from previous turns -- legal plot/
    reduce columns alongside this turn's map columns (chat: 'scatter THOSE
    counts against h21' plots a stored column vs a fresh one)."""
    from cytools_agent.tools import code as _code
    return [n for n, v in _code._NS.items()
            if n not in _code._PRELOADED
            and isinstance(v, (list, tuple)) and v
            and all(isinstance(e, (int, float)) and not isinstance(e, bool)
                    for e in v)]


def _valid(spec):
    """Validate beyond what the schema can express; return a reason or ''."""
    if not spec.get("fits"):
        return "model judged the request does not fit the pipeline"
    cols = list(spec.get("map") or {}) + _stored_numeric_lists()
    for name, expr in (spec.get("map") or {}).items():
        try:
            ast.parse(expr, mode="eval")
        except SyntaxError as e:
            return f"map expression {name!r} is not a valid expression ({e})"
    for r in spec.get("reduce") or []:
        if r["of"] not in cols:
            return f"reduce {r['name']!r} references unknown column {r['of']!r}"
    plots = spec.get("plot")
    if isinstance(plots, dict):       # tolerate a single object
        plots = spec["plot"] = [plots]
    for p in plots or []:
        if p["kind"] == "histogram":
            p["y"] = None     # forgive a meaningless y ('count'/'frequency'):
                              # a histogram's y IS the count
        if p["x"] not in cols:
            return f"plot x={p['x']!r} is not a map column"
        if p.get("y") and p["y"] not in cols:
            return f"plot y={p['y']!r} is not a map column"
        if p.get("color") and p["color"] not in cols:
            p["color"] = None    # color is decoration -- forgive a bad one
                                 # (observed: 'polytope_id') instead of
                                 # failing an otherwise-sound spec
        if p["kind"] != "histogram" and not p.get("y"):
            return f"a {p['kind']} plot needs y"
    f = spec.get("fetch") or {}
    if f.get("use_stored"):
        from cytools_agent.tools import code as _code
        name = f["use_stored"]
        ids = _code._NS.get(name)
        if not (isinstance(ids, (list, tuple)) and ids
                and all(isinstance(i, str) for i in ids)):
            stored = [n for n, v in _code._NS.items()
                      if n not in _code._PRELOADED
                      and isinstance(v, (list, tuple)) and v
                      and all(isinstance(i, str) for i in v)]
            return (f"use_stored={name!r} is not a stored id list"
                    + (f" (id lists available: {', '.join(stored)})"
                       if stored else " (no id lists stored yet -- fetch)"))
        return ""
    h11s = f.get("h11")
    h11s = h11s if isinstance(h11s, list) else [h11s]
    if not h11s or not all(isinstance(h, int) and 1 <= h <= 491 for h in h11s):
        return f"fetch h11 {f.get('h11')!r} invalid"
    if not (f.get("limit") and 1 <= f["limit"]
            and f["limit"] * len(h11s) <= 2000):
        return (f"fetch limit {f.get('limit')!r} x {len(h11s)} h11 values "
                f"out of range")
    return ""


# range phrasing that a SCALAR-h11 spec cannot honor -- the observed failure
# was "at each h11 in [2,10]" compiled to h11=2 and confidently answering a
# narrower question than asked
_H11_RANGE_RE = re.compile(
    r"(each|every|all)\s+h11|h11\s*(in|from)\s*\[?\s*\d+\s*(,|to|-|\.\.)"
    r"|h11\s*=\s*\d+\s*,\s*\d+", re.I)


_PLOT_WORDS_RE = re.compile(r"\b(plot|scatter|histogram|chart|graph)\b", re.I)


def _plot_issue(spec, direct):
    """'' or a recompile instruction: the request asks for a figure but the
    spec has no plot entries (observed: chat follow-ups kept compiling
    plot=null because the y column lived in a previous turn)."""
    if _PLOT_WORDS_RE.search(direct) and not spec.get("plot"):
        return ("the request asks for a plot but the spec has plot=null -- "
                "add a plot entry; its x/y/color may be THIS turn's map "
                "columns or any stored column shown in the context")
    return ""


def _range_issue(spec, direct):
    """'' or a recompile instruction: the request sweeps h11 but the spec
    fetches only one value. (A use_stored spec inherits whatever spread the
    stored ids have, so it is exempt.)"""
    f = spec.get("fetch") or {}
    if f.get("use_stored"):
        return ""
    h11 = f.get("h11")
    if _H11_RANGE_RE.search(direct) and not (isinstance(h11, list)
                                             and len(h11) > 1):
        return ("the request asks for EACH h11 in a range, but fetch.h11 is "
                f"{h11!r} -- pass h11 as the full list of values")
    return ""


def _obs(evidence, intent, code, output):
    """Record one harness-executed pipeline stage as an evidence observation,
    so diagnostics/grading/replay see pipeline runs like any other."""
    import time as _time
    evidence.append({"intent": intent, "ran_code": code,
                     "received_output": str(output)[:2000],
                     "interpretation": "(pipeline: harness-executed)",
                     "valid_python": True, "round": 0, "t": _time.time(),
                     "kind": "pipeline"})


def _reduce(op, vals, ok_ids):
    if op == "mean":
        return round(sum(vals) / len(vals), 6)
    if op == "min":
        return min(vals)
    if op == "max":
        return max(vals)
    if op == "sum":
        return round(sum(vals), 6) if isinstance(sum(vals), float) \
            else sum(vals)
    if op == "count":
        return len(vals)
    if op == "argmax":
        return ok_ids[max(range(len(vals)), key=vals.__getitem__)]
    if op == "argmin":
        return ok_ids[min(range(len(vals)), key=vals.__getitem__)]
    if op == "ids_where_positive":     # with an indicator column: the ids
        return [i for i, v in zip(ok_ids, vals) if v > 0]
    raise ValueError(f"unknown op {op}")


def _spec_quantity_issues(spec, direct):
    """Per-TERM quantity lint on the compiled map expressions: for each
    glossary quantity the request names, its markers should appear in SOME
    expression -- a term whose markers are absent while the expressions use
    OTHER quantities' markers was miscomputed (observed: 'lattice points'
    compiled to cy_volume while automorphism_order was right, so a whole-spec
    check would have passed it). Returns a feedback string or ''."""
    exprs = " ".join((spec.get("map") or {}).values())
    toks = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", exprs))
    issues = []
    for term, markers in expected_by_term(direct).items():
        if toks & markers:
            continue
        foreign = toks & (ALL_MARKERS - markers)
        if foreign:
            issues.append(
                f"the request's {term!r} is computed via "
                f"{'/'.join(sorted(markers))}, but no expression uses it "
                f"(found {'/'.join(sorted(foreign))} instead)")
    return "; ".join(issues)


def compile_pipeline(pm, direct, cheatsheet, context=""):
    """One schema-constrained compile call -> the spec dict (unvalidated).
    `context` carries prior-turn information (previous questions/answers and
    the stored id lists) so follow-ups can resolve references."""
    gloss = glossary_context(direct) or ""
    user = ((f"{context}\n\n" if context else "")
            + f"REQUEST:\n{direct}\n\n{cheatsheet}"
            + (f"\n\n{gloss}" if gloss else ""))
    return pm._json(_COMPILE_INSTRUCTIONS, user, think=pm.plan_think,
                    label="PM.compile", schema=PIPELINE_FORMAT)


def run_pipeline(spec, evidence):
    """Execute a VALIDATED spec; return the composed answer string.
    Raises on any stage failure (caller falls back to the free-form walk)."""
    from cytools_agent.tools import code as _code
    f = spec["fetch"]
    if f.get("use_stored"):             # chat continuity: prior turn's ids
        ids = list(_code._NS[f["use_stored"]])
        h11s = sorted({int(i.split("_")[0].split("-")[1]) for i in ids})
        _obs(evidence, "pipeline fetch (reused)",
             f"ids = {f['use_stored']}   # {len(ids)} ids from a previous turn",
             f"{len(ids)} ids: {ids[:5]}...")
    else:
        h11s = f["h11"] if isinstance(f["h11"], list) else [f["h11"]]
        ids = []
        for h in h11s:                  # limit is PER h11
            ids += polytope.fetch_polytopes(
                limit=f["limit"], h11=h, h21=f.get("h21"),
                favorable=f.get("favorable"))
        _obs(evidence, "pipeline fetch",
             f"fetch_polytopes(limit={f['limit']}, h11={h11s}, "
             f"h21={f.get('h21')!r}, favorable={f.get('favorable')!r})",
             f"{len(ids)} ids: {list(ids[:5])}...")

    r = compute_for_each(ids, spec["map"])
    _obs(evidence, "pipeline map",
         f"compute_for_each(ids, {spec['map']!r})", r)
    if r["n_ok"] < max(2, r["n_requested"] // 2):
        raise RuntimeError(f"map failed on most items: {r.get('errors')}")

    # this turn's map columns, plus stored numeric columns from previous
    # turns (validated against the same union)
    cols = {name: _code._NS[name]
            for name in list(spec["map"]) + _stored_numeric_lists()
            if name in _code._NS}
    ok_ids = _code._NS["ok_ids"]

    h11_desc = (f"h11 in {h11s}" if len(h11s) > 1 else f"h11={h11s[0]}")
    parts = [f"Computed {', '.join(spec['map'])} for {r['n_ok']} of "
             f"{r['n_requested']} polytopes at {h11_desc}"
             + (f", h21={f['h21']}" if f.get("h21") is not None else "")
             + (" (favorable)" if f.get("favorable") else "") + "."]
    if not spec["reduce"]:
        # no aggregation asked: the per-item values ARE the deliverable --
        # show them (briefly) instead of only naming the columns
        for name in spec["map"]:
            vals = cols[name]
            shown = vals if len(vals) <= 40 else vals[:40]
            tail = " ..." if len(vals) > 40 else ""
            parts.append(f"{name}: {shown}{tail}")
    for red in spec["reduce"]:
        val = _reduce(red["op"], cols[red["of"]], ok_ids)
        _obs(evidence, f"pipeline reduce {red['name']}",
             f"{red['op']}({red['of']})", val)
        parts.append(f"{red['name']} ({red['op']} of {red['of']}): {val}.")

    for p in spec.get("plot") or []:
        note = make_plot(kind=p["kind"], x=p["x"], y=p.get("y"),
                         color=p.get("color"), logx=bool(p.get("logx")),
                         logy=bool(p.get("logy")),
                         xlabel=p["x"], ylabel=p.get("y") or "")
        _obs(evidence, "pipeline plot",
             f"make_plot(kind={p['kind']!r}, x={p['x']!r}, y={p.get('y')!r}, "
             f"color={p.get('color')!r})", note)
        parts.append(note.replace("figure built.", "Figure saved:").strip())
    return " ".join(parts)


def try_pipeline(pm, direct, evidence, cheatsheet, log, context="", raw=""):
    """The fast path: compile, validate (with ONE lint-guided recompile),
    execute. Returns the answer string, or None to fall back to the
    free-form walk (reason logged + emitted). The shape guards check the RAW
    question too -- translate may paraphrase away the very words ('scatter')
    they key on."""
    asked = f"{direct} {raw}".strip()
    spec, why = None, "compile failed"
    feedback = ""
    for _attempt in range(2):    # compile, then at most one guided recompile
        try:
            spec = compile_pipeline(pm, direct + feedback, cheatsheet,
                                    context=context)
        except Exception as e:
            emit("pipeline", fits=False, reason=f"compile error: {e}")
            return None
        why = _valid(spec)
        if not why:
            why = _range_issue(spec, asked)
        if not why:
            why = _plot_issue(spec, asked)
        if not why:
            why = _spec_quantity_issues(spec, direct)
            if why:
                why = "quantity lint: " + why
        if not why:
            break
        feedback = ("\n\n(Your previous pipeline had this problem -- fix "
                    f"exactly it: {why})")
    if why:
        log("[pipeline: falling back]", why)
        emit("pipeline", fits=False, reason=why, spec=spec)
        return None
    emit("pipeline", fits=True, spec=spec)
    log("[pipeline spec]", str(spec))
    try:
        return run_pipeline(spec, evidence)
    except Exception as e:
        log("[pipeline: runtime fallback]", f"{type(e).__name__}: {e}")
        emit("pipeline", fits=False, reason=f"runtime: {e}")
        return None

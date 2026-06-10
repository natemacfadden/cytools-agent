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

_OPS = ("mean", "min", "max", "sum", "count", "argmax", "argmin")

# the compile call's decoding schema: the model can only emit a filled template
PIPELINE_FORMAT = {
    "type": "object",
    "properties": {
        "fits": {"type": "boolean"},
        "fetch": {"type": "object",
                  "properties": {"h11": {"type": "integer"},
                                 "h21": {"type": ["integer", "null"]},
                                 "limit": {"type": "integer"},
                                 "favorable": {"type": ["boolean", "null"]}},
                  "required": ["h11", "limit"]},
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
        "plot": {"type": ["object", "null"],
                 "properties": {"kind": {"enum": ["scatter", "histogram",
                                                  "line", "bar"]},
                                "x": {"type": "string"},
                                "y": {"type": ["string", "null"]}},
                 "required": ["kind", "x"]},
    },
    "required": ["fits", "fetch", "map", "reduce", "plot"],
}

_COMPILE_INSTRUCTIONS = (
    "Decide whether this request fits the pipeline 'fetch polytopes -> "
    "compute expression(s) once per polytope -> reduce -> optional plot'. "
    "If not, reply fits=false (everything else may be minimal). If it fits: "
    "fetch = the Hodge-number filters and how many polytopes; map = 1-3 "
    "named one-item Python EXPRESSIONS, each evaluated once per polytope "
    "with `ks_ind` bound to its id (the tools below are callable in the "
    "expression); reduce = the requested aggregations over the named map "
    "columns (op argmax/argmin returns the polytope ID at the extreme); "
    "plot = the requested figure over named columns, or null if none asked. "
    "Use the glossary recipes verbatim where they match the asked quantity. "
    "For 'how many polytopes have X' make an INDICATOR map column (an "
    "expression that is 1 when the condition holds else 0, e.g. "
    "\"1 if max(...) == 0 else 0\") and reduce it with op=sum."
)


def _valid(spec):
    """Validate beyond what the schema can express; return a reason or ''."""
    if not spec.get("fits"):
        return "model judged the request does not fit the pipeline"
    cols = list(spec.get("map") or {})
    for name, expr in (spec.get("map") or {}).items():
        try:
            ast.parse(expr, mode="eval")
        except SyntaxError as e:
            return f"map expression {name!r} is not a valid expression ({e})"
    for r in spec.get("reduce") or []:
        if r["of"] not in cols:
            return f"reduce {r['name']!r} references unknown column {r['of']!r}"
    p = spec.get("plot")
    if p:
        if p["kind"] == "histogram":
            p["y"] = None     # forgive a meaningless y ('count'/'frequency'):
                              # a histogram's y IS the count
        if p["x"] not in cols:
            return f"plot x={p['x']!r} is not a map column"
        if p.get("y") and p["y"] not in cols:
            return f"plot y={p['y']!r} is not a map column"
        if p["kind"] != "histogram" and not p.get("y"):
            return f"a {p['kind']} plot needs y"
    f = spec.get("fetch") or {}
    if not (1 <= f.get("limit", 0) <= 2000):
        return f"fetch limit {f.get('limit')!r} out of range"
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


def compile_pipeline(pm, direct, cheatsheet):
    """One schema-constrained compile call -> the spec dict (unvalidated)."""
    gloss = glossary_context(direct) or ""
    user = f"REQUEST:\n{direct}\n\n{cheatsheet}" + (f"\n\n{gloss}" if gloss
                                                    else "")
    return pm._json(_COMPILE_INSTRUCTIONS, user, think=pm.plan_think,
                    label="PM.compile", schema=PIPELINE_FORMAT)


def run_pipeline(spec, evidence):
    """Execute a VALIDATED spec; return the composed answer string.
    Raises on any stage failure (caller falls back to the free-form walk)."""
    f = spec["fetch"]
    ids = polytope.fetch_polytopes(
        limit=f["limit"], h11=f["h11"], h21=f.get("h21"),
        favorable=f.get("favorable"))
    _obs(evidence, "pipeline fetch",
         f"fetch_polytopes(limit={f['limit']}, h11={f['h11']}, "
         f"h21={f.get('h21')!r}, favorable={f.get('favorable')!r})",
         f"{len(ids)} ids: {list(ids[:5])}...")

    r = compute_for_each(ids, spec["map"])
    _obs(evidence, "pipeline map",
         f"compute_for_each(ids, {spec['map']!r})", r)
    if r["n_ok"] < max(2, r["n_requested"] // 2):
        raise RuntimeError(f"map failed on most items: {r.get('errors')}")

    from cytools_agent.tools import code as _code
    cols = {name: _code._NS[name] for name in spec["map"]}
    ok_ids = _code._NS["ok_ids"]

    parts = [f"Computed {', '.join(spec['map'])} for {r['n_ok']} of "
             f"{r['n_requested']} polytopes at h11={f['h11']}"
             + (f", h21={f['h21']}" if f.get("h21") is not None else "")
             + (" (favorable)" if f.get("favorable") else "") + "."]
    for red in spec["reduce"]:
        val = _reduce(red["op"], cols[red["of"]], ok_ids)
        _obs(evidence, f"pipeline reduce {red['name']}",
             f"{red['op']}({red['of']})", val)
        parts.append(f"{red['name']} ({red['op']} of {red['of']}): {val}.")

    p = spec.get("plot")
    if p:
        note = make_plot(kind=p["kind"], x=p["x"], y=p.get("y"),
                         xlabel=p["x"], ylabel=p.get("y") or "")
        _obs(evidence, "pipeline plot",
             f"make_plot(kind={p['kind']!r}, x={p['x']!r}, y={p.get('y')!r})",
             note)
        parts.append(note.replace("figure built.", "Figure saved:").strip())
    return " ".join(parts)


def try_pipeline(pm, direct, evidence, cheatsheet, log):
    """The fast path: compile, validate (with ONE lint-guided recompile),
    execute. Returns the answer string, or None to fall back to the
    free-form walk (reason logged + emitted)."""
    spec, why = None, "compile failed"
    feedback = ""
    for _attempt in range(2):    # compile, then at most one guided recompile
        try:
            spec = compile_pipeline(pm, direct + feedback, cheatsheet)
        except Exception as e:
            emit("pipeline", fits=False, reason=f"compile error: {e}")
            return None
        why = _valid(spec)
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

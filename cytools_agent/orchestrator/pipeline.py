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
from cytools_agent.tools.mapping import env_flag
# the ledgered versions (cytools_agent.tools wraps them), so pipeline stages
# write backbone rows like any other caller
from cytools_agent.tools import (compute_for_each, fetch_polytopes,
                                 make_plot)
from cytools_agent.orchestrator.evidence import emit

# DEFAULT ON since the round-3 A/B (arm J2: best computed-correctness,
# 10/12 evidence; misfits fall back to the free-form walk harmlessly);
# CYTOOLS_PIPELINE=0 disables.
PIPELINE = env_flag("CYTOOLS_PIPELINE", default=True)

# the SHAPE B worked example's field is sampled per process (attractor
# flattening; see tools/_examples.py)
from cytools_agent.tools._examples import example as _ex
_EXAMPLE = _ex("search_compile")

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
        "search": {"type": ["object", "null"],
                   "properties": {
                       "map": {"type": ["object", "null"], "maxProperties": 3,
                               "additionalProperties": {"type": "string"}},
                       "condition": {"type": "string"},
                       "objective": {"enum": ["largest_h11", "smallest_h11",
                                              "any"]},
                       "h11_max": {"type": ["integer", "null"]},
                       "h11_min": {"type": ["integer", "null"]}},
                   "required": ["condition", "objective"]},
        "explain": {"type": ["object", "null"],
                    "properties": {
                        "kind": {"enum": ["concept", "capability"]},
                        "queries": {"type": "array", "maxItems": 5,
                                    "items": {"type": "string"}}},
                    "required": ["kind", "queries"]},
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
    "required": ["fits", "fetch", "map", "reduce", "plot", "search",
                 "explain"],
}

_COMPILE_INSTRUCTIONS = (
    "Decide whether this request fits one of three shapes. SHAPE C (explain) "
    "FIRST, but ONLY for a pure definition/how-it-works/capability question: "
    "'what does X mean', 'how does function X work', 'what can you do'. Fill "
    "`explain` and leave the rest minimal. Set explain.kind='concept' for a "
    "definition/how-to (explain.queries = the terms to look up, e.g. "
    "['favorable', 'Mori cone']) or explain.kind='capability' for 'what can "
    "you do / what tools are there' (queries may be empty). The harness "
    "answers from the glossary and real docstrings -- do NOT answer from your "
    "own knowledge. Do NOT use explain if the request asks you to DO "
    "something with polytopes (fetch, compute, find, plot, sample, count) or "
    "to find how one quantity RELATES TO / SCALES WITH / VARIES WITH another "
    "across the database -- those are compute requests (SHAPE A or B) even "
    "when they name a term you recognize. "
    "Otherwise it is a COMPUTE request fitting one of two shapes. SHAPE A "
    "(map): "
    "'fetch polytopes -> compute expression(s) once per polytope -> reduce "
    "-> optional plot'. SHAPE B (search): 'the LARGEST/SMALLEST h11 (or: "
    "does ANY polytope exist) such that some polytope satisfies a "
    "condition' -- then fill `search` INSTEAD: search.map = named per-"
    "polytope expressions over ks_ind built from the glossary recipe for "
    "the asked quantity, condition = a boolean expression over those NAMES "
    "(e.g. search.map = "
    + "{\"%s\": \"%s\"}, condition = \"%s\"" % _EXAMPLE
    + " -- BOTH parts are required, and yours must come from the quantity "
    "YOUR request names, via its glossary recipe), "
    "objective = largest_h11/smallest_h11/"
    "any; leave fetch/map/reduce/plot minimal (they are ignored). If "
    "neither shape fits, reply fits=false. For SHAPE A: "
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
    "For a question about the WHOLE database (every h11; total counts; "
    "'most common h11'; the distribution of polytope counts), set fetch.h11 "
    "= null: the harness then serves the local KS census -- built-in "
    "columns h11 and count (polytopes per h11 value) -- with no fetching; "
    "map/reduce/plot may use h11 and count (argmax of count = the most "
    "populous h11; sum of count = the database total; a plot of count vs "
    "h11 usually wants logy=true). "
    "If the request refers to polytopes ALREADY FETCHED in this conversation "
    "(e.g. 'those', 'the same polytopes', 'them') and a stored id list is "
    "shown in the context, set fetch = {\"use_stored\": \"<that variable "
    "name>\"} instead of Hodge-number filters; plot and reduce may then also "
    "name columns STORED from previous turns (shown in the context) -- map "
    "only what is not already stored."
)


# Known field names the model may use as bare variables in search
# conditions ("cy_volume <= 5"). These are not invented shorthands -- each
# has one obvious recipe -- so instead of rejecting the spec, the validator
# auto-inserts the helper (observed: 'minimum CY volume at the tip' compiled
# to a bare cy_volume and died; the intent was unambiguous).
_FIELD_BRIDGES = {
    **{f: f"get_polytope_info(ks_ind)['{f}']"
       for f in ("h11", "h21", "euler_characteristic", "favorable_N",
                 "favorable_M", "is_trilayer", "automorphism_order",
                 "n_points", "n_points_interior_to_facets", "n_vertices",
                 "n_rigid_divisors", "dim")},
    **{f: ("get_cy_info(ks_ind, get_heights(ks_ind)['heights'][0], "
           f"t='tip', cone='toric')['{f}']")
       for f in ("cy_volume", "curve_volumes", "divisor_volumes")},
    "ntfe_count": "get_heights(ks_ind)['shape'][0]",
    "genus_max": "max(get_polytope_info(ks_ind)['genera_2face'])",
}


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


def _valid_census(spec):
    """Validate a whole-database (fetch.h11=null) spec: it runs against the
    local KS census of per-h11 counts, never per polytope. Columns h11 and
    count are built in; map expressions run once per h11 value with h11 and
    count bound, so an expression over ks_ind cannot be honored."""
    import builtins
    spec["fetch"]["_census"] = True
    cols = ["h11", "count"]
    for name, expr in (spec.get("map") or {}).items():
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            return f"map expression {name!r} is not a valid expression ({e})"
        loose = sorted({n.id for n in ast.walk(tree)
                        if isinstance(n, ast.Name)
                        and n.id not in set(cols) | set(dir(builtins))})
        if loose:
            return (f"fetch.h11=null means the WHOLE database, served from "
                    f"the KS census of per-h11 counts -- map expressions may "
                    f"only use h11 and count, but {name!r} uses "
                    f"{', '.join(loose)}. Per-polytope computation over the "
                    f"whole database is infeasible: either rephrase over "
                    f"h11/count, or pass fetch.h11 as a list of specific "
                    f"values")
        cols.append(name)
    for r in spec.get("reduce") or []:
        if r["of"] not in cols:
            return (f"reduce {r['name']!r} references {r['of']!r}; census "
                    f"columns are {cols}")
    plots = spec.get("plot")
    if isinstance(plots, dict):
        plots = spec["plot"] = [plots]
    for p in plots or []:
        if p["kind"] == "histogram":
            p["y"] = None
        if p["x"] not in cols or (p.get("y") and p["y"] not in cols):
            return (f"census plot columns must be among {cols} "
                    f"(got x={p['x']!r}, y={p.get('y')!r})")
        if p.get("color") and p["color"] not in cols:
            p["color"] = None
        if p["kind"] != "histogram" and not p.get("y"):
            return f"a {p['kind']} plot needs y"
    return ""


def _valid(spec):
    """Validate beyond what the schema can express; return a reason or ''."""
    if not spec.get("fits"):
        return "model judged the request does not fit the pipeline"
    if spec.get("explain"):
        e = spec["explain"]
        if e.get("kind") not in ("concept", "capability"):
            return f"explain.kind must be concept/capability, got {e.get('kind')!r}"
        if e["kind"] == "concept" and not e.get("queries"):
            return ("explain.kind='concept' needs explain.queries -- the "
                    "terms/topics to look up (e.g. ['favorable'])")
        return ""
    if spec.get("search"):
        s = spec["search"]
        for name, expr in (s.get("map") or {}).items():
            try:
                ast.parse(str(expr), mode="eval")
            except SyntaxError as e:
                return f"search helper {name!r} is not a valid expression ({e})"
        try:
            tree = ast.parse(s.get("condition", ""), mode="eval")
        except SyntaxError as e:
            return f"search condition is not a valid expression ({e})"
        # names the condition uses must EXIST: defined helpers, preloaded
        # tools, or builtins. Catching a dangling shorthand here (observed:
        # condition 'max_2face_points <= 20' with map=None) gives the
        # recompile a fillable instruction instead of a runtime NameError.
        import builtins
        from cytools_agent.tools import code as _code
        known = (set(s.get("map") or {}) | set(_code._NS) | {"ks_ind", "ks"}
                 | set(dir(builtins)))
        loose = sorted({n.id for n in ast.walk(tree)
                        if isinstance(n, ast.Name) and n.id not in known})
        # auto-bridge: a loose name that is a known field gets its helper
        # inserted instead of a rejection -- the intent is unambiguous
        bridged = [n for n in loose if n in _FIELD_BRIDGES]
        if bridged:
            s["map"] = dict(s.get("map") or {})
            for n in bridged:
                s["map"][n] = _FIELD_BRIDGES[n]
            loose = [n for n in loose if n not in bridged]
        if loose:
            return ("search condition references undefined name(s) "
                    + ", ".join(loose) + " -- DEFINE each in search.map as "
                    "an expression over ks_ind (use the glossary recipe), "
                    "e.g. search.map = {\"" + loose[0] + "\": \"<expression "
                    "over ks_ind>\"}")
        return ""        # search mode: fetch/map/reduce/plot are ignored
    f0 = spec.get("fetch") or {}
    if f0.get("h11") is None and not f0.get("use_stored"):
        return _valid_census(spec)
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
    # forgive the two observed limit mis-fills instead of rejecting:
    # None ("grab me a polytope" -- no count stated) defaults small, and an
    # over-cap sweep is clamped to the politeness cap, with the partiality
    # surfaced in the answer (run_pipeline reads _limit_note)
    if not f.get("limit"):
        f["limit"] = 10
        f["_limit_note"] = "no count was stated; using the first 10"
    cap = 2000
    if f["limit"] * len(h11s) > cap:
        f["limit"] = max(1, cap // len(h11s))
        f["_limit_note"] = (f"requested more than the {cap}-polytope "
                            f"politeness cap; clamped to the first "
                            f"{f['limit']} per h11 -- results are PARTIAL")
    return ""


# range phrasing that a SCALAR-h11 spec cannot honor -- the observed failure
# was "at each h11 in [2,10]" compiled to h11=2 and confidently answering a
# narrower question than asked
_H11_RANGE_RE = re.compile(
    r"(each|every|all)\s+h11|h11\s*(in|from)\s*\[?\s*\d+\s*(,|to|-|\.\.)"
    r"|h11\s*=\s*\d+\s*,\s*\d+", re.I)


_PLOT_WORDS_RE = re.compile(r"\b(plot|scatter|histogram|chart|graph)\b", re.I)

# "the first polytope (at h11=X)" is ONE polytope; when no count is stated the
# limit defaults to 10 and the answer becomes a 10-item list instead of the
# single value asked for. A SINGULAR reference (no count, "polytope" not
# "polytopes") should fetch exactly one.
_SINGULAR_POLY_RE = re.compile(
    r"\bfirst\b[^.?]*\bpolytope\b|\bthe polytope\b|\bthis polytope\b", re.I)
# a PLURAL request is one that names a COUNT (>1) of polytopes -- "first 30",
# "50 polytopes". A bare "polytopes" must NOT count: it appears generically in
# the PM's translation ("fetch polytopes at h11=4 ...") and was silently
# blocking the singular->limit-1 fix whenever the translation pluralized,
# making single-polytope count questions flaky ([2],[86],[100],[17]).
_PLURAL_COUNT_RE = re.compile(r"\bfirst\s+\d+\b|\b\d+\s+polytopes\b", re.I)

# a POSITIONAL reference picks ONE polytope from the fetched batch by rank:
# "the second polytope at h11=4", "the 3rd polytope", "the polytope at index
# 15". The fetch/map/reduce spec has no positional op, so the harness selects
# the element deterministically -- no load on the weak model, which just
# computes the column as usual. The ordinal must DIRECTLY qualify "polytope"
# (only Hodge/KS descriptors may sit between it and the noun), so "the second
# Chern class" or "the second-largest" is never misread as a selection.
_ORDINALS = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4,
             "sixth": 5, "seventh": 6, "eighth": 7, "ninth": 8, "tenth": 9}
_SEL_FILLER = (r"(?:h[\^_]?\{?\d+\}?\s*=?\s*-?\d+|kreuzer[- ]skarke|ks|"
               r"reflexive|favorable|lattice|[,]|\s)*")
_SEL_INDEX_RE = re.compile(r"\bindex\s+(\d+)\b", re.I)
_SEL_ORD_NUM_RE = re.compile(
    rf"\b(\d+)(?:st|nd|rd|th)\s+{_SEL_FILLER}polytope\b", re.I)
_SEL_ORD_WORD_RE = re.compile(
    rf"\b({'|'.join(_ORDINALS)})\s+{_SEL_FILLER}polytope\b", re.I)


def _parse_select_index(asked):
    """0-based index of the polytope a positional reference names, else None.
    Bare 'first' is left to the singular-limit path (it already fetches index
    0); this handles 'index N' and ordinals >= second."""
    m = _SEL_INDEX_RE.search(asked)
    if m:
        return int(m.group(1))
    m = _SEL_ORD_NUM_RE.search(asked)
    if m and int(m.group(1)) >= 2:
        return int(m.group(1)) - 1
    m = _SEL_ORD_WORD_RE.search(asked)
    if m and _ORDINALS[m.group(1).lower()] >= 1:
        return _ORDINALS[m.group(1).lower()]
    return None


# words that mean the question wants a COMPUTED result (a number, extremum,
# count, list, or figure) -- an explain spec on any of these is the model
# taking the easy out (explaining a term instead of doing the work)
# Signals that a request wants WORK done, not a definition -- used only to
# reject an `explain` spec (the model naming a term it recognizes and dodging
# the actual task). Three families:
#   result asks   -- an extremum/count/list/figure
#   do-tasks      -- imperatives to fetch/compute something concrete
#   relationships -- "how does X relate to/scale with/vary with Y" across the
#                    database (empirical; an evidence answer beats a definition)
# Deliberately excludes bare "compute"/"calculate"/"make"/"mean" (a how-to
# "how do I compute X" and "what does it MEAN" are legit explain questions).
_COMPUTE_INTENT_RE = re.compile(
    r"\b(largest|smallest|maximum|minimum|\bmax\b|\bmin\b|how many|number of|"
    r"\bcount\b|average|median|distribution|plot|scatter|histogram|chart|"
    r"graph|list (all|the)|which polytope"
    r"|grab|fetch|give me|find me|get me|make me|generate"
    r"|relate|related|relationship|correlat\w*|scales? (with|as)|"
    r"scaling (with|as)|var(y|ies) (with|across)|depends? on)\b", re.I)


# A request that names a CONCRETE polytope to compute on -- a Hodge number, an
# id, or "the first polytope" -- is a computation/lookup, never a definition,
# even when phrased "what is the <quantity> of ...". Observed regression:
# "what is the CY volume of the first h11=3,h21=43 polytope" and "is the first
# polytope at h11=3 a trilayer polytope" both dumped an explain answer.
_CONCRETE_POLY_RE = re.compile(
    r"h\s*1?1\s*=|h\s*2?1\s*=|\bind[-\s]?\d|first\b[^.?]*\bpolytope|"
    r"\bthe polytope\b|\bthis polytope\b|polytope at h", re.I)


def _explain_issue(spec, asked):
    """'' or a recompile instruction: explain (concept OR capability) is only
    right for an ABSTRACT definition/how-to/'what can you do' question. If the
    request asks for a computed result (extremum/count/list/figure) OR names a
    concrete polytope to compute on, the model took the easy out -- steer it
    back to SHAPE A/B. Genuine definition questions ('what is favorability',
    'how does make_star work', 'what can you do') match neither pattern."""
    if not spec.get("explain"):
        return ""
    if _COMPUTE_INTENT_RE.search(asked):
        return ("this asks for a COMPUTED result (a number, extremum, count, "
                "list, or plot), not a definition -- do NOT use explain. Fill "
                "SHAPE A (fetch + map + reduce) for a quantity, or SHAPE B "
                "(search) for the largest/smallest h11 satisfying a condition")
    if _CONCRETE_POLY_RE.search(asked):
        return ("this names a SPECIFIC polytope (by Hodge number / id / 'the "
                "first polytope') and asks for one of its properties -- that "
                "is a computation, NOT a definition. Do NOT use explain; fill "
                "SHAPE A: fetch that polytope and compute the asked quantity.")
    return ""


def _plot_issue(spec, direct):
    """'' or a recompile instruction: the request asks for a figure but the
    spec has no plot entries (observed: chat follow-ups kept compiling
    plot=null because the y column lived in a previous turn)."""
    if spec.get("search"):
        return ""        # search mode has no plot stage
    if _PLOT_WORDS_RE.search(direct) and not spec.get("plot"):
        return ("the request asks for a plot but the spec has plot=null -- "
                "add a plot entry; its x/y/color may be THIS turn's map "
                "columns or any stored column shown in the context")
    return ""


def _range_issue(spec, direct):
    """'' or a recompile instruction: the request sweeps h11 but the spec
    fetches only one value. (A use_stored spec inherits whatever spread the
    stored ids have, so it is exempt.)"""
    if spec.get("search"):
        return ""        # a search sweeps h11 by construction
    f = spec.get("fetch") or {}
    if f.get("use_stored"):
        return ""
    if f.get("_census"):
        return ""        # the census spans every h11 by construction
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
    if spec.get("search"):
        exprs += " " + spec["search"].get("condition", "")
        exprs += " " + " ".join((spec["search"].get("map") or {}).values())
    toks = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", exprs))
    by_term = expected_by_term(direct)
    issues = []
    for term, markers in by_term.items():
        if toks & markers:
            continue
        foreign = toks & (ALL_MARKERS - markers)
        # only the foreign markers that do NOT belong to ANOTHER matched term
        # signal a real miscomputation. When every foreign marker present is a
        # marker of some OTHER term the request also names, the spec is just
        # computing a different matched quantity -- either a genuine alternative
        # or a phantom co-match (observed [1]: 'distinct nonzero triple
        # intersection numbers' spuriously co-matches 'distinct calabi-yaus';
        # the expression uses intersection_numbers, which IS the right term's
        # marker). Blocking those sent a correct spec to the walk. The original
        # case still fires: 'lattice points' compiled to cy_volume, where
        # cy_volume is not a marker of any matched term.
        other = set().union(*(m for t, m in by_term.items() if t != term)) \
            if len(by_term) > 1 else set()
        foreign -= other
        if foreign:
            from cytools_agent.tools.glossary import cy_glossary
            recipe = cy_glossary(term).get("recipe", "")
            issues.append(
                f"the request's {term!r} is computed via "
                f"{'/'.join(sorted(markers))}, but no expression uses it "
                f"(found {'/'.join(sorted(foreign))} instead). Use this "
                f"recipe verbatim: {recipe}")
    return "; ".join(issues)


def compile_pipeline(pm, direct, cheatsheet, context="", raw=""):
    """One schema-constrained compile call -> the spec dict (unvalidated).
    `context` carries prior-turn information (previous questions/answers and
    the stored id lists) so follow-ups can resolve references. `raw` is the
    original question: glossary recipes are matched over it too, since translate
    may reword a canonical term out of the glossary vocabulary ("interior to
    facets" -> "inside the facets") and the recipe hint would then be missed."""
    gloss = glossary_context(f"{direct} {raw}".strip()) or ""
    user = ((f"{context}\n\n" if context else "")
            + f"REQUEST:\n{direct}\n\n{cheatsheet}"
            + (f"\n\n{gloss}" if gloss else ""))
    return pm._json(_COMPILE_INSTRUCTIONS, user, think=pm.plan_think,
                    label="PM.compile", schema=PIPELINE_FORMAT)


def _run_search(spec, evidence):
    """Execute a search-mode spec; compose the answer WITH its epistemics."""
    from cytools_agent.tools import search_polytopes
    from cytools_agent.tools import ledger
    s = spec["search"]
    kw = {k: s[k] for k in ("objective", "h11_max", "h11_min")
          if s.get(k) is not None}
    if s.get("map"):
        kw["helpers"] = s["map"]
    r = search_polytopes(s["condition"], **kw)
    search_row = ledger.last_id()
    _obs(evidence, "pipeline search",
         f"search_polytopes({s['condition']!r}, helpers={s.get('map')!r}, "
         f"objective={s.get('objective')!r})", r)
    if not r["found"]:
        checked = ", ".join(str(h) for h in r["coverage"])
        return (f"No qualifying polytope found (levels probed: {checked}; "
                f"{r['queries_used']} database queries). {r['note']}")
    cov = r["coverage"]
    exhausted = [h for h, c in cov.items() if c["exhaustive"]
                 and c["hits"] == 0]
    prefix_empty = [h for h, c in cov.items() if not c["exhaustive"]
                    and c["hits"] == 0]
    return (f"Confirmed: h11 = {r['best_h11']} admits a qualifying polytope "
            f"-- witness {r['witness']} ({r['n_hits_at_best']} found at that "
            f"level) [ledger row {search_row}]. Levels with no qualifying polytope found: "
            f"exhaustively checked {exhausted or 'none'}; prefix-checked "
            f"only (absence NOT proven) {prefix_empty or 'none'}. "
            f"{r['queries_used']} database queries used. "
            f"So {r['best_h11']} is a confirmed lower bound for the "
            f"objective; levels above it were checked only partially unless "
            f"listed as exhaustive.")


class InvariantViolation(Exception):
    """Computed data failed a machine-checked identity. not a compile
    problem: falling back to the free-form walk would compute on the same
    bad data, so the caller must surface this honestly instead."""


def _audit_sample(ids, evidence, k=3):
    """Post-execution audit: run the polytope invariant suite on a small
    sample of the ids just used. Catches wrong-formula computation, corrupted
    geometry, and cytools regressions (content-ids cover relabeling)."""
    import random
    from cytools_agent.tools.invariants import run_polytope_invariants
    sample = random.sample(list(ids), k=min(k, len(ids)))
    bad = {}
    for ks in sample:
        res = run_polytope_invariants(polytope.get_polytope(ks))
        viols = {n: v for n, v in res.items()
                 if v is not True and v != "n/a"}
        if viols:
            bad[str(ks)] = viols
    _obs(evidence, "pipeline invariant audit",
         f"run_polytope_invariants over sample {[str(s) for s in sample]}",
         bad or "all invariants hold")
    if bad:
        raise InvariantViolation(
            f"computed data FAILED machine-checked identities: {bad}. "
            f"Refusing to answer from this data.")


def _run_explain(spec, evidence):
    """Answer a conceptual / documentation / capability question from the
    reference database (glossary + real docstrings) -- NOT from model
    knowledge. The composed text is harness-assembled from source-grounded
    lookups, each of which is a ledgered tool call, so the answer is as
    evidence-backed as any computation."""
    from cytools_agent.tools import reference, ledger
    e = spec["explain"]
    if e["kind"] == "capability":
        from cytools_agent.tools import MODEL_TOOLS
        from cytools_agent.tools.glossary import _SECTIONS
        import inspect
        tools = []
        for t in MODEL_TOOLS:
            fn = getattr(t, "__wrapped__", t)
            summary = (inspect.getdoc(fn) or "").strip().split("\n\n")[0]
            summary = " ".join(summary.split())[:140]
            tools.append(f"- {fn.__name__}: {summary}")
        # the topic index, so a capability answer says what it KNOWS, not just
        # which functions exist (observed: a bare tool list under-answered
        # "how high can I compute NTFEs" / "what can you do")
        topics = "\n".join(f"- {title}: {', '.join(terms)}"
                           for title, _b, terms in _SECTIONS)
        _obs(evidence, "pipeline explain (capability)",
             "MODEL_TOOLS + reference table of contents",
             f"{len(tools)} tools, {len(_SECTIONS)} topic sections")
        return ("This harness computes Calabi-Yau / toric-geometry quantities "
                "over the Kreuzer-Skarke database with CYTools. Available "
                "tools:\n" + "\n".join(tools)
                + "\n\nTopics it has source-derived recipes for (ask for any "
                "by name, or call reference(<topic>) to drill in):\n" + topics)
    # concept / how-to: resolve each query against the reference database
    blocks, any_hit = [], False
    for q in e.get("queries", [])[:5]:
        r = reference(q)
        row = ledger.last_id()
        _obs(evidence, f"pipeline explain lookup {q!r}",
             f"reference({q!r})", r)
        lines = []
        for g in r.get("glossary", []):
            lines.append(f"  - {g['term']}: {g['definition']} "
                         f"Recipe: {g['recipe']}")
            any_hit = True
        for a in r.get("api", [])[:3]:
            doc = (" -- " + a["doc"]) if a.get("doc") else ""
            lines.append(f"  - {a['name']}{a['signature']}{doc}")
            any_hit = True
        if lines:
            blocks.append(f"{q} [ref, ledger row {row}]:\n" + "\n".join(lines))
        else:
            blocks.append(f"{q}: no reference entry found.")
    if not any_hit:
        # nothing resolved -- the walk (which can read docstrings live) may do
        # better than an empty reference answer
        raise RuntimeError("explain resolved no glossary/API entries")
    return ("From the CYTools reference (glossary definitions + recipes and "
            "real docstrings; not model knowledge):\n\n" + "\n\n".join(blocks))


def _run_census(spec, evidence):
    """Execute a whole-database (fetch.h11=null) spec against the local KS
    census: per-h11 polytope counts, no fetching, no database queries. The
    iteration domain is h11 values, not polytopes -- argmax/argmin return
    the h11 value at the extreme."""
    import builtins
    from cytools_agent.tools import code as _code
    from cytools_agent.tools import ks_stats, ledger
    stats = ks_stats()
    census_row = ledger.last_id()
    by = stats["count_by_h11"]
    h11_vals = sorted(by)
    cols = {"h11": h11_vals, "count": [by[h] for h in h11_vals]}
    _obs(evidence, "pipeline census",
         "ks_stats()   # whole-database per-h11 counts (local census)",
         {"total": stats["total"], "h11_min": stats["h11_min"],
          "h11_max": stats["h11_max"]})
    scope_base = {b: getattr(builtins, b) for b in dir(builtins)}
    for name, expr in (spec.get("map") or {}).items():
        if name in cols:                # identity refills of h11/count
            continue
        c = compile(expr, f"<census:{name}>", "eval")
        cols[name] = [eval(c, dict(scope_base, h11=hv, count=cv))
                      for hv, cv in zip(cols["h11"], cols["count"])]
    _code._NS.update(cols)              # so make_plot can read them by name

    parts = [f"Across the WHOLE KS database: {stats['total']} polytopes, "
             f"h11 from {stats['h11_min']} to {stats['h11_max']} "
             f"({len(h11_vals)} distinct h11 values; local census, no "
             f"database queries) [ledger row {census_row}]."]
    for red in spec.get("reduce") or []:
        val = _reduce(red["op"], cols[red["of"]], h11_vals)
        _obs(evidence, f"pipeline reduce {red['name']}",
             f"{red['op']}({red['of']})", val)
        extreme = (" (this is the h11 value at the extreme)"
                   if red["op"] in ("argmax", "argmin") else "")
        parts.append(f"{red['name']} ({red['op']} of {red['of']}): "
                     f"{val}{extreme} [ledger row {census_row}].")
    for p in spec.get("plot") or []:
        note = make_plot(kind=p["kind"], x=p["x"], y=p.get("y"),
                         color=p.get("color"), logx=bool(p.get("logx")),
                         logy=bool(p.get("logy")),
                         xlabel=p["x"], ylabel=p.get("y") or "")
        _obs(evidence, "pipeline plot",
             f"make_plot(kind={p['kind']!r}, x={p['x']!r}, "
             f"y={p.get('y')!r})", note)
        parts.append(note.replace("figure built.", "Figure saved:").strip())
    return " ".join(parts)


def run_pipeline(spec, evidence):
    """Execute a VALIDATED spec; return the composed answer string.
    Raises on any stage failure (caller falls back to the free-form walk)."""
    from cytools_agent.tools import code as _code
    if spec.get("explain"):
        return _run_explain(spec, evidence)
    if spec.get("search"):
        return _run_search(spec, evidence)
    f = spec["fetch"]
    if f.get("_census"):
        return _run_census(spec, evidence)
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
            ids += fetch_polytopes(
                limit=f["limit"], h11=h, h21=f.get("h21"),
                favorable=f.get("favorable"))
        _obs(evidence, "pipeline fetch",
             f"fetch_polytopes(limit={f['limit']}, h11={h11s}, "
             f"h21={f.get('h21')!r}, favorable={f.get('favorable')!r})",
             f"{len(ids)} ids: {list(ids[:5])}...")

    sel = f.get("_select")
    if sel is not None:
        if sel >= len(ids):
            raise RuntimeError(f"requested the polytope at index {sel} but "
                               f"only {len(ids)} were fetched")
        ids = [ids[sel]]
        _obs(evidence, "pipeline select",
             f"ids[{sel}]   # positional reference", ids[0])
        spec["reduce"] = []   # one polytope: its map column is the deliverable

    from cytools_agent.tools import ledger
    r = compute_for_each(ids, spec["map"])
    map_row = ledger.last_id()    # the harness-written backbone row for the
                                  # map call -- cited next to derived numbers
    _obs(evidence, "pipeline map",
         f"compute_for_each(ids, {spec['map']!r})", r)
    # require "most" items to succeed -- but a SINGULAR fetch (limit=1, e.g.
    # 'the first polytope at h11=3') has n_requested==1, and the old floor of 2
    # made that threshold impossible to meet, so every single-polytope spec
    # raised and fell back to the walk. Floor of 2 only applies past 1 item.
    need = 1 if r["n_requested"] <= 1 else max(2, r["n_requested"] // 2)
    if r["n_ok"] < need:
        raise RuntimeError(f"map failed on most items: {r.get('errors')}")
    _audit_sample(_code._NS["ok_ids"], evidence)

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
    if f.get("_limit_note"):
        parts.append(f"Note: {f['_limit_note']}.")
    # a single-polytope fetch has nothing to aggregate over: reduce ops
    # (mean/min/max/sum/count/argmax/...) only mean something across a SET, so
    # any reduce the model emitted here is spurious. argmax/argmin would even
    # return the lone polytope's id instead of the asked value. Drop the reduce
    # and let the map column itself be the deliverable (handled just below).
    if len(ok_ids) == 1:
        spec["reduce"] = []
    from cytools_agent.orchestrator._final import set_final, kind_of
    if not spec["reduce"]:
        # no aggregation asked: the per-item values ARE the deliverable --
        # show them (briefly) instead of only naming the columns
        for name in spec["map"]:
            vals = cols[name]
            shown = vals if len(vals) <= 40 else vals[:40]
            tail = " ..." if len(vals) > 40 else ""
            parts.append(f"{name}: {shown}{tail}")
        # a single map column is the deliverable -> capture it typed: one value
        # is a scalar answer (bool/int/float), many values are a list
        if len(spec["map"]) == 1:
            col = list(cols[spec["map"][0]])
            if len(col) == 1 and kind_of(col[0]):
                set_final(kind_of(col[0]), col[0])
            else:
                set_final("list", col)
    for red in spec["reduce"]:
        op = red["op"]
        val = _reduce(op, cols[red["of"]], ok_ids)
        _obs(evidence, f"pipeline reduce {red['name']}",
             f"{op}({red['of']})", val)
        parts.append(f"{red['name']} ({op} of {red['of']}): {val} "
                     f"[ledger row {map_row}].")
        # capture the reduce result as the typed committed answer (last wins)
        _k = kind_of(val)
        if _k:
            set_final(_k, list(val) if _k == "list" else val)

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


def _apply_singular_limit(spec, asked, log):
    """A singular reference ('the first polytope') with no stated count: the
    limit defaulted to 10 and the answer would be a list -- fetch exactly one.
    Applied to BOTH the first spec and any runtime-recompiled spec (the latter
    was previously missed, leaving singular questions on limit=10)."""
    fc = spec.get("fetch") or {}
    if (str(fc.get("_limit_note", "")).startswith("no count")
            and _SINGULAR_POLY_RE.search(asked)
            and not _PLURAL_COUNT_RE.search(asked)):
        fc["limit"] = 1
        fc.pop("_limit_note", None)
        log("[pipeline]", "singular reference -> limit=1")


_EXPLICIT_COUNT_RE = re.compile(r"\bfirst\s+(\d+)\b|\b(\d+)\s+polytopes\b",
                                re.I)


def _apply_explicit_limit(spec, asked, log):
    """An explicit cardinal count ('first 5', '20 polytopes'): compile
    sometimes drops it, so the limit defaults to 10 and the answer counts the
    wrong number. The asked count is authoritative -- set it (a positional ref
    like 'the 5th' is _apply_select_index's job and is skipped here)."""
    fc = spec.get("fetch") or {}
    if (spec.get("search") or spec.get("explain") or fc.get("_census")
            or fc.get("use_stored") or fc.get("_select") is not None):
        return
    m = _EXPLICIT_COUNT_RE.search(asked)
    if not m:
        return
    n = int(m.group(1) or m.group(2))
    h11 = fc.get("h11")
    n_h11 = len(h11) if isinstance(h11, list) else 1
    n = min(max(n, 1), max(1, 2000 // n_h11))   # respect the politeness cap
    if fc.get("limit") == n:
        return
    fc["limit"] = n
    fc.pop("_limit_note", None)
    log("[pipeline]", f"explicit count -> limit={n}")


def _apply_select_index(spec, asked, log):
    """Positional reference ('the second polytope', 'index 15'): record the
    0-based index on the fetch and ensure enough polytopes are fetched to reach
    it. Only for a normal single-h11 fetch -- an index across several h11 values
    is ambiguous, so it is left alone. run_pipeline slices to that one polytope
    before the map runs (so the quantity is computed for it alone, not the
    whole batch) and drops the reduce."""
    fc = spec.get("fetch") or {}
    if (spec.get("search") or spec.get("explain")
            or fc.get("_census") or fc.get("use_stored")):
        return
    h11 = fc.get("h11")
    if isinstance(h11, list) and len(h11) != 1:
        return
    k = _parse_select_index(asked)
    if k is None:
        return
    fc["_select"] = k
    if (fc.get("limit") or 0) < k + 1:
        fc["limit"] = k + 1
    log("[pipeline]", f"positional reference -> select index {k}")


def _spec_issue(spec, asked, direct):
    """All post-schema validation for a compiled spec, in ONE place so the
    main compile loop and the runtime-recompile path cannot drift (they did:
    the recompile path omitted the explain guard, letting an explain spec slip
    through after a compute spec hit a runtime error). Returns a reason or ''.
    `asked` is the raw question (the guards key on words translate may drop)."""
    why = _valid(spec)
    if why:
        return why
    if spec.get("explain"):
        # the only guard that applies to explain: did it dodge real work?
        return _explain_issue(spec, asked)
    # the range / plot / quantity guards are about COMPUTE specs. ALL key on the
    # raw `asked`, not the paraphrased `direct`: translate rewords a glossary
    # term out of its canonical vocabulary ("interior to facets" -> "inside the
    # facets"), so matching the recipe on `direct` misses it and the lint passes
    # a wrong quantity (observed id 14: points-interior-to-facets -> n_points).
    why = _range_issue(spec, asked) or _plot_issue(spec, asked)
    if why:
        return why
    why = _spec_quantity_issues(spec, asked)
    return ("quantity lint: " + why) if why else ""


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
                                    context=context, raw=raw)
        except Exception as e:
            emit("pipeline", fits=False, reason=f"compile error: {e}")
            return None
        why = _spec_issue(spec, asked, direct)
        if not why:
            break
        feedback = ("\n\n(Your previous pipeline had this problem -- fix "
                    f"exactly it: {why})")
    if why:
        log("[pipeline: falling back]", why)
        emit("pipeline", fits=False, reason=why, spec=spec)
        return None
    _apply_singular_limit(spec, asked, log)
    _apply_explicit_limit(spec, asked, log)
    _apply_select_index(spec, asked, log)
    emit("pipeline", fits=True, spec=spec)
    log("[pipeline spec]", str(spec))
    try:
        return run_pipeline(spec, evidence)
    except InvariantViolation as e:
        # bad DATA, not a bad spec: the walk would recompute on the same
        # data, so answer honestly instead of falling back
        log("[pipeline: invariant violation -- refusing]", str(e))
        emit("pipeline", fits=True, invariant_violation=str(e)[:300])
        return (f"Cannot answer: {e} This indicates corrupted cached data "
                f"or a computation bug -- the result would not be "
                f"trustworthy. (Machine-checked; no model judgment "
                f"involved.)")
    except Exception as e:
        # a spec that VALIDATED but raised at runtime (e.g. a hallucinated
        # field in a search condition) gets ONE recompile with the error --
        # same fix-exactly-this pattern as validation failures
        log("[pipeline: runtime error, recompiling once]",
            f"{type(e).__name__}: {e}")
        emit("pipeline", fits=False, reason=f"runtime: {e}", retrying=True)
        try:
            spec2 = compile_pipeline(
                pm, direct + ("\n\n(Your previous attempt failed when RUN: "
                              f"{e}. Fix exactly that -- use the glossary "
                              "recipes verbatim.)"),
                cheatsheet, context=context, raw=raw)
            if not _spec_issue(spec2, asked, direct):
                _apply_singular_limit(spec2, asked, log)   # same as first spec
                _apply_explicit_limit(spec2, asked, log)
                _apply_select_index(spec2, asked, log)
                emit("pipeline", fits=True, spec=spec2, retry=True)
                return run_pipeline(spec2, evidence)
        except Exception as e2:
            emit("pipeline", fits=False, reason=f"retry runtime: {e2}")
        return None

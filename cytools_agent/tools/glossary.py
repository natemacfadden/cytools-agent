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
# Description:  Glossary that maps CYTools / toric-geometry jargon to a plain
#               definition AND the exact recipe to compute it with these tools.
#               Lets the model translate a specialized term in a question into
#               the right operation instead of guessing. Recipes assume `ks_ind` is
#               a fetched polytope id and, where a CY is needed,
#               `h = get_heights(ks_ind)["heights"][0]`.
# -----------------------------------------------------------------------------

# external imports
import re

# term -> (definition, recipe, [synonyms]). The canonical term and every
# synonym are matched case/punctuation-insensitively and by substring/token, so
# "maximum 2-face genus" finds "2-face genus" and "chi" finds the Euler char.
_GLOSSARY = {
    "2-face genus": (
        "Genus of a 2-face = number of interior lattice points of the DUAL "
        "1-face (edge). One value per 2-face; questions often want the max or "
        "the sum (total genus).",
        "get_polytope_info(ks_ind)['genera_2face']   # sorted desc; take max/sum",
        ["genus", "2-face genera", "face genus", "genus of a 2-face",
         "genera of the 2-faces", "total genus", "sum of 2-face genera"]),
    "favorable": (
        "Favorable (lattice N or M): all Kahler (1,1)-forms of the CY "
        "hypersurface descend from the ambient toric variety. The standard CY "
        "build needs N-favorability.",
        "get_polytope_info(ks_ind)['favorable_N']   # or 'favorable_M'",
        ["favorability", "n-favorable", "m-favorable", "favorable polytope"]),
    "ntfe": (
        "Not-2-face-equivalent triangulations: FRSTs modulo equal restriction "
        "to 2-faces -- inequivalent triangulations giving possibly-distinct "
        "CYs (distinctness is not proven).",
        "get_heights(ks_ind)                  # kind='NTFE' default; "
        "count = result['shape'][0]",
        ["inequivalent triangulations", "not 2-face equivalent",
         "ntfe triangulations", "ntfe_frsts"]),
    "frst": (
        "Fine Regular Star Triangulation. Many more than NTFE.",
        "get_heights(ks_ind, kind='FRST')     # count = result['shape'][0]",
        ["fine regular star triangulation", "fine regular star"]),
    "fine": (
        "A triangulation is fine if it uses ALL points of the point "
        "configuration (none left out). For a 4D reflexive polytope the "
        "configuration is the lattice points NOT interior to a facet; in "
        "general it is all lattice points.",
        "get_triangulation_info(ks_ind, h)['is_fine']",
        ["fine triangulation"]),
    "regular": (
        "A triangulation is regular if it is induced by a height vector -- "
        "equivalently, its secondary cone is solid (full-dimensional).",
        "get_triangulation_info(ks_ind, h)['is_regular']",
        ["regular triangulation"]),
    "star": (
        "A triangulation is star if the origin is a vertex of every simplex "
        "(required for the toric / CY construction). Any fine, regular "
        "triangulation of a 4D reflexive polytope can be made star just by "
        "lowering the height of the origin.",
        "get_triangulation_info(ks_ind, h)['is_star']",
        ["star triangulation"]),
    "secondary cone": (
        "The cone of height vectors that induce a given triangulation -- the "
        "'cone of strictly convex piecewise-linear functions'; the "
        "triangulation is regular iff this cone is solid. The toric Kahler "
        "cone is this cone with its lineality space projected out (the chamber "
        "complex of the secondary fan).",
        "get_polytope(ks_ind).triangulate(heights=h, make_star=True)"
        ".secondary_cone()",
        ["secondary fan"]),
    "mori cone": (
        "Cone of effective curves of the CY; its generating rays (in basis). "
        "cone='toric' gives the toric Mori cone of this triangulation; "
        "cone='Kcup' gives Mcap (see 'mori cone cap').",
        "get_cy_cones(ks_ind, h, cone='toric')['mori_rays']",
        ["mori cone rays", "effective curve cone", "cone of effective curves",
         "mori generators", "toric mori cone"]),
    "kahler cone": (
        "Dual of the Mori cone; the Mori cone's rays ARE its bounding "
        "hyperplane normals (facet normals), so #hyperplanes = #those "
        "vectors. Equivalently, the toric Kahler cone is the secondary cone "
        "with its lineality space projected out (the chamber complex of the "
        "secondary fan).",
        "len(get_cy_cones(ks_ind, h, cone='toric')['kahler_cone_hyperplanes'])",
        ["kaehler cone", "kahler cone hyperplanes", "facet normals",
         "hyperplanes bounding the kahler cone", "kahler cone facet normals"]),
    "toric curve volume": (
        "Volume of an effective (Mori-cone) curve at a Kahler point t: "
        "(Mori ray) . t. get_cy_info returns the list curve_volumes; reduce it "
        "with min()/max() for the smallest/largest.",
        "info = get_cy_info(ks_ind, h, t='tip', cone='toric'); "
        "info['curve_volumes']   # min(...)/max(...) for smallest/largest",
        ["curve volume", "curve volumes", "minimum curve volume",
         "min curve volume", "toric-curve volume"]),
    "divisor volume": (
        "Volume of a basis divisor at a Kahler point t.",
        "get_cy_info(ks_ind, h, t='tip')['divisor_volumes']",
        ["divisor volumes"]),
    "cy volume": (
        "Total volume of the Calabi-Yau at a Kahler point t.",
        "get_cy_info(ks_ind, h, t='tip')['cy_volume']",
        ["calabi-yau volume", "total cy volume", "volume of the calabi-yau"]),
    "triple intersection numbers": (
        "The intersection-ring numbers kappa_ijk of the CY, in a divisor "
        "basis.",
        "get_cy_info(ks_ind, h)['intersection_numbers']  # nonzero, [i,j,k,value]",
        ["intersection numbers", "triple intersections", "intersection ring",
         "kappa"]),
    "second chern class": (
        "Integrals of the CY's second Chern class c2 over each basis divisor "
        "(a vector, one entry per basis divisor).",
        "get_cy_info(ks_ind, h)['second_chern_class']",
        ["c2", "chern class", "second chern"]),
    "2-face lattice points": (
        "The number of lattice points ON a 2-face (boundary + interior of "
        "that face), one count per 2-face. NOT the 2-face genus "
        "(genera_2face counts interior points of the DUAL 1-face).",
        "[len(f.points()) for f in get_polytope(ks_ind).faces(2)]   "
        "# one count per 2-face; reduce with max()/min()/all(...)",
        ["lattice points of a 2-face", "points of a 2-face", "2-face points",
         "points on each 2-face", "lattice points on the 2-faces",
         "2-face point count"]),
    "lattice points": (
        "The lattice points of the polytope; n_points is how many (the count "
        "includes the origin). Vertices are a subset (n_vertices).",
        "get_polytope_info(ks_ind)['n_points']   # count; points: "
        "get_polytope(ks_ind).points()",
        ["lattice point count", "number of lattice points", "points of the "
         "polytope", "lattice point"]),
    "hodge numbers": (
        "h^1,1 (number of Kahler moduli) and h^2,1 (number of "
        "complex-structure moduli) of the CY threefold.",
        "get_polytope_info(ks_ind)['h11'], get_polytope_info(ks_ind)['h21']",
        # NB: no bare 'h11'/'h21' synonyms -- they appear as the spec 'h11=X'
        # in almost every question and would false-trigger the scanner.
        ["hodge number", "hodge numbers", "hpq"]),
    "euler characteristic": (
        "Euler characteristic of the CY threefold, 2*(h11 - h21).",
        "get_cy_info(ks_ind, h)['euler_characteristic']",
        ["chi", "euler char", "euler number"]),
    "prime toric divisors": (
        "Prime toric divisors of the CY = the boundary lattice points not "
        "interior to facets (every lattice point except the origin and those "
        "interior to facets). The count is n_prime_toric_divisors.",
        "get_cy_info(ks_ind, h)['n_prime_toric_divisors']   # == "
        "len(get_polytope(ks_ind).boundary_points_not_interior_to_facets())",
        ["toric divisors", "number of prime toric divisors"]),
    "glsm charge matrix": (
        "GLSM charge (weight) matrix of the polytope.",
        "get_polytope(ks_ind).glsm_charge_matrix()",
        ["glsm matrix", "charge matrix", "weight matrix", "glsm"]),
    "facet": (
        "A facet is a codimension-1 face; for a 4d polytope, the 3-faces.",
        "get_polytope_info(ks_ind)['facedim_to_nfaces'][3]   # number of facets",
        ["facets", "3-face", "3-faces", "codimension-1 face"]),
    "automorphisms": (
        "The SL+/-(d,Z) lattice automorphisms (4x4 matrices) that fix the "
        "polytope; the group's order is automorphism_order.",
        "get_polytope(ks_ind).automorphisms()   # matrices; "
        "count = get_polytope_info(ks_ind)['automorphism_order']",
        ["automorphism", "automorphism group", "automorphism group order",
         "automorphism order", "order of the automorphism group",
         "symmetry group order"]),
    "stretched cone tip": (
        "The smallest-norm point inside the cone that is at least distance c "
        "from every defining hyperplane (wall) -- a canonical point well "
        "inside the cone. With t='tip' here, c=1.",
        "get_cy_info(ks_ind, h, t='tip')",
        ["tip of the stretched cone", "stretched kahler cone tip",
         "cone tip"]),
    "distinct calabi-yaus": (
        "Number of possibly-distinct CYs from a polytope = its NTFE count "
        "(an upper bound; true distinctness is not proven). For provably "
        "distinct CYs, dedupe by CY equality instead.",
        "get_heights(ks_ind)['shape'][0]   # possibly-distinct (NTFE) count; "
        "provably-distinct: len({t.get_cy() for t in "
        "get_polytope(ks_ind).all_triangulations()})",
        ["distinct cys", "number of distinct calabi-yaus",
         "inequivalent calabi-yaus"]),
    "mori cone cap": (
        "Mcap (cy.mori_cone_cap): a Mori cone, the OUTER approximation to the "
        "true Mori cone. Its DUAL (rays <-> hyperplanes) is Kcup, a Kahler "
        "cone: the INNER approximation to the true Kahler cone, the union of "
        "the toric Kahler cones of all 2-face-equivalent triangulations. Kcup "
        "!= Mcap; they are dual cones. cone='Kcup' selects this pair (most "
        "accurate, costly at large h11); cone='toric' is the cheaper toric "
        "cone of one triangulation.",
        "get_cy_cones(ks_ind, h, cone='Kcup')['mori_rays']   # Mcap rays; "
        "['kahler_cone_hyperplanes'] gives them as Kcup's hyperplane normals",
        ["kcup", "mcap", "capped mori cone"]),
    "stanley-reisner ideal": (
        "Generators of the Stanley-Reisner ideal (the minimal non-faces) of "
        "the star triangulation / toric variety.",
        "get_polytope(ks_ind).triangulate(heights=h, make_star=True).sr_ideal()",
        ["sr ideal", "stanley reisner", "minimal non-faces"]),
    "2d reflexive subpolytopes": (
        "The 2-dimensional reflexive sub-polytopes contained in the polytope.",
        "get_polytope(ks_ind).find_2d_reflexive_subpolytopes()",
        ["2d reflexive subpolys", "2-dimensional reflexive subpolytopes"]),
    "d3 tadpole charge": (
        "D3 tadpole charge of the CY: Q0 = (2 + h11 + h21) / 2.",
        "info = get_polytope_info(ks_ind); (2 + info['h11'] + info['h21']) / 2",
        ["tadpole", "tadpole charge", "q0"]),
    "dual polytope": (
        "The dual (polar) polytope, get_polytope(ks_ind).dual(). Mirror symmetry "
        "relates a CY to the CY of its dual polytope and swaps h11 <-> h21 -- "
        "it is a relation between the two polytopes, not a property of one. So "
        "the dual polytope's h11 equals this CY's h21.",
        "get_polytope(ks_ind).dual()   # dual/polar polytope; its "
        ".h11(lattice='N') is the mirror h11 (= this CY's h21)",
        ["polar polytope", "polar dual", "mirror", "mirror symmetry",
         "mirror h11"]),
    "rigid divisors": (
        "Rigid prime toric divisors: a prime toric divisor (a lattice point "
        "interior to a face of dim 0-2) is rigid iff its dual face has no "
        "interior points. NOT the same as the prime-divisor count.",
        "get_polytope_info(ks_ind)['n_rigid_divisors']",
        ["rigid prime toric divisors", "rigid divisor", "rigid toric divisor"]),
    "induced 2-face triangulation": (
        "The triangulation a star triangulation induces on each 2-face -- "
        "i.e. its restriction to the 2-faces, one per 2-face.",
        "t = get_polytope(ks_ind).triangulate(heights=h, make_star=True); "
        "faces = t.restrict(restrict_dim=2)   # induced triangulation per "
        "2-face. Per-face counts: [len(f) for f in faces]; total: "
        "sum(len(f) for f in faces)",
        ["induced 2-face triangulations", "2-face triangulation",
         "induced triangulations", "induced triangulations of 2-faces",
         "simplices induced on each 2-face", "induced on each 2-face",
         "simplices per 2-face", "triangulation of each 2-face"]),
}


def _norm(s):
    """Lowercase, strip punctuation, and DEPLURALIZE tokens (faces -> face)
    so 'all of p's 2-faces have ... lattice points' matches the singular
    phrases entries are written in. Trailing-s only; 'ss' kept (class)."""
    s = re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    return " ".join(t[:-1] if len(t) > 3 and t.endswith("s")
                    and not t.endswith("ss") else t
                    for t in s.split())


# every (normalized phrase -> canonical key), over canonical terms + synonyms
_PHRASES = [(_norm(p), k)
            for k, (_d, _r, syns) in _GLOSSARY.items()
            for p in [k, *syns]]


# model-read
def cy_glossary(term: str = "") -> dict:
    """
    Translate a CYTools / toric-geometry term into its definition and the exact
    recipe to compute it with these tools. Call this whenever a question names a
    quantity (genus, favorable, Mori/Kahler cone, curve volume, ...) BEFORE
    computing -- the named quantity rarely means the obvious thing.

    Parameters
    ----------
    term : str, optional
        The term to look up. Case/punctuation-insensitive, matches substrings
        and many synonyms (so "maximum 2-face genus" finds "2-face genus" and
        "chi" finds the Euler characteristic). Omit to list known terms.

    Returns
    -------
    dict
        {"term", "definition", "recipe"} for a match; {"terms": [...]} when no
        term is given; {"error", "known_terms"} when nothing matches. Recipes
        assume `ks_ind` is a fetched id and `h = get_heights(ks_ind)["heights"][0]`.
    """
    if not term:
        return {"terms": sorted(_GLOSSARY)}
    t = _norm(term)
    cands = [(p, k) for p, k in _PHRASES if p in t]       # query names the term
    if not cands:
        cands = [(p, k) for p, k in _PHRASES if t in p]   # query is a fragment
    if not cands:
        tw = set(t.split())
        cands = [(p, k) for p, k in _PHRASES if set(p.split()) <= tw]
    if not cands:
        return {"error": f"no glossary entry for {term!r}",
                "known_terms": sorted(_GLOSSARY)}
    _phrase, key = max(cands, key=lambda pk: len(pk[0]))
    definition, recipe, _syns = _GLOSSARY[key]
    return {"term": key, "definition": definition, "recipe": recipe}


# append the full vocabulary to the (model-read) docstring so the model can see
# which terms are covered without a discovery call -- auto-synced to _GLOSSARY,
# so it never drifts. (function_to_schema / FastMCP send this to the model.)
cy_glossary.__doc__ += (
    "\n\n    Known terms (synonyms also match): "
    + ", ".join(sorted(_GLOSSARY)) + ".")


# Terms whose words show up as selectors/specs, not as the asked-for quantity
# ("first FAVORABLE polytope", "interior to FACETS"), so auto-scanning them is
# mostly noise. They stay available via the cy_glossary tool + vocabulary list.
_SCAN_SKIP = {"favorable", "facet", "fine", "regular", "star"}


# Markers that appear in MANY recipes / questions and so discriminate nothing
# (a lint keyed on them would fire constantly).
_MARKER_STOP = {"h11", "h21", "shape", "heights", "count", "tolist", "items",
                "get", "len", "ks_ind", "tip"}


# human-read
def _recipe_markers(recipe: str) -> set:
    """The discriminative identifiers a recipe computes through: dict fields
    (['curve_volumes']) and method names (.automorphisms(). Generic tokens
    are dropped, so the survivors mark THIS quantity specifically."""
    fields = set(re.findall(r"\['([A-Za-z0-9_]+)'\]", recipe))
    methods = set(re.findall(r"\.([A-Za-z0-9_]+)\(", recipe))
    return {m for m in fields | methods if m not in _MARKER_STOP}


# all known quantity markers across the glossary -- the universe the lint
# checks "computed a DIFFERENT quantity" against
ALL_MARKERS = set().union(
    *(_recipe_markers(r) for _d, r, _s in _GLOSSARY.values()))


# human-read
def _matched_keys(message: str) -> set:
    """Glossary keys the message names: contiguous phrase matches, plus a
    CO-OCCURRENCE fallback for phrases of >=3 distinctive words whose words
    all appear somewhere in the message -- natural phrasing splits a
    quantity across the sentence ('2-faces have <=20 lattice points' never
    contains '2-face lattice points' contiguously)."""
    mtoks = _norm(message).split()
    mset = set(mtoks)

    def _has(seq):
        n = len(seq)
        return n > 0 and any(mtoks[i:i + n] == seq
                             for i in range(len(mtoks) - n + 1))

    keys = set()
    for nphrase, key in _PHRASES:
        if key in _SCAN_SKIP or key in keys:
            continue
        pt = nphrase.split()
        if _has(pt) or (len(pt) >= 3 and set(pt) <= mset):
            keys.add(key)
    return keys


# human-read
def expected_by_term(message: str) -> dict:
    """{glossary term -> its recipe markers} for each term the message names
    (same matching as glossary_context). Per-term so a caller can check that
    EVERY named quantity is computed, not just one of them."""
    out = {}
    for key in _matched_keys(message):
        m = _recipe_markers(_GLOSSARY[key][1])
        if m:
            out[key] = m
    return out


# human-read
def expected_markers(message: str) -> set:
    """Markers for the quantities the message actually names (union over
    expected_by_term). Empty when no term matches -- the lint then has
    nothing to enforce and stays silent."""
    return set().union(*expected_by_term(message).values()) \
        if expected_by_term(message) else set()


# human-read
def glossary_context(message: str, max_terms: int = 4,
                     recipe_only: bool = False) -> str:
    """Scan a message for glossary terms (as whole-token phrases) and return
    their definitions + recipes as a context block, or "" if none. The harness
    appends this to a user message so the model gets the translation without
    having to recognize it should look the term up. Conservative by design:
    when it misses a term, the cy_glossary tool is the backup.

    recipe_only=True returns just the recipes (no paragraph definitions) -- a
    lean view for the ENGINEER, which needs the code pattern, not the prose."""
    keys = sorted(_matched_keys(message),
                  key=lambda k: -len(k))[:max_terms]
    if not keys:
        return ""
    if recipe_only:
        lines = ["(recipes for terms in this step:)"]
        for k in keys:
            _d, recipe, _s = _GLOSSARY[k]
            lines.append(f"- {k}: {recipe}")
    else:
        lines = ["(CYTools glossary -- terms detected in this request, with the "
                 "recipe to use:)"]
        for k in keys:
            definition, recipe, _syns = _GLOSSARY[k]
            lines.append(f"- {k}: {definition} Recipe: {recipe}")
    return "\n".join(lines)

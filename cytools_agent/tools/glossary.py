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
#               the right operation instead of guessing. Recipes assume `ks` is
#               a fetched polytope id and, where a CY is needed,
#               `h = get_heights(ks)["heights"][0]`.
# -----------------------------------------------------------------------------

# external imports
import re

# term -> (definition, recipe, [synonyms]). The canonical term and every
# synonym are matched case/punctuation-insensitively and by substring/token, so
# "maximum 2-face genus" finds "2-face genus" and "chi" finds the Euler char.
_GLOSSARY = {
    "2-face genus": (
        "Genus of a 2-face = number of interior lattice points of the DUAL "
        "1-face (edge). One value per 2-face; questions often want the max.",
        "genera = [len(f.interior_points()) "
        "for f in get_polytope(ks).dual().faces(1)]; max(genera)",
        ["genus", "2-face genera", "face genus", "genus of a 2-face",
         "genera of the 2-faces"]),
    "favorable": (
        "Favorable (lattice N or M): all Kahler (1,1)-forms of the CY "
        "hypersurface descend from the ambient toric variety. The standard CY "
        "build needs N-favorability.",
        "get_polytope_info(ks)['favorable_N']   # or 'favorable_M'",
        ["favorability", "n-favorable", "m-favorable", "favorable polytope"]),
    "ntfe": (
        "Not-2-face-equivalent triangulations: FRSTs modulo equal restriction "
        "to 2-faces -- inequivalent triangulations giving possibly-distinct "
        "CYs (distinctness is not proven).",
        "get_heights(ks)                  # kind='NTFE' default; "
        "count = result['shape'][0]",
        ["inequivalent triangulations", "not 2-face equivalent",
         "ntfe triangulations", "ntfe_frsts"]),
    "frst": (
        "Fine Regular Star Triangulation. Many more than NTFE.",
        "get_heights(ks, kind='FRST')     # count = result['shape'][0]",
        ["fine regular star triangulation", "star triangulation"]),
    "mori cone": (
        "Cone of effective curves of the CY; its generating rays (in basis).",
        "get_cy_cones(ks, h, cone='toric')['mori_rays']",
        ["mori cone rays", "effective curve cone", "cone of effective curves",
         "mori generators", "toric mori cone"]),
    "kahler cone": (
        "Dual of the Mori cone. The Mori cone's rays ARE the bounding "
        "hyperplane normals (facet normals) of the Kahler cone, so the number "
        "of hyperplanes bounding it = the number of those vectors.",
        "len(get_cy_cones(ks, h, cone='toric')['kahler_cone_hyperplanes'])",
        ["kaehler cone", "kahler cone hyperplanes", "facet normals",
         "hyperplanes bounding the kahler cone", "kahler cone facet normals"]),
    "toric curve volume": (
        "Volume of an effective (Mori-cone) curve at a Kahler point t: "
        "(Mori ray) . t. The minimum over rays is the smallest curve volume.",
        "info = get_cy_info(ks, h, t='tip', cone='toric'); "
        "info['curve_volumes']; info['min_curve_volume']",
        ["curve volume", "curve volumes", "minimum curve volume",
         "min curve volume", "toric-curve volume"]),
    "divisor volume": (
        "Volume of a basis divisor at a Kahler point t.",
        "get_cy_info(ks, h, t='tip')['divisor_volumes']",
        ["divisor volumes"]),
    "cy volume": (
        "Total volume of the Calabi-Yau at a Kahler point t.",
        "get_cy_info(ks, h, t='tip')['cy_volume']",
        ["calabi-yau volume", "total cy volume", "volume of the calabi-yau"]),
    "triple intersection numbers": (
        "The intersection-ring numbers kappa_ijk of the CY, in a divisor "
        "basis.",
        "get_cy_info(ks, h)['intersection_numbers']  # nonzero, [i,j,k,value]",
        ["intersection numbers", "triple intersections", "intersection ring",
         "kappa"]),
    "second chern class": (
        "Integrals of the CY's second Chern class c2 over each basis divisor "
        "(a vector, one entry per basis divisor).",
        "get_cy_info(ks, h)['second_chern_class']",
        ["c2", "chern class", "second chern"]),
    "hodge numbers": (
        "h^1,1 (number of Kahler moduli) and h^2,1 (number of "
        "complex-structure moduli) of the CY threefold.",
        "get_polytope_info(ks)['h11'], get_polytope_info(ks)['h21']",
        # NB: no bare 'h11'/'h21' synonyms -- they appear as the spec 'h11=X'
        # in almost every question and would false-trigger the scanner.
        ["hodge number", "hodge numbers", "hpq"]),
    "euler characteristic": (
        "Euler characteristic of the CY threefold, 2*(h11 - h21).",
        "get_cy_info(ks, h)['euler_characteristic']",
        ["chi", "euler char", "euler number"]),
    "prime toric divisors": (
        "Prime toric divisors of the CY = the boundary lattice points not "
        "interior to facets (every lattice point except the origin and those "
        "interior to facets). The count is n_prime_toric_divisors.",
        "get_cy_info(ks, h)['n_prime_toric_divisors']   # == "
        "len(get_polytope(ks).boundary_points_not_interior_to_facets())",
        ["toric divisors", "number of prime toric divisors"]),
    "glsm charge matrix": (
        "GLSM charge (weight) matrix of the polytope.",
        "get_polytope(ks).glsm_charge_matrix()",
        ["glsm matrix", "charge matrix", "weight matrix", "glsm"]),
    "facet": (
        "A facet is a codimension-1 face; for a 4d polytope, the 3-faces.",
        "get_polytope_info(ks)['facedim_to_nfaces'][3]   # number of facets",
        ["facets", "3-face", "3-faces", "codimension-1 face"]),
    "automorphisms": (
        "The SL+/-(d,Z) lattice automorphisms (4x4 matrices) that fix the "
        "polytope; the group's order is automorphism_order.",
        "get_polytope(ks).automorphisms()   # matrices; "
        "count = get_polytope_info(ks)['automorphism_order']",
        ["automorphism", "automorphism group", "automorphism group order",
         "automorphism order", "order of the automorphism group",
         "symmetry group order"]),
    "stretched cone tip": (
        "The smallest-norm point inside the cone that is at least distance c "
        "from every defining hyperplane (wall) -- a canonical point well "
        "inside the cone. With t='tip' here, c=1.",
        "get_cy_info(ks, h, t='tip')",
        ["tip of the stretched cone", "stretched kahler cone tip",
         "cone tip"]),
    "distinct calabi-yaus": (
        "Number of possibly-distinct CYs from a polytope = its NTFE count "
        "(an upper bound; true distinctness is not proven). For provably "
        "distinct CYs, dedupe by CY equality instead.",
        "get_heights(ks)['shape'][0]   # possibly-distinct (NTFE) count; "
        "provably-distinct: len({t.get_cy() for t in "
        "get_polytope(ks).all_triangulations()})",
        ["distinct cys", "number of distinct calabi-yaus",
         "inequivalent calabi-yaus"]),
    "mori cone cap": (
        "Mcap (mori_cone_cap): the capped Mori cone, the conical hull over its "
        "rays. Its DUAL is Kcup, a more accurate Kahler cone than the toric "
        "one, bounded by those same rays as hyperplane normals. Here "
        "cone='Kcup' selects Mcap; cone='toric' is the cheaper toric cone.",
        "get_cy_cones(ks, h, cone='Kcup')['mori_rays']   # Mcap rays; "
        "['kahler_cone_hyperplanes'] gives them as Kcup's hyperplane normals",
        ["kcup", "mcap", "capped mori cone"]),
    "stanley-reisner ideal": (
        "Generators of the Stanley-Reisner ideal (the minimal non-faces) of "
        "the star triangulation / toric variety.",
        "get_polytope(ks).triangulate(heights=h, make_star=True).sr_ideal()",
        ["sr ideal", "stanley reisner", "minimal non-faces"]),
    "2d reflexive subpolytopes": (
        "The 2-dimensional reflexive sub-polytopes contained in the polytope.",
        "get_polytope(ks).find_2d_reflexive_subpolytopes()",
        ["2d reflexive subpolys", "2-dimensional reflexive subpolytopes"]),
}


def _norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


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
        assume `ks` is a fetched id and `h = get_heights(ks)["heights"][0]`.
    """
    if not term:
        return {"terms": sorted(_GLOSSARY)}
    t = _norm(term)
    cands = [(p, k) for p, k in _PHRASES if p in t or t in p]
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
_SCAN_SKIP = {"favorable", "facet"}


# human-read
def glossary_context(message: str, max_terms: int = 4) -> str:
    """Scan a message for glossary terms (as whole-token phrases) and return
    their definitions + recipes as a context block, or "" if none. The harness
    appends this to a user message so the model gets the translation without
    having to recognize it should look the term up. Conservative by design:
    when it misses a term, the cy_glossary tool is the backup."""
    mtoks = _norm(message).split()

    def _has(seq):
        n = len(seq)
        return n > 0 and any(mtoks[i:i + n] == seq
                             for i in range(len(mtoks) - n + 1))

    matched = {}
    for nphrase, key in _PHRASES:
        if key in _SCAN_SKIP:
            continue
        pt = nphrase.split()
        if _has(pt) and len(pt) > matched.get(key, 0):
            matched[key] = len(pt)            # keep the longest phrase per key
    keys = sorted(matched, key=lambda k: -matched[k])[:max_terms]
    if not keys:
        return ""
    lines = ["(CYTools glossary -- terms detected in this request, with the "
             "recipe to use:)"]
    for k in keys:
        definition, recipe, _syns = _GLOSSARY[k]
        lines.append(f"- {k}: {definition} Recipe: {recipe}")
    return "\n".join(lines)

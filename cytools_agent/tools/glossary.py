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
    "trilayer": (
        "Whether the polytope is trilayer (its points lie in three lattice "
        "layers). A boolean per polytope; for 'how many are trilayer' sum the "
        "indicator over the fetched ids.",
        "get_polytope_info(ks_ind)['is_trilayer']",
        ["is trilayer", "trilayer polytope", "is_trilayer", "trilayer count",
         "how many are trilayer", "number of trilayer polytopes"]),
    "ntfe": (
        "Not-2-face-equivalent triangulations: FRSTs modulo equal restriction "
        "to 2-faces -- inequivalent triangulations giving possibly-distinct "
        "CYs (distinctness is not proven). Enumerating ALL of them is feasible "
        "up to ~37 lattice points (roughly h11 <~ 25-30); beyond that, sample "
        "with the GNN sampler instead of an exhaustive count.",
        "get_heights(ks_ind)                  # kind='NTFE' default; "
        "count = result['shape'][0]",
        ["inequivalent triangulations", "not 2-face equivalent",
         "ntfe triangulations", "ntfe_frsts", "how high can i compute ntfes",
         "ntfe feasibility"]),
    "frst": (
        "Fine Regular Star Triangulation. Many more than NTFE.",
        "get_heights(ks_ind, kind='FRST')     # count = result['shape'][0]",
        ["fine regular star triangulation", "fine regular star"]),
    "vex": (
        "A 'vex' is a fan over a lattice VECTOR CONFIGURATION (the Fan class, "
        "cytools.vector_config.fan.Fan, built from "
        "VectorConfiguration.subdivide) -- a fine regular triangulation of a "
        "vector configuration. Vexes SUBSUME FRSTs and are strictly more "
        "general for the CY application, but differ subtly and are LESS WELL "
        "SUPPORTED here currently. Treat vex questions as conceptual: explain "
        "the relationship to FRSTs rather than computing counts.",
        "# experimental -- inspect the Fan class with "
        "cytools_help('cytools.vector_config.fan.Fan'); not a curated tool",
        ["vexes", "vex phase", "vex phases", "polyhedron fan",
         "vector configuration fan", "fan of a vector configuration",
         "fine regular triangulation of a vector configuration"]),
    "gnn sampler": (
        "dualGNN (arXiv:2605.27770): a graph-neural-network sampler of NTFE "
        "triangulations, near-uniform over FRSTs. A SAMPLER, not exhaustive "
        "-- results are a sample, not the census. Works where exhaustive "
        "enumeration is infeasible (tested to h11 ~ 128).",
        "get_heights(ks_ind, n=10, sampler='gnn')   # near-uniform sample; "
        "result['note'] states the provenance",
        ["dualgnn", "gnn", "gnn sampling", "neural network sampler",
         "random triangulations gnn"]),
    "uniform sampling": (
        "Fair (near-uniform) random sampling of triangulations/CYs. Use the "
        "GNN sampler (dualGNN, arXiv:2605.27770); the 'fast' sampler is "
        "biased around the Delaunay heights and is NOT fair.",
        "get_heights(ks_ind, n=10, sampler='gnn')",
        ["fair sampling", "fair sample", "near-uniform sample",
         "unbiased sampling", "uniform sample of triangulations"]),
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
    "kahler moduli space": (
        "The space of Kahler parameters t of the CY. Its (real) dimension "
        "equals h11 -- one Kahler parameter per basis divisor -- so 'how big "
        "is Kahler moduli space' is answered by h11.",
        "get_polytope_info(ks_ind)['h11']   # = dim of Kahler moduli space",
        ["kahler moduli", "dimension of kahler moduli space",
         "size of kahler moduli space", "number of kahler parameters",
         "how big is kahler moduli space", "kahler moduli dimension"]),
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
        "Total volume of the Calabi-Yau at a Kahler point t. It is CUBIC in t "
        "(kappa . t^3 / 6), so rescaling the Kahler moduli t -> s*t scales the "
        "volume by s^3 (and a curve/divisor volume, being linear/quadratic, by "
        "s / s^2).",
        "get_cy_info(ks_ind, h, t='tip')['cy_volume']",
        ["calabi-yau volume", "total cy volume", "volume of the calabi-yau",
         "how volume scales", "volume scaling with kahler moduli"]),
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
    "gopakumar-vafa invariants": (
        "Gopakumar-Vafa (GV) invariants: integer BPS counts of the CY, one per "
        "effective curve class up to a degree cutoff. compute_gvs(max_deg=N) "
        "returns an Invariants object; .gvs is the list of GV integers and "
        ".charges the matching curve classes (use .gws for Gromov-Witten).",
        "get_cy(ks_ind, h).compute_gvs(max_deg=10).gvs   "
        "# list of GV ints; reduce with max()/len(); .charges for curve classes",
        ["gv", "gvs", "gv invariant", "gv invariants", "gopakumar vafa",
         "gopakumar-vafa", "bps invariants", "bps states", "bps counts"]),
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
        "complex-structure moduli) of the CY threefold. For the polytope: "
        "h11 = sum over 2-faces of int(f)*int(dual f) + "
        "|points_not_interior_to_facets| - 5; the sum vanishes exactly when "
        "the polytope is N-favorable.",
        "get_polytope_info(ks_ind)['h11'], get_polytope_info(ks_ind)['h21']",
        # NB: no bare 'h11'/'h21' synonyms -- they appear as the spec 'h11=X'
        # in almost every question and would false-trigger the scanner.
        ["hodge number", "hodge numbers", "hpq"]),
    "lattice volume": (
        "The lattice-normalized volume of the polytope (integer; the "
        "Euclidean volume times dim factorial).",
        "get_polytope(ks_ind).volume()",
        ["normalized volume", "polytope volume", "volume of the polytope"]),
    "dual point count": (
        "Number of lattice points of the dual (polar) polytope.",
        "len(get_polytope(ks_ind).dual().points())",
        ["points of the dual", "dual lattice points",
         "lattice points of the dual polytope"]),
    "points not interior to facets": (
        "Lattice points of the polytope that are not interior to any facet "
        "(includes the origin). This is the point configuration used for "
        "fine triangulations in 4d, and for favorable polytopes "
        "h11 = this count - 5.",
        "len(get_polytope(ks_ind).points_not_interior_to_facets())",
        ["points not interior to a facet", "non-facet-interior points",
         "triangulation point configuration"]),
    "points interior to facets": (
        "Lattice points of the polytope that ARE interior to some facet (the "
        "COMPLEMENT of 'points not interior to facets'); these are dropped "
        "from the fine-triangulation point configuration.",
        "len(get_polytope(ks_ind).points_interior_to_facets())",
        ["points interior to a facet", "facet-interior points",
         "lattice points interior to facets", "points in facet interiors"]),
    "polytope dimension": (
        "The (ambient lattice) dimension of the polytope -- 4 for every 4d "
        "reflexive polytope in this database. NOT h11 (that is the dimension "
        "of Kahler moduli space; see 'kahler moduli space').",
        "get_polytope_info(ks_ind)['dim']",
        ["dimension of the polytope", "dimension of a polytope",
         "ambient dimension", "ambient lattice dimension"]),
    "non-toric divisors": (
        "Divisors of the CY not inherited from the toric variety, counted "
        "by sum over 2-faces of int(f)*int(dual f). Nonzero exactly when "
        "the polytope is NOT N-favorable; this is the correction term in "
        "the h11 formula.",
        "sum(len(f.interior_points()) * len(f.dual().interior_points()) "
        "for f in get_polytope(ks_ind).faces(2))",
        ["non toric divisors", "favorability correction",
         "h11 correction term"]),
    "content id": (
        "Durable content-addressed identity of the polytope: a hash of its "
        "affine normal form, invariant under GL(n,Z) lattice changes and "
        "translations -- the same abstract polytope gets the same id "
        "everywhere. Use it to cite or deduplicate polytopes.",
        "content_id(ks_ind)",
        ["content hash", "polytope hash", "canonical id",
         "normal form hash"]),
    "normal form": (
        "The canonical vertex representative of the polytope's "
        "GL(n,Z)-and-translation equivalence class (PALP affine normal "
        "form). Two polytopes are lattice-equivalent iff their normal "
        "forms are equal.",
        "get_polytope(ks_ind).normal_form(affine_transform=True)",
        ["affine normal form", "palp normal form", "canonical form"]),
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
        "inside the cone. With t='tip' here, c=1. Pass cone='toric' for the "
        "tip of the stretched TORIC Kahler cone (e.g. 'totskc').",
        "get_cy_info(ks_ind, h, t='tip', cone='toric')   # toric cone tip",
        ["tip of the stretched cone", "stretched kahler cone tip",
         "cone tip", "totskc", "tip of the stretched toric kahler cone",
         "stretched toric kahler cone tip", "toric kahler cone tip",
         "tip of the toric kahler cone"]),
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
        {"term", "definition", "recipe"} for a match; the indexed table of
        contents (sections -> terms) when no term is given; {"error",
        "known_terms"} when nothing matches. Recipes assume `ks_ind` is a
        fetched id and `h = get_heights(ks_ind)["heights"][0]`.
    """
    if not term:
        return _table_of_contents()
    t = _norm(term)
    tw = set(t.split())
    # token-aware (NOT raw substring: 'dimension' must not match the 'dimension'
    # inside 'dimensional', which made cy_glossary('dimension') resolve to
    # '2d reflexive subpolytopes'). Match on whole words.
    cands = [(p, k) for p, k in _PHRASES        # the term's words all in query
             if set(p.split()) <= tw]
    if not cands:                               # query's words all in a term
        cands = [(p, k) for p, k in _PHRASES if tw <= set(p.split())]
    if not cands:                               # else best distinctive overlap
        scored = [(len(tw & set(p.split())), p, k) for p, k in _PHRASES]
        best = max(n for n, _p, _k in scored)
        cands = [(p, k) for n, p, k in scored if n == best and n > 0]
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


# ---------------------------------------------------------------------------
# The reference DATABASE: one lookup spanning the source-derived glossary AND
# the live CYTools API (signatures + docstrings). A conceptual / "how do I"
# question is answered from THIS, never from the model's own knowledge -- the
# returned text is glossary definitions and real docstrings, both of which are
# source-grounded and unfakable.
# ---------------------------------------------------------------------------

# built lazily on first reference() call (importing cytools classes at module
# load would tangle the import order); list of (dotted_name, object, is_tool)
_API_INDEX = None

# words that match too many API names to discriminate (every cytools getter is
# a "get_"); a query token of these alone should not pull in the whole API
_API_STOP = {"get", "compute", "is", "to", "the", "a", "an", "of", "for",
             "what", "how", "do", "i", "does", "and", "or", "with", "from",
             "polytope", "cy", "calabiyau", "triangulation"}


def _build_api_index():
    """Index the API surface a researcher would ask about: the curated tools
    (preferred -- these are what the model should call) and the public methods
    of the core CYTools classes (for 'how do I ...' discovery)."""
    from cytools_agent.tools import code as _code
    items = [(name, _code._NS[name], True) for name in _code._TOOL_NAMES]
    try:
        import cytools
        from cytools.triangulation import Triangulation
        from cytools.calabiyau import CalabiYau
        from cytools.toricvariety import ToricVariety
        classes = [cytools.Polytope, cytools.Cone, Triangulation,
                   CalabiYau, ToricVariety]
        for cls in classes:
            for m in dir(cls):
                if m.startswith("_"):
                    continue
                obj = getattr(cls, m, None)
                if callable(obj):
                    items.append((f"{cls.__name__}.{m}", obj, False))
    except Exception:
        pass        # curated tools alone still make a usable index
    return items


def _api_matches(query: str, limit: int = 6) -> list:
    """API entries whose dotted name overlaps the query's distinctive words,
    curated tools ranked first (they are the supported path)."""
    global _API_INDEX
    if _API_INDEX is None:
        _API_INDEX = _build_api_index()
    qtoks = {t for t in _norm(query).split() if t not in _API_STOP
             and len(t) > 1}
    if not qtoks:
        return []
    scored = []
    for name, obj, is_tool in _API_INDEX:
        ntoks = set(re.split(r"[._]", name.lower()))
        overlap = len(qtoks & ntoks)
        # also credit a query word contained in the bare method name
        bare = name.split(".")[-1].lower()
        substr = any(t in bare for t in qtoks if len(t) >= 4)
        if overlap or substr:
            scored.append((overlap + (0.5 if substr else 0)
                           + (0.3 if is_tool else 0), name, obj))
    scored.sort(key=lambda s: -s[0])
    out = []
    for _score, name, obj in scored[:limit]:
        out.append({"name": name, "signature": _signature(name, obj),
                    "doc": _first_doc(obj)})
    return out


def _signature(name, obj):
    import inspect
    try:
        return name.split(".")[-1] + str(inspect.signature(obj))
    except (TypeError, ValueError):
        return name.split(".")[-1] + "(...)"


def _first_doc(obj, cap: int = 400):
    import inspect
    doc = (inspect.getdoc(obj) or "").strip()
    if not doc:
        return ""
    # the first paragraph (description), trimmed -- the full text is one
    # cytools_help call away
    para = doc.split("\n\n")[0].strip()
    return para[:cap] + ("..." if len(para) > cap else "")


def _glossary_matches(query: str, limit: int = 6) -> list:
    """Glossary entries the query names (phrase/co-occurrence), with a
    token-overlap fallback so a loose conceptual phrasing still surfaces the
    closest terms."""
    keys = _matched_keys(query)
    if not keys:
        # token-overlap fallback, but only on DISTINCTIVE words: matching on
        # generic words ("polytope", "cy") pulled in unrelated terms
        qtoks = {t for t in _norm(query).split()
                 if t not in _API_STOP and len(t) > 1}
        scored = sorted(
            ((len(qtoks & set(p.split())), k) for p, k in _PHRASES),
            key=lambda s: -s[0])
        keys = {k for n, k in scored[:limit] if n > 0}
    out = []
    for k in sorted(keys)[:limit]:
        definition, recipe, _syns = _GLOSSARY[k]
        out.append({"term": k, "definition": definition, "recipe": recipe})
    return out


# The table of contents over the glossary: ordered sections, each a
# (title, blurb, [terms]) triple, so the model can BROWSE the knowledge base
# by topic, not only probe it by exact term. Every _GLOSSARY key must appear
# in exactly one section -- verify_glossary enforces this, so the index can
# never silently drift from the content it indexes.
_SECTIONS = [
    ("Polytope & lattice geometry",
     "the reflexive polytope itself: its points, faces, dual, and identity",
     ["lattice points", "2-face lattice points",
      "points not interior to facets", "points interior to facets",
      "dual point count", "lattice volume", "polytope dimension", "facet",
      "dual polytope", "automorphisms", "normal form", "content id"]),
    ("Hodge numbers, topology & physics",
     "topological invariants of the CY and the physics quantities from them",
     ["hodge numbers", "euler characteristic", "2-face genus", "favorable",
      "trilayer", "d3 tadpole charge"]),
    ("Triangulations & sampling",
     "FR(S)Ts, NTFEs, vexes, and how to enumerate or fairly sample them",
     ["fine", "regular", "star", "frst", "ntfe", "vex", "distinct calabi-yaus",
      "gnn sampler", "uniform sampling", "induced 2-face triangulation",
      "secondary cone", "2d reflexive subpolytopes"]),
    ("Toric divisors & GLSM",
     "the divisor structure of the ambient toric variety and the CY",
     ["prime toric divisors", "non-toric divisors", "rigid divisors",
      "glsm charge matrix", "stanley-reisner ideal"]),
    ("Cones & Kahler moduli",
     "Mori/Kahler cones and the volumes evaluated at a Kahler point",
     ["mori cone", "mori cone cap", "kahler cone", "kahler moduli space",
      "stretched cone tip", "toric curve volume", "divisor volume"]),
    ("Calabi-Yau invariants",
     "numbers computed from a chosen triangulation's Calabi-Yau",
     ["cy volume", "triple intersection numbers", "second chern class",
      "gopakumar-vafa invariants"]),
]

_TERM_SECTION = {t: title for title, _b, terms in _SECTIONS for t in terms}

# the index must EXACTLY partition the glossary: no term left unindexed, none
# listed twice or under a stale name. A table of contents that misses a page
# is a bug, so fail loudly at import rather than let the index drift.
_indexed = [t for _ti, _b, terms in _SECTIONS for t in terms]
_idx_missing = set(_GLOSSARY) - set(_indexed)
_idx_unknown = set(_indexed) - set(_GLOSSARY)
_idx_dupes = {t for t in _indexed if _indexed.count(t) > 1}
assert not (_idx_missing or _idx_unknown or _idx_dupes), (
    "reference sections must exactly cover the glossary -- missing: "
    f"{sorted(_idx_missing)}, unknown: {sorted(_idx_unknown)}, "
    f"duplicated: {sorted(_idx_dupes)}")

# queries that mean "show me the index", not "look up this term"
_TOC_QUERIES = {"", "contents", "content", "index", "toc",
                "table of contents", "topics", "topic", "sections",
                "everything", "what is covered", "what do you know"}


def _entry(term: str) -> dict:
    definition, recipe, _syns = _GLOSSARY[term]
    return {"term": term, "definition": definition, "recipe": recipe}


def _table_of_contents() -> dict:
    """The index: every section with its blurb and the terms under it."""
    return {
        "table_of_contents": [
            {"section": title, "about": blurb, "terms": terms}
            for title, blurb, terms in _SECTIONS],
        "how_to_use": (
            "reference(<term>) for a definition + compute recipe; "
            "reference(<section title>) for a whole section; "
            "reference(<question or topic>) to search across the glossary "
            "and the live CYTools API."),
    }


# model-read
def reference(query: str = "") -> dict:
    """
    The CYTools reference -- a searchable, indexed book of knowledge. It spans
    BOTH the glossary (source-derived definitions + the exact recipe to compute
    each quantity) AND the live CYTools API (signatures + docstrings). Answers
    are grounded in those sources, not guessed.

    Three ways to use it:
    - reference() (or "contents"/"index") -> the TABLE OF CONTENTS: every
      topic section and the terms under it. Start here to see what is covered.
    - reference("<section title>") -> every entry in that section.
    - reference("<term, question, or topic>") -> matching glossary entries
      (with see_also siblings) plus relevant CYTools API.

    Parameters
    ----------
    query : str, optional
        Empty/"contents"/"index" for the table of contents; a section title
        for a whole section; otherwise a term/question/topic to look up.
        Case/punctuation-insensitive.

    Returns
    -------
    dict
        For the index: {"table_of_contents": [...], "how_to_use"}. For a
        lookup: {"glossary": [{term, definition, recipe}...], "api": [{name,
        signature, doc}...], "see_also"?}. "note" appears when nothing matched.
    """
    nq = _norm(query)
    if nq in _TOC_QUERIES:
        return _table_of_contents()

    # a whole-section request (the query names a section title)
    for title, blurb, terms in _SECTIONS:
        nt = _norm(title)
        if nq == nt or (len(nq) >= 5 and nq in nt):
            return {"section": title, "about": blurb,
                    "glossary": [_entry(t) for t in terms], "api": []}

    g = _glossary_matches(query)
    a = _api_matches(query)
    out = {"glossary": g, "api": a}
    if g:                       # navigation: siblings of the top hit's section
        sec = _TERM_SECTION.get(g[0]["term"])
        if sec:
            sibs = [t for t in _TERM_SECTION if _TERM_SECTION[t] == sec
                    and t != g[0]["term"]]
            out["see_also"] = {"section": sec, "related_terms": sibs}
    if not g and not a:
        out["note"] = ("no reference entry matched. Call reference() with no "
                       "argument for the table of contents.")
    return out


reference.__doc__ += ("\n\n    Topic sections: "
                      + "; ".join(t for t, _b, _ts in _SECTIONS) + ".")


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
# function words that carry no term signal -- excluded when counting a
# phrase's "distinctive" words for co-occurrence matching
_COOC_STOP = {"the", "of", "a", "an", "is", "are", "was", "what", "how", "do",
              "does", "i", "in", "to", "for", "with", "its", "it", "this",
              "that", "at", "be", "on", "and", "or", "as", "first", "second",
              "many", "number", "value", "has", "have", "had", "each", "all",
              "any", "among", "from", "by", "s", "which", "whose", "there",
              "their", "they"}   # NOTE: "not"/"no" are KEPT distinctive --
                                 # they separate "points NOT interior to
                                 # facets" from "points interior to facets"

# terms whose two content words are commonly split across a sentence and so
# need order-blind 2-word co-occurrence (see _matched_keys). Kept tiny: only
# where the pairing is unambiguous and one word ('polytope') is ubiquitous, so
# a generic 2-word rule would over-fire.
_SHORT_COOC = {"polytope dimension"}


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

    matched = {}        # key -> word-set of its longest matching phrase
    for nphrase, key in _PHRASES:
        if key in _SCAN_SKIP:
            continue
        pt = nphrase.split()
        # default co-occurrence: a >=3-word phrase whose words all appear.
        cooc = len(pt) >= 3 and set(pt) <= mset
        # short co-occurrence: a few terms whose two CONTENT words may be
        # split across the sentence ("what dimension is the polytope", "let p
        # be a polytope ... its dimension"). Allowed only for terms in
        # _SHORT_COOC, since for a generic 2-word term one word is often
        # ubiquitous ("polytope") and the pair would over-fire.
        if not cooc and key in _SHORT_COOC:
            distinct = [w for w in pt if w not in _COOC_STOP]
            cooc = len(distinct) >= 2 and set(distinct) <= mset
        if _has(pt) or cooc:
            ws = set(pt)
            if key not in matched or len(ws) > len(matched[key]):
                matched[key] = ws
    keys = set(matched)
    # specificity: drop a key whose matching words are a PROPER SUBSET of
    # another matched key's words -- the more-qualified phrase is the intended
    # one ('points NOT interior to facets' over 'points interior to facets';
    # '2-face lattice points' over 'lattice points'). Prevents a negation or
    # qualifier from also matching the bare term.
    drop = {k1 for k1 in keys for k2 in keys
            if k1 != k2 and matched[k1] < matched[k2]}
    return keys - drop


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
def glossary_context(message: str, max_terms: int = 4) -> str:
    """Scan a message for glossary terms (as whole-token phrases) and return
    their definitions + recipes as a context block, or "" if none. The harness
    appends this to a user message so the model gets the translation without
    having to recognize it should look the term up. Conservative by design:
    when it misses a term, the cy_glossary tool is the backup."""
    keys = sorted(_matched_keys(message),
                  key=lambda k: -len(k))[:max_terms]
    if not keys:
        return ""
    lines = ["(CYTools glossary -- terms detected in this request, with the "
             "recipe to use:)"]
    for k in keys:
        definition, recipe, _syns = _GLOSSARY[k]
        lines.append(f"- {k}: {definition} Recipe: {recipe}")
    return "\n".join(lines)

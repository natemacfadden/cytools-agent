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
# Description:  Machine-checkable identities between CYTools quantities,
#               derived from the cytools source (file:line cited per entry)
#               and verified numerically against the KS database before
#               admission (python -m eval.verify_glossary).
#
#               Trust model: models are helpful but untrustworthy -- these
#               checks audit any agent's computed values without an LLM.
#               Scope (honest): invariants catch wrong-formula computation,
#               cytools regressions, and constraint violations. They cannot
#               catch relabeled-but-self-consistent geometry -- that is the
#               content-id layer's job (polytope.content_id).
#
#               Each invariant: name, source (cytools file:line or theorem),
#               applies(p) -> bool, check(p) -> bool. Checks take a cytools
#               Polytope and recompute both sides independently.
# -----------------------------------------------------------------------------

# human-read (entire module)


def _pnif(p):
    return len(p.points_not_interior_to_facets())


def _h11_correction(p):
    """The non-favorable correction: sum over 2-faces of
    (interior points of the face) x (interior points of its dual face).
    Source: cytools polytope.py hpq(), p==1 branch."""
    return sum(len(f.interior_points()) * len(f.dual().interior_points())
               for f in p.faces(2))


INVARIANTS = [
    {
        "name": "h11_full_formula",
        "source": "cytools polytope.py:2991-2995 (hpq, p=q=1 branch)",
        "statement": "h11_N == sum_{2-faces} int(f)*int(dual f) "
                     "+ |points_not_interior_to_facets| - d - 1",
        "applies": lambda p: p.dim() == 4 and p.is_reflexive(),
        "check": lambda p: int(p.h11(lattice="N"))
                 == _h11_correction(p) + _pnif(p) - p.dim() - 1,
    },
    {
        "name": "h11_favorable",
        "source": "user-supplied; special case of h11_full_formula with "
                  "zero correction",
        "statement": "favorable_N => h11_N == |pnif| - d - 1",
        "applies": lambda p: p.dim() == 4 and p.is_reflexive()
                   and p.is_favorable(lattice="N"),
        "check": lambda p: int(p.h11(lattice="N")) == _pnif(p) - p.dim() - 1,
    },
    {
        "name": "favorable_iff_zero_correction",
        "source": "definition of N-favorability (non-toric divisors counted "
                  "by the correction term)",
        "statement": "favorable_N <=> sum_{2-faces} int(f)*int(dual f) == 0",
        "applies": lambda p: p.dim() == 4 and p.is_reflexive(),
        "check": lambda p: bool(p.is_favorable(lattice="N"))
                 == (_h11_correction(p) == 0),
    },
    {
        "name": "mirror_swap",
        "source": "hpq formula symmetry under dualization (Batyrev)",
        "statement": "h11_N(p) == h21_N(p.dual()) and vice versa",
        "applies": lambda p: p.dim() == 4 and p.is_reflexive(),
        "check": lambda p: (int(p.h11(lattice="N"))
                            == int(p.dual().h21(lattice="N"))
                            and int(p.h21(lattice="N"))
                            == int(p.dual().h11(lattice="N"))),
    },
    {
        "name": "double_dual_identity",
        "source": "polar duality is an involution on reflexive polytopes",
        "statement": "content_id(p.dual().dual()) == content_id(p)",
        "applies": lambda p: p.is_reflexive(),
        "check": lambda p: __import__(
            "cytools_agent.tools.polytope", fromlist=["content_id"]
        ).content_id(p.dual().dual()) == __import__(
            "cytools_agent.tools.polytope", fromlist=["content_id"]
        ).content_id(p),
    },
    {
        "name": "prime_divisor_count",
        "source": "cytools polytope.py (prime toric divisors = boundary "
                  "points not interior to facets); pnif includes the origin",
        "statement": "|boundary_pts_not_int_to_facets| == |pnif| - 1",
        "applies": lambda p: p.dim() == 4 and p.is_reflexive(),
        "check": lambda p: len(p.boundary_points_not_interior_to_facets())
                 == _pnif(p) - 1,
    },
    {
        "name": "genera_count_matches_2faces",
        "source": "structural: one genus per 2-face",
        "statement": "len(genera_2face) == number of 2-faces",
        "applies": lambda p: p.dim() == 4,
        "check": lambda p: len(p.dual().faces(1)) == len(p.faces(2)),
    },
    {
        "name": "reflexive_single_interior_point",
        "source": "definition of reflexivity (origin the unique interior "
                  "lattice point)",
        "statement": "reflexive => exactly 1 interior lattice point",
        "applies": lambda p: p.is_reflexive(),
        "check": lambda p: len(p.interior_points()) == 1,
    },
]


# CY-level invariants: need a CY built from a triangulation; checked on a
# smaller sample because construction is the expensive part.
def _cy(p):
    return p.triangulate(make_star=True).get_cy()


CY_INVARIANTS = [
    {
        "name": "cy_hodge_match_favorable",
        "source": "cytools calabiyau.py (favorable: CY Hodge numbers equal "
                  "the polytope's)",
        "statement": "favorable_N => cy.h11 == p.h11 and cy.h21 == p.h21",
        "applies": lambda p: p.dim() == 4 and p.is_reflexive()
                   and p.is_favorable(lattice="N"),
        "check": lambda p, cy: (int(cy.h11()) == int(p.h11(lattice="N"))
                                and int(cy.h21()) == int(p.h21(lattice="N"))),
    },
    {
        "name": "cy_euler_consistency",
        "source": "chi(CY3) = 2(h11 - h21)",
        "statement": "cy.chi() == 2*(cy.h11() - cy.h21())",
        "applies": lambda p: p.dim() == 4 and p.is_reflexive()
                   and p.is_favorable(lattice="N"),
        "check": lambda p, cy: int(cy.chi())
                 == 2 * (int(cy.h11()) - int(cy.h21())),
    },
    {
        "name": "tip_curve_volumes_at_least_one",
        "source": "tip_of_stretched_cone(c=1): every wall at distance >= 1, "
                  "and curve volume = ray . t >= |ray| >= 1",
        "statement": "at t=tip (toric cone): min(curve_volumes) >= 1 - 1e-6",
        "applies": lambda p: p.dim() == 4 and p.is_reflexive()
                   and p.is_favorable(lattice="N"),
        "check": lambda p, cy: _min_curve_vol(cy) >= 1 - 1e-6,
    },
    {
        "name": "tip_volumes_positive",
        "source": "Kahler-cone interior point: all divisor volumes and the "
                  "CY volume are strictly positive",
        "statement": "at t=tip: cy_volume > 0 and all divisor_volumes > 0",
        "applies": lambda p: p.dim() == 4 and p.is_reflexive()
                   and p.is_favorable(lattice="N"),
        "check": lambda p, cy: _tip_volumes_positive(cy),
    },
]


def _tip(cy):
    import numpy as np
    mori = cy.toric_mori_cone(in_basis=True)
    t = mori.dual().tip_of_stretched_cone(1)
    return np.asarray(t, dtype=float), mori


def _min_curve_vol(cy):
    import numpy as np
    t, mori = _tip(cy)
    return float(np.min(np.asarray(mori.rays()) @ t))


def _tip_volumes_positive(cy):
    import numpy as np
    t, _ = _tip(cy)
    kappa = cy.intersection_numbers(in_basis=True, format="dense")
    ktt = np.tensordot(kappa, t, axes=([2], [0])) @ t
    return float(ktt @ t / 6) > 0 and bool(np.all(0.5 * ktt > 0))


def run_polytope_invariants(p, names=None):
    """Run every applicable polytope-level invariant on `p`. Returns
    {name: True|False|'n/a'} -- False means a VIOLATION."""
    out = {}
    for inv in INVARIANTS:
        if names and inv["name"] not in names:
            continue
        try:
            out[inv["name"]] = inv["check"](p) if inv["applies"](p) else "n/a"
        except Exception as e:
            out[inv["name"]] = f"error: {type(e).__name__}: {e}"
    return out


def run_cy_invariants(p, cy=None, names=None):
    """Run every applicable CY-level invariant (builds the default CY once
    unless one is supplied)."""
    out = {}
    applicable = [inv for inv in CY_INVARIANTS
                  if (not names or inv["name"] in names)]
    if not any(inv["applies"](p) for inv in applicable):
        return {inv["name"]: "n/a" for inv in applicable}
    if cy is None:
        cy = _cy(p)
    for inv in applicable:
        try:
            out[inv["name"]] = (inv["check"](p, cy) if inv["applies"](p)
                                else "n/a")
        except Exception as e:
            out[inv["name"]] = f"error: {type(e).__name__}: {e}"
    return out

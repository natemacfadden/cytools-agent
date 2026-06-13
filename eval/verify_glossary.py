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
# Description:  Admission gate for invariants and glossary recipes: every
#               invariant must hold on a sample of database polytopes
#               spanning h11 and favorability, and every glossary recipe
#               must EXECUTE against a fetched polytope. An invariant that
#               fails here is either wrong (fix or remove it) or has found a
#               cytools bug (report it) -- both are blocking.
#
#     python -m eval.verify_glossary [n_per_h11] [--cy-sample N]
#
# All functions here are human-read (developer tooling).
# -----------------------------------------------------------------------------

# external imports
import os
import sys

import eval._env  # noqa: F401  (env pins; must precede cytools_agent imports)

# local imports  (after the env pin)
from cytools_agent.tools import polytope as P                    # noqa: E402
from cytools_agent.tools.invariants import (INVARIANTS,          # noqa: E402
                                            CY_INVARIANTS,
                                            run_cy_invariants,
                                            run_polytope_invariants)
from cytools_agent.tools.glossary import _GLOSSARY               # noqa: E402
from cytools_agent.tools import code as _code                    # noqa: E402


def sample_polytopes(n_per_h11):
    """A spread across h11 and favorability."""
    ids = []
    for h11 in (2, 3, 4, 5, 6, 7, 10):
        ids += list(P.fetch_polytopes(n_per_h11, h11))
        try:   # make sure non-favorable polytopes are represented
            ids += list(P.fetch_polytopes(2, h11, favorable=False))
        except Exception:
            pass
    return list(dict.fromkeys(str(i) for i in ids))


def verify_invariants(ids, cy_sample):
    print(f"== polytope invariants on {len(ids)} polytopes ==", flush=True)
    fails = 0
    counts = {inv["name"]: 0 for inv in INVARIANTS}
    for ks in ids:
        res = run_polytope_invariants(P.get_polytope(ks))
        for name, ok in res.items():
            if ok is True:
                counts[name] += 1
            elif ok is not True and ok != "n/a":
                fails += 1
                print(f"  VIOLATION {name} on {ks}: {ok}", flush=True)
    for name, n in counts.items():
        print(f"  {name}: held on {n} applicable polytopes")

    print(f"== CY invariants on {cy_sample} polytopes ==", flush=True)
    cy_counts = {inv["name"]: 0 for inv in CY_INVARIANTS}
    for ks in ids[:cy_sample]:
        res = run_cy_invariants(P.get_polytope(ks))
        for name, ok in res.items():
            if ok is True:
                cy_counts[name] += 1
            elif ok is not True and ok != "n/a":
                fails += 1
                print(f"  VIOLATION {name} on {ks}: {ok}", flush=True)
    for name, n in cy_counts.items():
        print(f"  {name}: held on {n} applicable polytopes")
    return fails


def verify_recipes(ks):
    """Every glossary recipe must EXECUTE in the run_python namespace (with
    ks_ind and h prepared as the glossary documents)."""
    print("== glossary recipes execute ==", flush=True)
    _code.run_python(f"ks_ind = {ks!r}")
    _code.run_python("h = get_heights(ks_ind)['heights'][0]")
    fails = 0
    for term, (_d, recipe, _s) in sorted(_GLOSSARY.items()):
        code = recipe.split("#")[0].strip()
        out = _code.run_python(code)
        if "Traceback" in out:
            fails += 1
            print(f"  RECIPE FAILS [{term}]: {code[:70]}")
            print(f"    {out.strip().splitlines()[-1][:100]}")
    print(f"  {len(_GLOSSARY) - fails}/{len(_GLOSSARY)} recipes execute")
    return fails


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = int(args[0]) if args else 20
    cy_sample = (int(sys.argv[sys.argv.index("--cy-sample") + 1])
                 if "--cy-sample" in sys.argv else 8)
    ids = sample_polytopes(n)
    fails = verify_invariants(ids, cy_sample)
    fails += verify_recipes(ids[0])
    # the reference index must cover every glossary term (the module's import
    # asserts this; report it so the coverage is visible in the gate)
    from cytools_agent.tools.glossary import _SECTIONS, _GLOSSARY
    print(f"== reference index: {len(_GLOSSARY)} terms across "
          f"{len(_SECTIONS)} sections, all covered ==", flush=True)
    print(f"\n{'ALL CLEAN' if fails == 0 else f'{fails} FAILURES'} "
          f"({len(ids)} polytopes)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

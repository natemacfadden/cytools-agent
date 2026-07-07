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
# Description:  Randomized worked examples for model-facing prompts. A concrete
#               field name becomes an attractor the model drifts into, so it is
#               sampled per process (seed CYTOOLS_EXAMPLE_SEED, draw exposed via
#               EXAMPLE_CHOICES) to flatten it across sessions. All human-read.
# -----------------------------------------------------------------------------

# external imports
import os
import random

# the pool: (column_name, expression, condition_over_name). Every entry is a
# real get_polytope_info field with a sensible example condition, so any draw
# yields a correct, runnable example.
_POOL = [
    ("n_vertices", "get_polytope_info(ks_ind)['n_vertices']",
     "n_vertices <= 6"),
    ("euler", "get_polytope_info(ks_ind)['euler_characteristic']",
     "euler % 2 == 0"),
    ("n_rigid", "get_polytope_info(ks_ind)['n_rigid_divisors']",
     "n_rigid == 0"),
    ("h21_val", "get_polytope_info(ks_ind)['h21']",
     "h21_val >= 40"),
    ("n_pts", "get_polytope_info(ks_ind)['n_points']",
     "n_pts <= 12"),
    ("aut_order", "get_polytope_info(ks_ind)['automorphism_order']",
     "aut_order > 1"),
]

_seed = os.environ.get("CYTOOLS_EXAMPLE_SEED")
_rng = random.Random(int(_seed)) if _seed else random.Random()

# one independent draw per prompt site, fixed for the life of the process
# (so a session's prompts are self-consistent and its log can record them)
EXAMPLE_CHOICES = {
    "map_cheat": _rng.choice(_POOL),
    "search_compile": _rng.choice(_POOL),
    "expr_error": _rng.choice(_POOL),
}


def example(site: str) -> tuple:
    """(name, expression, condition) drawn for this prompt site."""
    return EXAMPLE_CHOICES[site]

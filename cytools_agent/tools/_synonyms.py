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
# Description:  Forgive reasonable parameter-name synonyms the model reaches for.
#               The tool SCHEMA still advertises the canonical name (here
#               `ks_ind`); this is a silent accept-layer so a model that uses a
#               natural variant (e.g. `ks`) is not rejected with a TypeError.
#
#               GROW THIS by adding a word to a set below whenever we OBSERVE the
#               model use a new reasonable synonym -- accommodating the model's
#               unambiguous intent instead of demanding a strict ruleset.
#               All human-read.
# -----------------------------------------------------------------------------

# external imports
import functools
import inspect

# canonical parameter name -> reasonable variants the model may type instead.
# Only `ks_ind` is seeded (we observed the model use `ks`); the rest are
# unambiguous ways to say "a polytope id". Add more here as we see them.
PARAM_SYNONYMS = {
    "ks_ind": {"ks", "ks_id", "polytope_id", "poly_id", "pid", "poly",
               "polytope"},
}


# human-read
def forgive_kwargs(fn):
    """Remap synonym kwargs to the canonical parameter `fn` actually takes.

    A synonym is remapped ONLY when its canonical name is a real parameter of
    `fn` AND that canonical was not already supplied -- so a function's own real
    parameter is never shadowed (e.g. get_heights keeps its real `n`). The
    wrapped function keeps its signature and docstring via functools.wraps, so
    `function_to_schema` still advertises the canonical name to the model.
    """
    params = set(inspect.signature(fn).parameters)
    alias_to_canon = {alias: canon
                      for canon, aliases in PARAM_SYNONYMS.items()
                      if canon in params
                      for alias in aliases if alias not in params}
    if not alias_to_canon:
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        for alias, canon in alias_to_canon.items():
            if alias in kwargs and canon not in kwargs:
                kwargs[canon] = kwargs.pop(alias)
        return fn(*args, **kwargs)

    return wrapper

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
# Description:  Forgive reasonable parameter-name synonyms the model reaches
#               for: the schema still advertises the canonical name, but a
#               natural variant (e.g. `ks` for `ks_ind`) is silently accepted
#               instead of raising a TypeError. Grow the sets below as observed.
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
    """Make a tool tolerant of how the model actually calls it:

    (1) accept reasonable synonyms for the canonical parameter names (only when
        the canonical is a real parameter of `fn` and was not already supplied,
        so a real parameter, e.g. get_heights's `n`, is never shadowed); and
    (2) on an argument-binding mistake (wrong positional order, a positional/
        keyword mix-up, a missing or unexpected argument), raise a pointed error
        that reminds the model of this function's argument order and shows the
        keyword form to use, instead of Python's opaque "got multiple values
        for argument 'limit'". The model then fixes its own call.

    functools.wraps keeps the signature/docstring, so `function_to_schema` still
    advertises the canonical names to the model.
    """
    sig = inspect.signature(fn)
    names = list(sig.parameters)
    alias_to_canon = {alias: canon
                      for canon, aliases in PARAM_SYNONYMS.items()
                      if canon in names
                      for alias in aliases if alias not in names}
    required = [p.name for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty]
    example = f"{fn.__name__}(" + ", ".join(f"{r}=..." for r in required) + ")"
    order = ", ".join(names)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        for alias, canon in alias_to_canon.items():
            if alias in kwargs and canon not in kwargs:
                kwargs[canon] = kwargs.pop(alias)
        try:
            sig.bind(*args, **kwargs)        # validate the call shape only
        except TypeError as e:
            raise TypeError(
                f"{e}. {fn.__name__} takes its arguments in this order: "
                f"({order}). Pass them BY KEYWORD to avoid mix-ups, e.g. "
                f"{example}.") from None
        return fn(*args, **kwargs)

    return wrapper

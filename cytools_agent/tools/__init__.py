from . import code as _code
from .polytope import fetch_polytopes, get_polytope_info, ks_stats
from .triangulation import get_heights, get_triangulation_info
from .cy import (get_cy, get_cy_info, get_cy_cones,
                 find_kahler_for_divisor_volumes)
from .code import run_python, cytools_help
from .glossary import cy_glossary, reference
from .mapping import (MAP_TOOLS_ENABLED, compute_for_each, make_plot,
                      search_polytopes)

# `reference` is defined in glossary, not in the base run_python namespace;
# add it so model-written code can call it too.
_code._NS.setdefault("reference", reference)

# the model-facing tool set -- single source of truth so the in-house agent and
# the MCP server expose identical tools (get_cy/get_polytope are run_python
# namespace helpers, not standalone tools; save_history is Agent-session bound).
MODEL_TOOLS = [fetch_polytopes, get_polytope_info, ks_stats, get_heights,
               get_triangulation_info, get_cy_info, get_cy_cones,
               find_kahler_for_divisor_volumes,
               run_python, cytools_help, cy_glossary, reference]
if MAP_TOOLS_ENABLED:     # A/B arm: harness-side iteration + plotting
    MODEL_TOOLS += [compute_for_each, make_plot, search_polytopes]

__all__ = ['fetch_polytopes', 'get_polytope_info', 'ks_stats', 'get_heights',
           'get_triangulation_info', 'get_cy', 'get_cy_info', 'get_cy_cones',
           'find_kahler_for_divisor_volumes',
           'run_python', 'cytools_help', 'cy_glossary', 'reference',
           'MODEL_TOOLS']

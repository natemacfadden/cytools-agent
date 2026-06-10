from .polytope import fetch_polytopes, get_polytope_info, ks_stats
from .triangulation import get_heights, get_triangulation_info
from .cy import get_cy, get_cy_info, get_cy_cones
from .code import run_python, cytools_help
from .glossary import cy_glossary
from .mapping import (MAP_TOOLS_ENABLED, compute_for_each, make_plot)

# the model-facing tool set -- single source of truth so the in-house agent and
# the MCP server expose identical tools (get_cy/get_polytope are run_python
# namespace helpers, not standalone tools; save_history is Agent-session bound).
MODEL_TOOLS = [fetch_polytopes, get_polytope_info, ks_stats, get_heights,
               get_triangulation_info, get_cy_info, get_cy_cones,
               run_python, cytools_help, cy_glossary]
if MAP_TOOLS_ENABLED:     # A/B arm: harness-side iteration + plotting
    MODEL_TOOLS += [compute_for_each, make_plot]

__all__ = ['fetch_polytopes', 'get_polytope_info', 'ks_stats', 'get_heights',
           'get_triangulation_info', 'get_cy', 'get_cy_info', 'get_cy_cones',
           'run_python', 'cytools_help', 'cy_glossary', 'MODEL_TOOLS']

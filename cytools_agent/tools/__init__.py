from . import code as _code
from . import ledger
from .polytope import fetch_polytopes, get_polytope_info, ks_stats
from .triangulation import get_heights, get_triangulation_info
from .cy import (get_cy, get_cy_info, get_cy_cones,
                 find_kahler_for_divisor_volumes)
from .code import run_python, cytools_help
from .glossary import cy_glossary, reference
from .mapping import (MAP_TOOLS_ENABLED, compute_for_each, make_plot,
                      search_polytopes)

# The evidence backbone: every curated tool is ledgered -- each top-level
# call records its exact args and structured result in a harness-written
# ledger (tools/ledger.py) that no model can author. run_python is not
# ledgered as a unit (its rows are the free-form class); the curated tools
# it calls internally are, via the rebinding below.
fetch_polytopes = ledger.wrap(fetch_polytopes)
get_polytope_info = ledger.wrap(get_polytope_info)
ks_stats = ledger.wrap(ks_stats)
get_heights = ledger.wrap(get_heights)
get_triangulation_info = ledger.wrap(get_triangulation_info)
get_cy_info = ledger.wrap(get_cy_info)
get_cy_cones = ledger.wrap(get_cy_cones)
find_kahler_for_divisor_volumes = ledger.wrap(find_kahler_for_divisor_volumes)
cy_glossary = ledger.wrap(cy_glossary)
reference = ledger.wrap(reference)
compute_for_each = ledger.wrap(compute_for_each)
make_plot = ledger.wrap(make_plot)
search_polytopes = ledger.wrap(search_polytopes)

# rebind the run_python namespace to the ledgered versions, so tool calls
# made from model-written code land in the ledger too
for _name in ("fetch_polytopes", "get_polytope_info", "ks_stats",
              "get_heights", "get_triangulation_info", "get_cy_info",
              "get_cy_cones", "find_kahler_for_divisor_volumes",
              "compute_for_each", "make_plot",
              "search_polytopes", "reference"):
    if _name in _code._NS:
        _code._NS[_name] = globals()[_name]
# reference is defined in glossary (not in the base run_python namespace);
# add it so model-written code can call it too
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

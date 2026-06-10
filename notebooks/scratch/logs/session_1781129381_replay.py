# auto-generated replay -- regenerates this session's result
# run with:  python <this_file>.py   (figures land in scratch/)
import warnings; warnings.filterwarnings('ignore')
from cytools_agent.tools import run_python

CODES = [
    'fetch_polytopes(limit=10, h11=2, h21=None, favorable=None)',
    'compute_for_each(ids, {\'h11\': "get_polytope_info(ks_ind)[\'h11\']", \'automorphism_order\': "get_polytope_info(ks_ind)[\'automorphism_order\']"})',
    "make_plot(kind='scatter', x='h11', y='automorphism_order')",
]

for code in CODES:
    print(run_python(code))

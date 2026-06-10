# auto-generated replay -- regenerates this session's result
# run with:  python <this_file>.py   (figures land in scratch/)
import warnings; warnings.filterwarnings('ignore')
from cytools_agent.tools import run_python

CODES = [
    'fetch_polytopes(limit=500, h11=[8], h21=None, favorable=None)',
    'compute_for_each(ids, {\'ntfe_count\': "get_heights(ks_ind)[\'shape\'][0]"})',
    'mean(ntfe_count)',
    "make_plot(kind='histogram', x='ntfe_count', y=None)",
]

for code in CODES:
    print(run_python(code))

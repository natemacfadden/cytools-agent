# Shared eval environment pins. IMPORT THIS FIRST in every eval entry point,
# before any cytools_agent import reads the env:
#  - the sampled prompt examples are seeded (arm-to-arm comparisons must not
#    confound a config change with a lucky example draw);
#  - the on-disk KS cache is enabled (a dev feature, off by default for end
#    users; evals re-run identical queries constantly).
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CYTOOLS_EXAMPLE_SEED", "0")
os.environ.setdefault("CYTOOLS_AGENT_KS_CACHE",
                      os.path.join(_REPO, "scratch", "ks_cache.json"))
# Read-only TRUSTED BASE layer (used if the file exists). The old corpus reads
# its polytopes from here; nothing -- not even a model-driven eval -- writes to
# it, so those answers cannot regress from cache poisoning. New polytopes a run
# discovers land in the writable overlay (CYTOOLS_AGENT_KS_CACHE) instead. The
# overlay is also self-healing: every process prunes dangling fetch records on
# load. Freeze/refresh the base by copying a verified-clean overlay over it.
os.environ.setdefault("CYTOOLS_AGENT_KS_BASE",
                      os.path.join(_REPO, "scratch", "ks_base.json"))

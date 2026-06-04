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
# Description:  Scaling experiment: agent performance vs model size. Generates
#               ~50 OBJECTIVE cases (ground truth computed from ks_counts and
#               cytools), tagged by difficulty tier T1-T5, runs each once per
#               model, and reports/plots pass-rate vs model size (B params),
#               one curve per tier.
#
# Usage (cytools-agent env, Ollama serving the models):
#     python eval/scaling.py qwen3:4b,qwen3:8b,qwen3:14b,qwen3:32b [timeout_s]
# -----------------------------------------------------------------------------

# external imports
import json
import os
import re
import signal
import sys
import time

import matplotlib.pyplot as plt   # cytools_agent.tools sets the Agg backend
from openai import OpenAI

# local imports
from cytools_agent.tools import (polytope, triangulation, cy, code, files,
                                 history)
from cytools_agent.schema import function_to_schema
from cytools_agent.agent import Agent
from cytools_agent.prompt import DEFAULT_SYSTEM_PROMPT

DEFAULT_MODELS = "qwen3:4b,qwen3:8b"
MODELS = (sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODELS).split(",")
MODELS.sort(key=lambda m: int(re.search(r"(\d+)b", m.lower()).group(1)))
TIMEOUT = int(sys.argv[2]) if len(sys.argv) > 2 else 240

base = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
client = OpenAI(base_url=base + "/v1", api_key="ollama")

TOOL_FNS = [
    polytope.fetch_polytopes, polytope.get_polytope_info, polytope.ks_stats,
    triangulation.get_heights, triangulation.get_triangulation_info,
    cy.get_cy_info, cy.get_cy_cones,
    code.run_python, code.cytools_help,
    files.read_file, history.save_history,
]
tools = [function_to_schema(fn) for fn in TOOL_FNS]
tool_impls = {fn.__name__: fn for fn in TOOL_FNS}


# BaseException (not Exception) so the agent's `except Exception` around tool
# calls does not swallow the alarm and let a run blow past the timeout
class _TimedOut(BaseException):
    pass


signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(_TimedOut()))


def run_chat(model, prompt, max_steps=12):
    """One agent turn with a wall-clock timeout; returns the answer text."""
    ag = Agent(client, model, DEFAULT_SYSTEM_PROMPT, tools, tool_impls,
               max_steps=max_steps, verbosity=0)
    signal.alarm(TIMEOUT)
    try:
        return (ag.chat(prompt) or "").strip().replace("\n", " ")
    except _TimedOut:
        return "(timed out)"
    finally:
        signal.alarm(0)


# objective answer checks (graded on the final text only)
# -------------------------------------------------------
def _num(truth):
    """The exact integer `truth` appears (commas ignored)."""
    s = str(int(truth))
    return lambda ans: s in ans.replace(",", "")


def _floatish(truth):
    """A 1- or 2-dp rounding of `truth` appears."""
    cands = {f"{round(truth, d)}" for d in (1, 2)}
    return lambda ans: any(c in ans for c in cands)


def _ids(ids):
    """Every id appears."""
    return lambda ans: all(i in ans for i in ids)


def _kw(*words):
    """Any of `words` appears (case-insensitive)."""
    return lambda ans: any(w in ans.lower() for w in words)


# case generation (ground truth computed here, at build time)
# -----------------------------------------------------------
def gen_cases():
    P, T, C = polytope, triangulation, cy
    cases = []

    def add(tier, label, prompt, check):
        cases.append({"tier": tier, "label": label, "prompt": prompt,
                      "check": check})

    # ---- T1: counts / existence via ks_stats (all numeric) ----
    for h in (5, 12, 27, 42, 100, 200, 300, 433, 491):
        add("T1", f"count h11={h}",
            f"How many 4d reflexive polytopes have h11={h}?",
            _num(P.ks_stats(h)["count"]))
    for h in (495, 600):
        add("T1", f"count h11={h} (none)",
            f"How many 4d reflexive polytopes have h11={h}?", _num(0))

    # ---- small-polytope pool for the compute tiers ----
    pool = []  # (ks, h11, info, ntfe_heights)
    for h11 in (2, 3, 4):
        for ks in P.fetch_polytopes(8, h11):
            try:
                hs = T.get_heights(ks)
            except Exception:
                continue
            pool.append((ks, h11, P.get_polytope_info(ks), hs))

    # ---- T2: single-fact retrieval ----
    for ks, _h, info, _hs in pool[:4]:
        add("T2", f"h21 of {ks}", f"What is h21 of the polytope {ks}?",
            _num(info["h21"]))
    for ks, _h, _i, hs in pool[4:8]:
        add("T2", f"NTFEs of {ks}",
            f"How many inequivalent triangulations (NTFEs) does polytope "
            f"{ks} have?", _num(hs["shape"][0]))
    for h11 in (3, 4, 5):
        add("T2", f"fetch 3 h11={h11}",
            f"Fetch the first 3 polytopes at h11={h11} and list their ids.",
            _ids(P.fetch_polytopes(3, h11)))
    for ks, _h, info, _hs in pool[:3]:
        add("T2", f"n_points {ks}",
            f"How many lattice points does polytope {ks} have?",
            _num(info["n_points"]))

    # ---- T3: aggregation over the first M ----
    for h11, M in ((2, 10), (3, 10), (4, 5)):
        ids = P.fetch_polytopes(M, h11)
        favs = sum(P.get_polytope_info(i)["favorable_N"] for i in ids)
        add("T3", f"frac fav h11={h11} M={M}",
            f"Among the first {M} polytopes at h11={h11}, what fraction are "
            f"N-favorable?", _floatish(favs / len(ids)))
    for h11, M in ((2, 5), (3, 5), (4, 5)):
        ids = P.fetch_polytopes(M, h11)
        avg = sum(T.get_heights(i)["shape"][0] for i in ids) / len(ids)
        add("T3", f"avg NTFE h11={h11} M={M}",
            f"What is the average number of NTFEs over the first {M} "
            f"polytopes at h11={h11}?", _floatish(avg))
    for h11, M in ((2, 5), (3, 5)):
        ids = P.fetch_polytopes(M, h11)
        mx = max(max(T.get_triangulation_info(i, h)["n_simplices"]
                     for h in T.get_heights(i)["heights"]) for i in ids)
        add("T3", f"max simp h11={h11} M={M}",
            f"What is the maximum simplex count among triangulations of the "
            f"first {M} polytopes at h11={h11}?", _num(mx))

    # ---- T4: CY pipeline (favorable, single-NTFE ids => unique CY) ----
    fav1 = [(ks, hs) for ks, _h, info, hs in pool
            if info["favorable_N"] and hs["shape"][0] == 1]
    for ks, hs in fav1[:5]:
        v = C.get_cy_info(ks, hs["heights"][0], t="tip")["cy_volume"]
        add("T4", f"CY vol {ks}",
            f"Compute the CY volume at the tip of the stretched Kahler cone "
            f"for polytope {ks}.", _floatish(v))
    for ks, hs in fav1[:4]:
        rays = len(C.get_cy_cones(ks, hs["heights"][0],
                                  cone="toric")["mori_rays"])
        add("T4", f"mori rays {ks}",
            f"How many rays does the toric Mori cone of the Calabi-Yau from "
            f"polytope {ks} have?", _num(rays))

    # ---- T5: compositional + should-fail-gracefully ----
    for h11, M in ((3, 8), (4, 8)):
        ids = P.fetch_polytopes(M, h11)
        c = sum(1 for i in ids
                if P.get_polytope_info(i)["favorable_N"]
                and T.get_heights(i)["shape"][0] == 1)
        add("T5", f"fav&1NTFE h11={h11} M={M}",
            f"Among the first {M} polytopes at h11={h11}, how many are BOTH "
            f"N-favorable AND have exactly one NTFE?", _num(c))
    add("T5", "h11=491 all triangulations (infeasible)",
        "Construct all triangulations of the polytope at h11=491.",
        _kw("too", "cannot", "can't", "not possible", "infeasible",
            "large", "many"))
    add("T5", "h11=600 count (none)",
        "How many polytopes are there at h11=600?", _num(0))

    return cases


def report(data, sizes, cases):
    """Print the tier x size matrix, save JSON, and plot the curves."""
    tiers = sorted({c["tier"] for c in cases})
    print("\n###### pass-rate by tier vs model size ######", flush=True)
    print("tier".ljust(6) + "".join(f"{m}({sizes[m]}B)".rjust(16)
                                     for m in MODELS), flush=True)
    rates = {m: {} for m in MODELS}
    for t in tiers:
        row = t.ljust(6)
        for m in MODELS:
            p, n = data[m][t]
            rates[m][t] = p / n if n else 0.0
            row += f"{p}/{n}".rjust(16)
        print(row, flush=True)

    out = {"models": MODELS, "sizes": sizes,
           "data": {m: {t: data[m][t] for t in tiers} for m in MODELS}}
    with open(os.path.join(os.path.dirname(__file__),
                           "scaling_results.json"), "w") as f:
        json.dump(out, f, indent=1)

    xs = [sizes[m] for m in MODELS]
    for t in tiers:
        plt.plot(xs, [rates[m][t] for m in MODELS], marker="o", label=t)
    overall = [sum(data[m][t][0] for t in tiers)
               / sum(data[m][t][1] for t in tiers) for m in MODELS]
    plt.plot(xs, overall, marker="s", color="k", lw=2.5, label="overall")
    plt.xlabel("model size (B params)")
    plt.ylabel("pass rate")
    plt.ylim(0, 1.02)
    plt.legend()
    plt.title("CYTools-agent performance vs model size")
    path = os.path.join(os.path.dirname(__file__), "scaling_curve.png")
    plt.savefig(path, bbox_inches="tight")
    print(f"\nsaved curve -> {path}", flush=True)


def main():
    t0 = time.monotonic()
    sizes = {m: int(re.search(r"(\d+)b", m.lower()).group(1)) for m in MODELS}
    cases = gen_cases()
    tiers = sorted({c["tier"] for c in cases})
    print(f"generated {len(cases)} cases: "
          + ", ".join(f"{t}={sum(c['tier'] == t for c in cases)}"
                      for t in tiers), flush=True)

    data = {}
    for model in MODELS:
        print(f"\n###### {model} ######", flush=True)
        per_tier = {t: [0, 0] for t in tiers}
        for c in cases:
            ans = run_chat(model, c["prompt"])
            ok = bool(c["check"](ans))
            per_tier[c["tier"]][1] += 1
            per_tier[c["tier"]][0] += ok
            print(f"  [{c['tier']}] {'PASS' if ok else 'FAIL'} | "
                  f"{c['label'][:40]:40} | {ans[:45]}", flush=True)
        data[model] = per_tier
        print("  " + ", ".join(f"{t} {p}/{n}"
                               for t, (p, n) in per_tier.items()), flush=True)

    report(data, sizes, cases)
    print(f"\n###### total {time.monotonic() - t0:.0f}s ######", flush=True)


if __name__ == "__main__":
    main()

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
# Description:  End-to-end agent test suite. Each case pairs a prompt with
#               per-dimension graders -- answer correctness, tool-call behavior,
#               honesty, etc. -- that may read the answer text and/or the
#               agent's tool-call trace. Each case runs N times (a local model
#               is not deterministic) and the per-dimension pass rates are
#               reported. Covers both the baseline regression tasks and the
#               higher-level behavioral cases.
#
# Usage (in the cytools-agent env, with Ollama serving the model):
#     python eval/agent_tests.py qwen3:4b 5
# -----------------------------------------------------------------------------

# external imports
import json
import os
import re
import signal
import sys
import time

# local imports
from cytools_agent.tools import polytope
from eval._harness import make_agent, _TimedOut

MODELS = (sys.argv[1] if len(sys.argv) > 1 else "qwen3:4b").split(",")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2
TIMEOUT = int(sys.argv[3]) if len(sys.argv) > 3 else 240


# grading helpers: read a finished agent's trace
# ----------------------------------------------
def _msg_calls(msg):
    """The tool_calls on a message, be it an SDK object or our rebuilt dict."""
    if isinstance(msg, dict):
        return msg.get("tool_calls") or []
    return getattr(msg, "tool_calls", None) or []


def tool_calls(agent):
    """Every (name, args) the agent invoked, real or recovered from text."""
    out = []
    for m in agent.messages:
        for c in _msg_calls(m):
            if isinstance(c, dict):
                name, raw = c["function"]["name"], c["function"]["arguments"]
            else:
                name, raw = c.function.name, c.function.arguments
            out.append((name, json.loads(raw)))
    return out


def tool_names(agent):
    """The set of tool names the agent invoked."""
    return {name for name, _ in tool_calls(agent)}


def ran_code(agent):
    """The concatenated source the agent passed to run_python."""
    return "\n".join(args.get("code", "") for name, args in tool_calls(agent)
                     if name == "run_python")


def called(agent, name, **arg_eq):
    """True if the agent invoked `name` with args matching all of arg_eq."""
    return any(n == name and all(a.get(k) == v for k, v in arg_eq.items())
               for n, a in tool_calls(agent))


def used(agent, name):
    """True if `name` ran as a direct call or appears in run_python code."""
    return name in tool_names(agent) or name in ran_code(agent)


# graders + fixtures
# ------------------
# triangulation/CY tools, for asserting a "just fetch" task did NOT compute
COMPUTE = {"get_heights", "get_triangulation_info",
           "get_cy_info", "get_cy_cones"}

# pinned vertices of the case-1 favorable polytopes (their canonical identity)
with open(os.path.join(os.path.dirname(__file__), "case1_verts.json")) as f:
    CASE1_VERTS = json.load(f)


def _has(ans, *subs):
    """True if any of `subs` appears in `ans` (case-insensitive)."""
    low = ans.lower()
    return any(s in low for s in subs)


def _asked(ans):
    """True if the reply is a clarifying question (a fine, non-failing move)."""
    return _has(ans, "how many", "would you like", "need to know",
                "not specified", "specify the limit", "please specify")


def _verts_match(verts_by_id):
    """True if each id reconstructs to its pinned vertices (right polytope)."""
    try:
        return all(polytope.get_polytope(i).vertices().tolist() == v
                   for i, v in verts_by_id.items())
    except KeyError:
        return False


# Each case: label, prompt, max_steps, and a dict of dimension -> check. A check
# is check(answer, agent) -> bool, free to read the answer text and/or the
# agent's tool-call trace (via the helpers above).
CASES = [
    # ---- baseline regression tasks (answer-only) ----
    {
        "label": "fetch 3 at h11=5",
        "prompt": "Fetch 3 polytopes at h11=5",
        "max_steps": 6,
        "checks": {"answer": lambda ans, ag: all(i in ans for i in (
            "h11-5_h21-20_ind-0", "h11-5_h21-29_ind-0",
            "h11-5_h21-29_ind-1"))},
    },
    {
        "label": "favorable fraction at h11=2 (sample 20)",
        "prompt": "What fraction of polytopes at h11=2 are N-favorable? "
                  "Use a sample of 20.",
        "max_steps": 10,
        "checks": {"answer": lambda ans, ag: _has(ans, "1.0", "100%",
                                                   "20/20", "all 20")},
    },
    {
        "label": "average NTFEs at h11=2 (sample 5)",
        "prompt": "On average, how many NTFEs do polytopes at h11=2 have? "
                  "Sample 5.",
        "max_steps": 10,
        "checks": {"answer": lambda ans, ag: "1.2" in ans},
    },
    {
        "label": "max simplex count at h11=3 (sample 3)",
        "prompt": "What is the maximum simplex count among triangulations of "
                  "polytopes at h11=3? Sample 3 polytopes.",
        "max_steps": 14,
        "checks": {"answer": lambda ans, ag: "12" in ans},
    },
    # ---- higher-level behavioral cases ----
    {
        "label": "h11=5 volumes (one favorable, two to skip)",
        "prompt": "Fetch 3 polytopes at h11=5. For each, compute the CY "
                  "volume at the tip of the stretched Kahler cone.",
        "max_steps": 15,
        "checks": {
            # only the favorable CY has a volume, and it is ~33.6
            "answer": lambda ans, ag: "33.6" in ans,
            # actually computed it, and did not hardcode a point t
            "tools": lambda ans, ag: (
                {"run_python", "get_cy_info"} & tool_names(ag)
                and "t=[" not in ran_code(ag).replace(" ", "")
            ),
            # no fabricated volumes: every decimal reported is the real ~33.6
            "honest": lambda ans, ag: "33.6" in ans and all(
                f.startswith("33.6") for f in re.findall(r"\d+\.\d+", ans)),
        },
    },
    {
        # 1) just fetch favorable ids, nothing more
        "label": "fetch 5 favorable at (20, 60)",
        "prompt": "fetch 5 favorable polytopes at h11=20, h21=60",
        "max_steps": 6,
        "checks": {
            # the 5 favorable ids at (20,60) are ind 5, 14, 15, 16, 20
            "answer": lambda ans, ag: all(f"h11-20_h21-60_ind-{i}" in ans
                                          for i in (5, 14, 15, 16, 20)),
            # and those ids really are the expected polytopes (by vertices)
            "vertices": lambda ans, ag: _verts_match(CASE1_VERTS),
            # one favorable fetch, and NOT triangulated/analyzed
            "tools": lambda ans, ag: (
                called(ag, "fetch_polytopes", h11=20, h21=60, favorable=True)
                and not (tool_names(ag) & COMPUTE)),
        },
    },
    {
        # count via the ks_stats lookup, not by fetching millions
        "label": "count polytopes at h11=42 (ks_stats)",
        "prompt": "how many polytopes have h11=42?",
        "max_steps": 6,
        "hint": "call ks_stats(42) -- don't try to fetch and count them",
        "checks": {
            # truth: 8,391,799
            "answer": lambda ans, ag: "8391799" in ans.replace(",", ""),
            # used the lookup, did not try to enumerate via fetch
            "tools": lambda ans, ag: (
                used(ag, "ks_stats")
                and "fetch_polytopes" not in tool_names(ag)),
        },
    },
    {
        # 2) fetch all (there are 4, all h21=13) and describe each
        "label": "fetch all at h11=433, describe each",
        "prompt": "fetch all polytopes at h11=433 and describe them 1-by-1",
        "max_steps": 12,
        "checks": {
            # described them (all four share h21=13) -- or sensibly asked
            "answer": lambda ans, ag: _asked(ans) or ("13" in ans
                                                      and "433" in ans),
            # fetched then pulled per-polytope info -- or asked first
            "tools": lambda ans, ag: _asked(ans) or (
                used(ag, "fetch_polytopes")
                and used(ag, "get_polytope_info")),
        },
    },
    {
        # 3) infeasible -- should fail loudly, not fabricate
        "label": "h11=491 construct all triangulations (should fail)",
        "prompt": "fetch the polytope at h11=491 and construct all "
                  "triangulations of it",
        "max_steps": 8,
        "checks": {
            # it tried to enumerate triangulations -- or asked how many first
            "tools": lambda ans, ag: (_asked(ans)
                                      or used(ag, "get_heights")),
            # reported infeasibility (or asked), and did NOT falsely claim the
            # polytope does not exist (it does)
            "honest": lambda ans, ag: (
                (_asked(ans) or _has(ans, "too large", "too many", "too hard",
                 "cannot", "can't", "too big", "not possible", "infeasible",
                 "error"))
                and not _has(ans, "does not exist", "no polytope",
                             "not in the database", "maximum h11")),
        },
    },
    {
        # 4) NTFE counts + a simplex-count histogram
        "label": "h11=6 first 10: NTFE counts + simplex histogram",
        "prompt": "for the first 10 polytopes at h11=6, how many NTFEs do "
                  "each of them have? plot me a distribution of simplex counts",
        "max_steps": 18,
        "hint": "per polytope, the NTFE count is get_heights(id)['shape'][0]; "
                "do not conflate it with the simplex count",
        "checks": {
            # reports the actual NTFE counts, not just the word "NTFE"
            "answer": lambda ans, ag: (
                "1,3,6,1,2,2,4,4,2,4" in ans.replace(" ", "")),
            # looped the right tools in code
            "tools": lambda ans, ag: (
                "run_python" in tool_names(ag)
                and "get_heights" in ran_code(ag)
                and _has(ran_code(ag), "simplic", "get_triangulation_info")),
            # actually drew a histogram
            "plot": lambda ans, ag: _has(ran_code(ag), "hist", "savefig",
                "plt.", "matplotlib"),
        },
    },
    {
        # 5) sample Kahler-cone points, distribution of divisor volumes
        "label": "h11=40: sample Kahler points, divisor-volume histogram",
        "prompt": "for a polytope at h11=40, triangulate it, sample points in "
                  "its kahler cone, and give me a distribution of divisor "
                  "volumes at each point",
        "max_steps": 20,
        "hint": "Kcup is very expensive at h11=40; pass cone='toric' to "
                "get_cy_cones / get_cy_info for a cheap Kahler cone",
        "checks": {
            # reports a divisor-volume distribution
            "answer": lambda ans, ag: _has(ans, "divisor"),
            # triangulated (random, since h11=40 is too big for all NTFEs) and
            # evaluated the CY at sampled points
            "tools": lambda ans, ag: (
                "run_python" in tool_names(ag)
                and used(ag, "get_heights")
                and "get_cy_info" in ran_code(ag)
                and _has(ran_code(ag), "kahler", "mori", "tip_of_stretched",
                         "get_cy_cones")),
            # actually drew a histogram
            "plot": lambda ans, ag: _has(ran_code(ag), "hist", "savefig",
                "plt.", "matplotlib"),
        },
    },
]


def run_suite(model):
    """Run every case for one model; print it; return {label: allpass}."""
    print(f"\n###### {model}  (N={N} each) ######", flush=True)
    results = {}
    for case in CASES:
        dims = list(case["checks"])
        tally = {d: 0 for d in dims}
        allpass = 0
        case_model = case_tools = 0.0
        tool_secs = {}
        print(f"\n## [{case['label']}]", flush=True)
        for i in range(N):
            ag = make_agent(model, case["max_steps"])
            signal.alarm(TIMEOUT)
            try:
                ans = (ag.chat(case["prompt"]) or "").strip().replace("\n", " ")
            except _TimedOut:
                ans = f"(timed out after {TIMEOUT}s)"
            finally:
                signal.alarm(0)
            case_model += ag.timing["model"]
            case_tools += ag.timing["tools"]
            for k, v in ag.tool_secs.items():
                tool_secs[k] = tool_secs.get(k, 0.0) + v
            res = {d: bool(case["checks"][d](ans, ag)) for d in dims}
            for d in dims:
                tally[d] += res[d]
            ok = all(res.values())
            allpass += ok
            flags = " ".join(f"{d}={'Y' if res[d] else 'N'}" for d in dims)
            print(f"  run {i+1}: {'PASS' if ok else 'FAIL'} [{flags}] "
                  f"(model {ag.timing['model']:.0f}s / "
                  f"tools {ag.timing['tools']:.0f}s) | {ans[:70]}", flush=True)
            if not ok and case.get("hint"):
                print(f"        hint -> {case['hint']}", flush=True)
        print(f"  -> all-pass {allpass}/{N}  ("
              + ", ".join(f"{d} {tally[d]}/{N}" for d in dims) + ")",
              flush=True)
        top = sorted(tool_secs.items(), key=lambda kv: -kv[1])[:3]
        print(f"     time: model {case_model:.0f}s, tools {case_tools:.0f}s  ("
              + ", ".join(f"{k} {v:.0f}s" for k, v in top) + ")", flush=True)
        results[case["label"]] = allpass
    return results


def main():
    suite_t0 = time.monotonic()
    results = {m: run_suite(m) for m in MODELS}
    if len(MODELS) > 1:
        print("\n###### all-pass comparison (x/N) ######", flush=True)
        print("case".ljust(38) + "".join(m.rjust(12) for m in MODELS),
              flush=True)
        for case in CASES:
            lab = case["label"]
            cells = "".join(f"{results[m][lab]}/{N}".rjust(12) for m in MODELS)
            print(lab[:36].ljust(38) + cells, flush=True)
    print(f"\n###### total {time.monotonic() - suite_t0:.0f}s ######",
          flush=True)


if __name__ == "__main__":
    main()

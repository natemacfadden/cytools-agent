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
# Description:  Evidence: reading the engineer's outputs and guaranteeing their
#               accuracy. Holds the two on-disk streams a session writes -- the
#               progress events (session.jsonl) and the observations
#               (evidence.jsonl) -- plus log archiving, and the accuracy checks
#               that make an answer trustworthy: ran_code must parse, and a
#               final answer must appear in a real COMPUTED output (not a typed
#               literal). All functions here are human-read.
# -----------------------------------------------------------------------------

# external imports
import ast
import glob
import json
import os
import re
import shutil
import time

# local imports
from cytools_agent.tools import code as _code

EVIDENCE_PATH = os.path.join("scratch", "evidence.jsonl")
SESSION_PATH = os.path.join("scratch", "session.jsonl")
LOG_DIR = os.path.join("scratch", "logs")   # archived sessions, never clobbered

_STORE_CAP = 2000   # cap each stored received_output (protect the file)
_RENDER_CAP = 600   # cap each rendered received_output (protect the context)


# session events + observations (the live streams the viewer reads)
# -----------------------------------------------------------------
def reset_session(path=SESSION_PATH):
    """Start a fresh, empty session event log."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()


def emit(event, path=SESSION_PATH, **fields):
    """Append one progress event (PM step, dispatch, report, ...)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"event": event, "t": time.time(), **fields}) + "\n")


def reset_evidence(path=EVIDENCE_PATH):
    """Start a fresh, empty evidence file for a new session."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()


def write_evidence(obs_list, path=EVIDENCE_PATH):
    """Rewrite the whole evidence file from `obs_list` -- flushed after every
    change so the viewer sees each step (and filled-in interpretation) live."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for obs in obs_list:
            obs = dict(obs)
            obs["received_output"] = (obs.get("received_output") or "")[
                :_STORE_CAP]
            f.write(json.dumps(obs) + "\n")


def read_evidence(path=EVIDENCE_PATH):
    """All observations recorded so far, in order."""
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path) if line.strip()]


def read_session(path=SESSION_PATH):
    """All session progress events so far, in order."""
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path) if line.strip()]


def render_evidence(path=EVIDENCE_PATH, last=None):
    """The evidence as the text block shown to both agents (ground truth vs
    claim is labelled; received_output is truncated to bound the prompt).
    last=N renders only the most recent N observations (older ones noted as
    omitted) -- a lean view for the per-step dispatch, where the scratchpad
    summary already carries accumulated state."""
    obs = read_evidence(path)
    if not obs:
        return "(evidence log is empty)"
    shown = obs[-last:] if last else obs
    start = len(obs) - len(shown)
    out = []
    for i, o in enumerate(shown, start + 1):
        recv = o.get("received_output", "")
        if len(recv) > _RENDER_CAP:
            recv = recv[:_RENDER_CAP] + " ...(truncated)"
        out.append(
            f"#{i} intent: {o.get('intent') or '(none stated)'}\n"
            f"    ran_code: {o.get('ran_code', '')}\n"
            f"    received_output: {recv}\n"
            f"    interpretation: {o.get('interpretation') or '(none stated)'}"
        )
    head = "[Evidence log -- ran_code/received_output are ground truth; " \
           "intent/interpretation are the engineer's claims]\n"
    if start:
        head += f"...({start} earlier observation(s) omitted)...\n"
    return head + "\n".join(out)


# archiving: keep every session (and its figures + a runnable replay)
# -------------------------------------------------------------------
def save_log(user_message, answer, stamp):
    """Archive the session JSON, its figures, and a runnable replay script to
    uniquely-named files, so all survive the next run's reset. Returns the
    log path."""
    os.makedirs(LOG_DIR, exist_ok=True)
    evidence = read_evidence()
    figs = []
    for i, fig in enumerate(sorted(glob.glob(
            os.path.join(_code._FIG_DIR, "fig_*.png")))):
        dst = os.path.join(LOG_DIR, f"session_{stamp}_fig_{i}.png")
        shutil.copyfile(fig, dst)
        figs.append(dst)
    path = os.path.join(LOG_DIR, f"session_{stamp}.json")
    with open(path, "w") as f:
        json.dump({"question": user_message, "answer": answer,
                   "session": read_session(), "evidence": evidence,
                   "figures": figs}, f, indent=2)
    export_script(evidence, os.path.join(LOG_DIR, f"session_{stamp}_replay.py"))
    return path


def export_script(evidence, out_path):
    """Write a standalone replay: it re-executes every ran_code via run_python
    (real namespace, auto-saves figures), regenerating the result/plot from
    the log alone -- even after a rerun overwrote the live files."""
    lines = ["# auto-generated replay -- regenerates this session's result",
             "# run with:  python <this_file>.py   (figures land in scratch/)",
             "import warnings; warnings.filterwarnings('ignore')",
             "from cytools_agent.tools import run_python", "", "CODES = ["]
    lines += [f"    {o['ran_code']!r}," for o in evidence
              if (o.get("ran_code") or "").strip()]
    lines += ["]", "", "for code in CODES:", "    print(run_python(code))"]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


# accuracy checks -- what makes an answer trustworthy
# ---------------------------------------------------
def valid_python(src):
    """True if `src` parses as Python (ran_code must be runnable)."""
    try:
        ast.parse(src or "")
        return True
    except SyntaxError:
        return False


def _prints_only_literals(code):
    """True if every value this code prints/echoes is a literal CONSTANT (e.g.
    print(15)) -- no variable or function call. Such an 'answer' was typed,
    not computed (a fabrication like print(15)/print(0)/print(44))."""
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return False

    def is_print(v):
        return (isinstance(v, ast.Call) and isinstance(v.func, ast.Name)
                and v.func.id == "print")

    printed = [a for n in ast.walk(tree) if is_print(n) for a in n.args]
    last = tree.body[-1] if tree.body else None    # run_python echoes a bare
    if isinstance(last, ast.Expr) and not is_print(last.value):  # trailing
        printed.append(last.value)
    if not printed:
        return False
    return not any(isinstance(s, (ast.Name, ast.Call))
                   for e in printed for s in ast.walk(e))


# harness-only marker that run_python appends ONLY when matplotlib actually
# wrote a figure to disk; the engineer cannot emit it without making a real
# figure, so it is unfakable provenance for a plot deliverable
_FIG_SAVE_RE = re.compile(r"\[saved \d+ figure\(s\): ")
# an answer that is a plot/file deliverable (names a figure rather than a value)
_FIG_ANS_RE = re.compile(
    r"\.(?:png|pdf|svg)\b|\b(?:plot|figure|chart|scatter|histogram)\b", re.I)


def grounded(answer, observations):
    """True if the answer is backed by real, harness-captured ground truth.
    A PLOT/FILE deliverable (the answer names a figure) is grounded by a real
    saved-figure marker -- the figure itself is the provenance, not a printed
    number. Otherwise the answer's result (last number, else its text) must
    appear in the received_output of an observation whose code actually
    computes -- not a typed literal. Either way the answer is unfakable."""
    if _FIG_ANS_RE.search(answer or ""):       # plot deliverable: a real saved
        for o in observations:                 # figure IS the ground truth
            if _FIG_SAVE_RE.search(str(o.get("received_output", ""))):
                return True
    nums = re.findall(r"-?\d+\.?\d*", answer)
    target = nums[-1] if nums else answer.strip()[:60]
    if not target:
        return False
    for o in observations:
        out = str(o.get("received_output", ""))
        present = (target in set(re.findall(r"-?\d+\.?\d*", out))) if nums \
            else (target in out)
        if present and not _prints_only_literals(o.get("ran_code", "")):
            return True
    return False

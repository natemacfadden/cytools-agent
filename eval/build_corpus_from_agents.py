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
# Description:  Collect the Q&A that the notebook-mining sub-agents produced and
#               GATE on self-contained reproducibility: pull the final JSONL
#               from each agent transcript, then run every entry's `code` in a
#               FRESH interpreter and keep only entries whose standalone program
#               reproduces the stored answer. Writes eval/corpus.jsonl, so every
#               corpus entry's code is self-contained and correct.
# -----------------------------------------------------------------------------

# external imports
import ast
import html
import json
import os
import subprocess
from collections import Counter

PY = "/Users/natemacfadden/miniforge3/envs/cytools-agent/bin/python3"
ROOT = os.path.dirname(os.path.dirname(__file__))
TASKS = ("/private/tmp/claude-501/-Users-natemacfadden-cytools-agent/"
         "934d9ed9-728d-469b-ac40-9b402c5d6310/tasks")
AGENTS = {
    "cytools_ext": "ac7b1f8c680326ce8",
    "geom": "aca32575739167921",
    "physics": "af9b79662d3f6ce1c",
    "sm_in_IIB": "a5496b4ce304c9130",
}


def _texts(path):
    for line in open(path):
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") != "assistant":
            continue
        c = ev.get("message", {}).get("content", "")
        yield c if isinstance(c, str) else "".join(
            b.get("text", "") for b in c
            if isinstance(b, dict) and b.get("type") == "text")


def final_jsonl(path):
    best = []
    for text in _texts(path):
        rows = []
        for ln in text.splitlines():
            ln = html.unescape(ln.strip())
            if ln.startswith('{"question"'):
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
        if rows:
            best = rows
    return best


def reproduces(code, ans):
    """Run `code` standalone; True iff stdout matches `ans` (num-tolerant)."""
    if not code or "print(" not in code:
        return False, "no print"
    try:
        p = subprocess.run([PY, "-c", code], capture_output=True, text=True,
                           cwd=ROOT, timeout=180)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if p.returncode != 0:
        return False, "error: " + p.stderr.strip()[-60:]
    out = p.stdout.strip()
    try:
        val = ast.literal_eval(out)
    except (ValueError, SyntaxError):
        return (out == str(ans)), f"got {out[:40]}"
    if val == ans:
        return True, ""
    try:
        import numpy as np
        if np.allclose(np.array(val, float), np.array(ans, float), atol=1e-4):
            return True, ""
    except Exception:
        pass
    return False, f"got {out[:40]} want {ans}"


def main():
    seen, kept, dropped = set(), [], []
    for label, aid in AGENTS.items():
        for r in final_jsonl(os.path.join(TASKS, f"{aid}.output")):
            q = r.get("question")
            if not q or q in seen:
                continue
            seen.add(q)
            ok, why = reproduces(r.get("code", ""), r.get("answer"))
            if ok:
                r["agent"] = label
                kept.append(r)
            else:
                dropped.append((label, r.get("kind"), why))

    out = os.path.join(os.path.dirname(__file__), "corpus.jsonl")
    with open(out, "w") as f:
        for i, r in enumerate(kept):
            r["id"] = i
            f.write(json.dumps(r) + "\n")

    print(f"kept {len(kept)} (self-contained + reproduced), "
          f"dropped {len(dropped)} -> {out}")
    print("by agent:", dict(Counter(r["agent"] for r in kept)))
    print(f"distinct kinds: {len(set(r.get('kind') for r in kept))}")
    for d in dropped[:25]:
        print("  DROP", d)


if __name__ == "__main__":
    main()

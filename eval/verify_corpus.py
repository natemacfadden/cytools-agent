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
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  Independently double-check the corpus: re-execute each Q&A's
#               self-contained `code` snippet in a fresh interpreter and compare
#               the printed result to the stored answer. Reports match /
#               mismatch / skipped (non-self-contained) counts. (Counts are also
#               cross-checked since ks_stats matches cornell-dev's table.)
# -----------------------------------------------------------------------------

# external imports
import ast
import json
import os
import subprocess

PY = "/Users/natemacfadden/miniforge3/envs/cytools-agent/bin/python3"
CORPUS = os.path.join(os.path.dirname(__file__), "corpus.jsonl")


def _eq(out, ans):
    out = out.strip()
    try:
        val = ast.literal_eval(out)
    except (ValueError, SyntaxError):
        return out == str(ans)
    if val == ans:
        return True
    try:                                  # numeric tolerance (scalars or lists)
        import numpy as np
        return np.allclose(np.array(val, float), np.array(ans, float),
                           atol=1e-4)
    except Exception:
        return False


def main():
    rows = [json.loads(line) for line in open(CORPUS)]
    ok = mismatch = err = skip = 0
    bad = []
    for r in rows:
        code = r.get("code", "")
        # self-contained = brings its own imports AND prints a result
        if not (code.lstrip().startswith(("from ", "import "))
                and "print(" in code):
            skip += 1
            continue
        p = subprocess.run([PY, "-c", code], capture_output=True, text=True,
                           cwd=os.path.dirname(os.path.dirname(__file__)),
                           timeout=120)
        if p.returncode != 0:
            err += 1
            bad.append(("ERR", r["id"], r["kind"], p.stderr.strip()[-80:]))
            continue
        if _eq(p.stdout, r["answer"]):
            ok += 1
        else:
            mismatch += 1
            bad.append(("MISMATCH", r["id"], r["kind"],
                        f"got {p.stdout.strip()[:40]} want {r['answer']}"))

    print(f"verified {ok}, mismatch {mismatch}, error {err}, "
          f"skipped(non-self-contained) {skip}  / {len(rows)} total")
    for b in bad[:20]:
        print(" ", b)


if __name__ == "__main__":
    main()

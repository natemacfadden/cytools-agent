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
# Description:  Measured cost model for the expensive operations. Every call is
#               recorded (op, features, cold/warm, seconds) in memory, and to a
#               JSONL when the dev cache is on; estimate() answers "what will
#               this cost" from cold-path measurements (warm calls ~0). Empty
#               until measurements accrue. All human-read.
# -----------------------------------------------------------------------------

# external imports
import json
import os
import statistics
import time

_LOG = []          # in-memory this process: dicts
_PERSIST = None    # set lazily from the dev-cache env (same opt-in)


def _persist_path():
    global _PERSIST
    if _PERSIST is None:
        cache = os.environ.get("CYTOOLS_AGENT_KS_CACHE", "")
        _PERSIST = (os.path.join(os.path.dirname(cache), "cost_log.jsonl")
                    if cache else "")
    return _PERSIST


def record(op: str, seconds: float, cold: bool = True, **features):
    """One measurement. `cold` False marks cache-served/memoized calls
    (kept for confirmation, excluded from estimates)."""
    row = {"op": op, "s": round(seconds, 4), "cold": bool(cold),
           "t": time.time(), **features}
    _LOG.append(row)
    path = _persist_path()
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row) + "\n")
        except OSError:
            pass


class timed:
    """Context manager: with timed('get_heights', h11=4, kind='NTFE') as t:
    ... ; t.cold(False) downgrades to a warm measurement."""

    def __init__(self, op, **features):
        self.op, self.features, self._cold = op, features, True

    def cold(self, is_cold):
        self._cold = bool(is_cold)

    def __enter__(self):
        self._t0 = time.monotonic()
        return self

    def __exit__(self, exc_type, *_):
        record(self.op, time.monotonic() - self._t0,
               cold=self._cold and exc_type is None,
               ok=exc_type is None, **self.features)


def _rows():
    """All measurements: this process + the persisted log."""
    rows = list(_LOG)
    path = _persist_path()
    if path and os.path.exists(path):
        try:
            with open(path) as f:
                rows += [json.loads(l) for l in f if l.strip()]
        except (OSError, ValueError):
            pass
    return rows


def estimate(op: str, h11: int | None = None, strict: bool = False
             ) -> dict | None:
    """{n, median_s, p90_s} for the cold path of `op`, bucketed by h11 when
    given. strict=True returns None when the h11 bucket itself is empty (no
    falling back to all-sizes data: callers quoting 'cost at this size' must
    not be fed other sizes' numbers)."""
    rows = [r for r in _rows()
            if r["op"] == op and r.get("cold") and r.get("ok", True)]
    if h11 is not None:
        bucket = [r for r in rows if r.get("h11") == h11]
        rows = bucket if (bucket or strict) else rows
    if not rows:
        return None
    secs = sorted(r["s"] for r in rows)
    return {"n": len(secs), "median_s": round(statistics.median(secs), 3),
            "p90_s": round(secs[max(0, int(len(secs) * 0.9) - 1)], 3)}

# =============================================================================
#    Copyright (C) 2026  Nate MacFadden for the Liam McAllister Group
#    (GPL-3.0-or-later; see eval/answer.py header.)
# =============================================================================
#
# Deterministic tests for eval/answer.py, the typed committed-answer checker.
# Pure code, no model: run with `python -m pytest eval/test_answer.py` or
# `python -m eval.test_answer`. These lock the grading semantics so future
# changes cannot silently regress them.

from eval.answer import parse_final, check, build_final


def _c(kind, value, truth):
    return check({"kind": kind, "value": value}, truth)


# parse_final
# -----------
def test_parse_basic():
    assert parse_final('...<final>{"kind":"int","value":5}</final>') == \
        {"kind": "int", "value": 5}


def test_parse_last_wins():
    # an example block shown earlier must not shadow the real answer last
    t = ('example: <final>{"kind":"int","value":0}</final> ... '
         'answer <final>{"kind":"int","value":5}</final>')
    assert parse_final(t)["value"] == 5


def test_parse_multiline_and_ci():
    assert parse_final('<FINAL>{"kind":"list",\n"value":[1,2]}</FINAL>') == \
        {"kind": "list", "value": [1, 2]}


def test_parse_malformed_or_absent():
    assert parse_final("no block here") is None
    assert parse_final("<final>{not json}</final>") is None
    assert parse_final(None) is None
    # skips a malformed trailing block, falls back to the valid earlier one
    assert parse_final('<final>{"kind":"int","value":5}</final> '
                       '<final>{oops</final>')["value"] == 5


# int
# ---
def test_int():
    assert _c("int", 5, 5)
    assert _c("int", 5.0, 5)         # 5.0 counts as 5
    assert _c("int", "5", 5)         # stringified number ok
    assert not _c("int", 4, 5)
    assert not _c("int", 3, 5)       # the h11-3 digit-leak: only value matters
    assert not _c("none", None, 5)
    assert not _c("int", None, 5)


# float (checked to the truth's precision)
# ----------------------------------------
def test_float():
    assert _c("float", 1.46, 1.46)
    assert _c("float", 1.4583, 1.46)     # rounds to 1.46
    assert _c("float", "1.46", 1.46)
    assert not _c("float", 1.5, 1.46)    # rounds to 1.5, wrong
    assert not _c("float", 1.40, 1.46)
    assert _c("float", 1.25, 1.25)


# bool
# ----
def test_bool():
    assert _c("bool", True, True)
    assert _c("bool", False, False)
    assert not _c("bool", False, True)
    assert not _c("bool", 1, True)       # 1 is not the bool True here
    # a bool truth must not be satisfied by an int-kind answer
    assert not _c("int", 1, True)


# list / tuple (entry order arbitrary, entry content strict)
# ----------------------------------------------------------
def test_list():
    assert _c("list", [1, 2, 3], [1, 2, 3])
    assert _c("list", [3, 2, 1], [1, 2, 3])          # order-insensitive
    assert _c("list", [[1, 2], [3, 4]], [[1, 2], [3, 4]])
    assert _c("list", [[3, 4], [1, 2]], [[1, 2], [3, 4]])   # entry order free
    assert not _c("list", [1, 2], [1, 2, 3])         # missing entry
    assert not _c("list", [1, 2, 9], [1, 2, 3])      # wrong entry
    assert not _c("list", [[2, 1], [3, 4]], [[1, 2], [3, 4]])  # entry perm bad


# single-value column unwraps to a scalar (the id94 fix)
# ------------------------------------------------------
def test_scalar_unwrap():
    # a one-element map column is the scalar answer, however it was typed
    assert _c("list", [False], False)        # is_favorable_M for one polytope
    assert _c("bool", [True], True)
    assert _c("int", [5], 5)
    assert _c("float", [1.46], 1.46)
    assert not _c("list", [True], False)     # wrong value still fails
    # but a real list truth is not satisfied by unwrapping
    assert not _c("list", [3], [1, 2, 3])
    assert _c("list", [1, 2, 3], [1, 2, 3])  # multi-value list unchanged


# impossible negative test
# ------------------------
def test_impossible():
    assert _c("impossible", None, "IMPOSSIBLE")
    assert not _c("int", 0, "IMPOSSIBLE")            # a number is not "reported impossible"
    assert not _c("impossible", None, 5)             # impossible claim vs real truth


# build_final round-trips through parse_final
# -------------------------------------------
def test_build_roundtrip():
    for kind, value in [("int", 5), ("float", 1.46), ("list", [[1, 2], [3, 4]]),
                        ("bool", True), ("impossible", None)]:
        assert parse_final("blah " + build_final(kind, value)) == \
            {"kind": kind, "value": value}


if __name__ == "__main__":
    import sys
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    bad = 0
    for n, f in fns:
        try:
            f()
            print(f"ok   {n}")
        except AssertionError as e:
            bad += 1
            print(f"FAIL {n}: {e}")
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    sys.exit(1 if bad else 0)

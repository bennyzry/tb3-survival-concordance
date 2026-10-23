"""Differential checks against the unmodified installed 0.28.0 release.

Run from the submitted source directory with
``python tests/validate_concordance.py``.
"""

from importlib.metadata import distribution
import importlib.util
import math
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sksurv import metrics
from sksurv.util import Surv


spec = importlib.util.spec_from_file_location(
    "sksurv._release_metrics", distribution("scikit-survival").locate_file("sksurv/metrics.py")
)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def outcome(function, args, kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = function(*args, **kwargs)
        except Exception as error:
            result = error
    return result, {w.category.__name__ for w in caught}


def compare(name, *args, **kwargs):
    actual, actual_warnings = outcome(getattr(metrics, name), args, kwargs)
    expected, expected_warnings = outcome(getattr(release, name), args, kwargs)
    context = (name, args, kwargs, actual, expected, actual_warnings, expected_warnings)
    if isinstance(expected, Exception):
        assert isinstance(actual, type(expected)), context
    else:
        assert not isinstance(actual, Exception), context
        assert actual[1:] == expected[1:], context
        assert all(isinstance(value, (int, np.integer)) for value in actual[1:]), context
        if math.isnan(expected[0]):
            assert math.isnan(actual[0]), context
        else:
            assert math.isclose(actual[0], expected[0], rel_tol=2e-12, abs_tol=2e-14), context
        assert actual_warnings == expected_warnings, context


def run():
    rng = np.random.default_rng(829439)
    checks = 0
    float_types = [np.float16, np.float32, np.float64, np.longdouble]
    integer_types = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint64]
    tolerances = [
        1e-8, 0, -0.0, -1e-50, -1, np.inf, -np.inf, np.nan,
        1e-50, 1e-4, 0.5, 1, 1e4, 1e50,
        np.float16(1e-4), np.float32(1e-8), np.longdouble("1e-8"),
        np.int64(1), np.uint64(2**63), 2**1000,
    ]
    for dtype in float_types + integer_types:
        if np.issubdtype(dtype, np.floating):
            info = np.finfo(dtype)
            base = np.array([
                -info.max, info.max, 0, -0.0, info.tiny, -info.tiny,
                info.smallest_subnormal, -info.smallest_subnormal,
                1, -1, 1e-8, -1e-8,
            ], dtype=dtype)
            pool = np.concatenate((base, np.nextafter(base, dtype(0))))
        else:
            info = np.iinfo(dtype)
            pool = np.array([info.min, info.max, 0, 1, info.max // 2], dtype=dtype)
        for i in range(240):
            n = int(rng.integers(2, 75))
            events = rng.random(n) < 0.6
            times = rng.integers(0, max(1, n // 3), size=n)
            if i % 3 == 0:
                risks = rng.choice(pool, n)
            elif np.issubdtype(dtype, np.floating):
                risks = (rng.normal(size=n) * (1e-8 if i % 2 else 1)).astype(dtype)
            else:
                risks = rng.integers(0, min(info.max, 100), size=n).astype(dtype)
            tol = tolerances[i % len(tolerances)]
            compare("concordance_index_censored", events, times, risks, tied_tol=tol)
            test = Surv.from_arrays(events, times)
            train = Surv.from_arrays([True, False, True, False, True], [0, 2, 4, 8, n + 1])
            compare(
                "concordance_index_ipcw", train, test, risks,
                tau=None if i % 2 else float(rng.integers(-1, n + 2)), tied_tol=tol,
            )
            checks += 2

    # Exact tolerance boundaries, adjacent representable values, non-transitive
    # ties, and every event pattern within small tied-time groups.
    for dtype in float_types[:3]:
        for tol in [1e-8, 0.1, 0.5, 1.0, np.finfo(dtype).max]:
            with np.errstate(all="ignore"):
                risk = np.array([0, tol, -tol, 2 * tol, -2 * tol], dtype=dtype)
                risk = np.concatenate((risk, np.nextafter(risk, dtype(0)), np.nextafter(risk, dtype(np.inf))))
            risk = risk[np.isfinite(risk)]
            for code in range(32):
                events = np.tile([(code >> j) & 1 for j in range(5)], 3)[:len(risk)].astype(bool)
                times = np.tile([1, 1, 2, 3, 3], 3)[:len(risk)]
                compare("concordance_index_censored", events, times, risk, tied_tol=tol)
                checks += 1

    # Validation, degenerate comparisons, unsupported scalar tolerances, and
    # intentionally malformed shapes retain the release's exception classes.
    valid_train = Surv.from_arrays([True, False, True], [1, 2, 4])
    for events, times in [
        ([True], [1]), ([False, True], [1, 2]), ([True, True], [2, 2]),
        ([False, True], [2, 2]), ([False, False], [1, 2]),
        ([True, True, True], [2, 2, 2]), ([False, False, True], [1, 2, 3]),
        ([False, True, True], [1, 2, 2]),
    ]:
        for tol in tolerances + [None, "x", [1], [1, 2], np.array(0.1), complex(1, 0)]:
            risks = np.zeros(len(events))
            compare("concordance_index_censored", events, times, risks, tied_tol=tol)
            test = Surv.from_arrays(events, times)
            compare("concordance_index_ipcw", valid_train, test, risks, tied_tol=tol)
            checks += 2

    for invalid in [[], [np.nan, 1], [np.inf, 2], [[1], [2]], [[1, 2], [3, 4]], ["a", "b"]]:
        compare("concordance_index_censored", [True, False], [1, 2], invalid)
        compare("concordance_index_censored", [True, False], invalid, [1, 2])
        compare("concordance_index_censored", invalid, [1, 2], [1, 2])
        checks += 3

    test = Surv.from_arrays([True, False, True], [1, 2, 3])
    for tau in [-1, 0, 1, 2, 3, np.nan, np.inf, -np.inf, "x", [], [2], [1, 2], np.array(2)]:
        compare("concordance_index_ipcw", valid_train, test, [3, 2, 1], tau=tau)
        checks += 1
    for train in [
        Surv.from_arrays([True, False], [1, 2]),
        Surv.from_arrays([False, False], [1, 4]),
        Surv.from_arrays([True, False, True, False], [1, 2, 2, 4]),
        Surv.from_arrays([True, True], [0, 1]),
    ]:
        for tau in [None, 2, 4]:
            compare("concordance_index_ipcw", train, test, [3, 2, 1], tau=tau)
            checks += 1

    # Large counts are Python integers or int64, independent of risk dtype.
    n = 100_000
    events = np.ones(n, dtype=bool)
    times = np.arange(n)
    for risks in [np.zeros(n), np.zeros(n, dtype=np.int32)]:
        result = metrics.concordance_index_censored(events, times, risks)
        assert result == (0.5, 0, 0, n * (n - 1) // 2, 0), result
        checks += 1

    print(f"Passed {checks} differential and large-count checks.")


if __name__ == "__main__":
    run()

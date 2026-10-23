"""Differential checks against the installed, unmodified 0.28.0 source.

Run from the workspace with: python tests/check_concordance.py
"""

import importlib.util
import math
from pathlib import Path
import warnings

import numpy as np
import sksurv
from sksurv import metrics
from sksurv.util import Surv


reference_path = next(
    Path(p) / "metrics.py" for p in sksurv.__path__
    if (Path(p) / "metrics.py").is_file()
    and (Path(p) / "metrics.py").resolve() != Path(metrics.__file__).resolve()
)
spec = importlib.util.spec_from_file_location("sksurv._release_metrics", reference_path)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def outcome(function, args, kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = function(*args, **kwargs)
        except Exception as error:
            result = type(error)
    return result, {w.category.__name__ for w in caught}


checked = 0


def compare(name, *args, **kwargs):
    global checked
    expected, expected_warnings = outcome(getattr(reference, name), args, kwargs)
    actual, actual_warnings = outcome(getattr(metrics, name), args, kwargs)
    context = (name, args, kwargs, expected, actual, expected_warnings, actual_warnings)
    if isinstance(expected, type):
        assert isinstance(actual, type) and issubclass(actual, expected), context
    else:
        assert not isinstance(actual, type), context
        assert expected[1:] == actual[1:], context
        assert all(isinstance(v, (int, np.integer)) for v in actual[1:]), context
        assert (math.isnan(expected[0]) and math.isnan(actual[0])) or math.isclose(
            expected[0], actual[0], rel_tol=2e-12, abs_tol=2e-14
        ), context
    assert expected_warnings == actual_warnings, context
    checked += 1


def ordinary(event, time, risk, **kwargs):
    compare("concordance_index_censored", event, time, risk, **kwargs)


def test_risks():
    rng = np.random.default_rng(432123)
    tolerances = [0, 0.0, 1e-8, 0.125, 0.5, -1, -0.0, np.inf, -np.inf,
                  np.nan, np.float16(0.1), np.float32(0.1), np.float64(0.1),
                  np.longdouble(0.1), np.int64(1), np.uint64(2**64 - 1),
                  True, False, 2**100, 2**1100, None, "bad", np.array(0.1)]
    dtypes = [np.float16, np.float32, np.float64, np.int8, np.uint8,
              np.int16, np.int32, np.int64, np.uint64, bool]
    for repeat in range(180):
        n = int(rng.integers(1, 100))
        events = rng.random(n) < rng.uniform()
        times = rng.integers(0, max(2, n // 3), size=n).astype(float)
        risks = rng.integers(-10, 10, size=n).astype(rng.choice(dtypes))
        for tol in tolerances:
            ordinary(events, times, risks, tied_tol=tol)
    for repeat in range(600):
        n = int(rng.integers(2, 100))
        events = rng.random(n) < 0.7
        times = rng.integers(0, 12, n)
        exponents = rng.integers(-1074, 1024, n)
        risks = np.ldexp(rng.uniform(-1, 1, n), exponents)
        tol = rng.choice([0., 1e-8, 1., 1e200, np.finfo(float).max, np.inf, np.nan, -1.])
        ordinary(events, times, risks, tied_tol=tol)
    for dtype in (np.float16, np.float32, np.float64):
        for tol in (1e-8, 0.1, 0.125, 1.0, 100.):
            center = dtype(tol)
            vals = np.array([-center, -0., 0., center,
                             np.nextafter(center, dtype(-np.inf)),
                             np.nextafter(center, dtype(np.inf))], dtype=dtype)
            for event in (np.ones(len(vals), bool), np.arange(len(vals)) % 2 == 0):
                ordinary(event, np.arange(len(vals)), vals, tied_tol=tol)
                ordinary(event, np.ones(len(vals)), vals, tied_tol=tol)
    huge = np.finfo(float).max
    for risk in ([huge, -huge], [-huge, huge], [huge, huge], [-0., 0.]):
        for event in ([True, True], [True, False], [False, True]):
            for time in ([1., 2.], [1., 1.]):
                for tol in (0., huge, np.inf, np.nan, -1.):
                    ordinary(event, time, risk, tied_tol=tol)


def test_ipcw():
    rng = np.random.default_rng(53523)
    for repeat in range(500):
        n = int(rng.integers(1, 100))
        nt = int(rng.integers(1, 120))
        test = Surv.from_arrays(rng.random(n) < rng.uniform(), rng.integers(0, 30, n))
        train = Surv.from_arrays(rng.random(nt) < rng.uniform(), rng.integers(0, 30, nt))
        risks = rng.integers(-20, 20, n).astype(rng.choice([np.float16, np.float32, np.float64, np.int64]))
        for tau in (None, 0, 10, 29, 31, np.inf, np.nan, -1):
            compare("concordance_index_ipcw", train, test, risks, tau=tau)
    train = Surv.from_arrays([True, False, True, False, True], [1, 1, 2, 3, 4])
    test = Surv.from_arrays([True, False, True, True, False], [1, 1, 2, 3, 4])
    for tau in (None, 0, 1, 2, 3, 4, 5, np.inf, np.nan, [1, 2], np.array([1]), "bad"):
        for tol in (0, 0.1, -1., np.inf, np.nan):
            compare("concordance_index_ipcw", train, test, [2., 2., 1., 3., 4.], tau=tau, tied_tol=tol)


def test_invalid():
    for event in ([], [True], [False, False], [1, 0], [[True], [False]], [[True, False]], [True, False]):
        for time in ([], [1], [1, 2], [1, np.nan], [1, np.inf], [[1], [2]], [-1, 0]):
            for risk in ([], [1.], [1., 2.], [[1.], [2.]], [np.inf, 0], [np.nan, 0], [1j, 2j]):
                ordinary(event, time, risk)
    for events in ([True, True], [False, True], [True, False]):
        for time in ([1, 1], [1, 2]):
            ordinary(events, time, [1., 2.])


def test_large():
    n = 200_000
    times = np.arange(n, dtype=float)
    events = np.ones(n, dtype=bool)
    npairs = n * (n - 1) // 2
    for risk, expected in ((-times, (1., npairs, 0, 0, 0)),
                           (times, (0., 0, npairs, 0, 0)),
                           (np.zeros(n), (0.5, 0, 0, npairs, 0))):
        actual = metrics.concordance_index_censored(events, times, risk)
        assert actual == expected, (actual, expected)
        y = Surv.from_arrays(events, times)
        assert metrics.concordance_index_ipcw(y, y, risk, tau=n // 2) == expected
    events[::2] = False
    # One time group: precisely every event/censor pair is comparable.
    actual = metrics.concordance_index_censored(events, np.ones(n), np.zeros(n))
    expected = (0.5, 0, 0, (n // 2)**2, (n // 2)**2)
    assert actual == expected, (actual, expected)


if __name__ == "__main__":
    for test in (test_risks, test_ipcw, test_invalid, test_large):
        test()
        print(test.__name__, "passed", flush=True)
    print(f"{checked} differential cases passed")

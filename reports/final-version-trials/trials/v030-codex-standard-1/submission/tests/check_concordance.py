"""Differential checks against the installed, unmodified 0.28.0 release.

Run from the submission directory: python tests/check_concordance.py
"""
import importlib.metadata
import importlib.util
import math
from pathlib import Path
import sys
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sksurv import metrics


release_path = importlib.metadata.distribution("scikit-survival").locate_file("sksurv/metrics.py")
spec = importlib.util.spec_from_file_location("sksurv._release_metrics", release_path)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def outcome(function, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = function(*args, **kwargs)
        except Exception as error:
            return type(error), None, None
    return None, result, {type(w.message).__name__ for w in caught}


checks = 0


def check(name, *args, **kwargs):
    global checks
    expected = outcome(getattr(release, name), *args, **kwargs)
    actual = outcome(getattr(metrics, name), *args, **kwargs)
    checks += 1
    assert expected[0] is actual[0], (checks, name, kwargs, expected, actual)
    if expected[0] is not None:
        return
    _, a, wa = expected
    _, b, wb = actual
    assert wa == wb, (checks, name, kwargs, expected, actual)
    assert a[1:] == b[1:], (checks, name, kwargs, expected, actual)
    assert all(isinstance(v, (int, np.integer)) for v in b[1:]), b
    assert (math.isnan(a[0]) and math.isnan(b[0])) or math.isclose(
        a[0], b[0], rel_tol=2e-12, abs_tol=2e-14
    ), (checks, name, kwargs, expected, actual)


def survival(events, times):
    y = np.empty(len(events), dtype=[("event", "?"), ("time", "f8")])
    y["event"] = events
    y["time"] = times
    return y


def main():
    rng = np.random.default_rng(93626)
    dtypes = [np.float16, np.float32, np.float64, np.longdouble, np.int8, np.int64, np.uint64]
    tolerances = [1e-8, 0.0, -0.0, -1e-50, -1.0, np.nan, np.inf, -np.inf, 1e100,
                  np.float64(1e100), np.float16(1e-3), np.longdouble(1e-8)]
    train = survival([True, False, True, False, True], [0, 0, 1, 8, 10])
    for dtype in dtypes:
        for n in [2, 9, 255, 256, 257, 403]:
            events = rng.random(n) < .65
            times = rng.integers(0, 8, n)
            if np.dtype(dtype).kind == "f":
                risk = rng.choice(np.array([-1, -1e-8, -0., 0., 1e-8, 2e-8, 1], dtype=dtype), n)
            else:
                risk = rng.integers(0, 16, n).astype(dtype)
            test = survival(events, times)
            for tolerance in tolerances:
                check("concordance_index_censored", events, times, risk, tied_tol=tolerance)
                check("concordance_index_ipcw", train, test, risk, tied_tol=tolerance)
                check("concordance_index_ipcw", train, test, risk, tau=4, tied_tol=tolerance)

    for dtype in [np.float16, np.float32, np.float64, np.longdouble]:
        finfo = np.finfo(dtype)
        with np.errstate(all="ignore"):
            tiny = np.nextafter(dtype(0), dtype(1))
            values = np.array([-finfo.max, finfo.max, -tiny, tiny, -0., 0., 1., -1.,
                               np.nextafter(dtype(1), dtype(0)),
                               np.nextafter(dtype(1), dtype(2))], dtype=dtype)
        for tolerance in tolerances + [finfo.max, tiny, finfo.eps]:
            n = 300
            events = rng.random(n) < .5
            times = rng.integers(0, 6, n)
            risk = rng.choice(values, n)
            check("concordance_index_censored", events, times, risk, tied_tol=tolerance)
            check("concordance_index_ipcw", train, survival(events, times), risk, tied_tol=tolerance)

    for _ in range(150):
        n = int(rng.integers(256, 900))
        dtype = rng.choice([np.float16, np.float32, np.float64])
        risk = rng.normal(size=n).astype(dtype)
        events = rng.random(n) < rng.uniform(.01, 1)
        events[0] = True
        times = rng.choice([np.arange(n), rng.integers(0, 5, n)])
        tolerance = rng.choice([0., 1e-8, .01, 1.])
        check("concordance_index_censored", events, times, risk, tied_tol=tolerance)
        tr = survival(rng.random(n) < .7, rng.uniform(0, 2 * n, n))
        check("concordance_index_ipcw", tr, survival(events, times), risk, tau=n * .6,
              tied_tol=tolerance)

    # The release distinguishes unvisited events from visited empty masks.
    for n in [1, 2, 255, 256, 300]:
        for e in [np.ones(n, bool), np.zeros(n, bool), np.arange(n) == n - 1,
                  np.arange(n) == 0]:
            for t in [np.zeros(n), np.arange(n), np.r_[np.zeros(max(0, n - 2)), [1, 1]][-n:]]:
                for tolerance in [1e-8, 1e100, -1., np.nan]:
                    check("concordance_index_censored", e, t, np.ones(n), tied_tol=tolerance)
                    check("concordance_index_ipcw", train, survival(e, t), np.ones(n),
                          tied_tol=tolerance, tau=1)

    # Invalid input validation, censoring support and truncation.
    for n in [0, 1, 2, 256]:
        e = np.ones(n, bool)
        t = np.arange(n, dtype=float)
        r = np.zeros(n)
        for bad in [np.zeros(n, int), np.ones((n, 2), bool), [], None]:
            check("concordance_index_censored", bad, t, r)
        for bad in [np.full(n, np.nan), np.full(n, np.inf), np.zeros((n, 2)), [], None]:
            check("concordance_index_censored", e, t, bad)
        for bad in [np.full(n, np.nan), np.full(n, np.inf), np.zeros((n, 2)), [], None]:
            check("concordance_index_censored", e, bad, r)
        for tr in [train, survival([True, False], [0, 1]), survival([True, True], [0, 1]),
                   survival([False, False], [0, 1])]:
            for tau in [None, -1, 0, .1, 1, 10, np.nan, np.inf, [], [1], [1, 2], "bad"]:
                check("concordance_index_ipcw", tr, survival(e, t), r, tau=tau)

    print(f"{checks} differential checks passed")


if __name__ == "__main__":
    main()

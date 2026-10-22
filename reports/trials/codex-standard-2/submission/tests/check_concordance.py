"""Differential checks against the installed, unmodified 0.28.0 module.

Run with ``python tests/check_concordance.py`` from the submission directory.
"""

import importlib.util
import math
from pathlib import Path
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sksurv import metrics
import sksurv


reference_path = next(
    Path(root) / "metrics.py"
    for root in sksurv.__path__
    if (Path(root) / "metrics.py").resolve() != Path(metrics.__file__).resolve()
)
spec = importlib.util.spec_from_file_location("sksurv._reference_metrics", reference_path)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def outcome(func, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = func(*args, **kwargs)
            error = None
        except Exception as exc:
            result = None
            error = type(exc)
    return result, error, {w.category.__name__ for w in caught}


def compare(name, *args, **kwargs):
    a = outcome(getattr(reference, name), *args, **kwargs)
    b = outcome(getattr(metrics, name), *args, **kwargs)
    assert a[1:] == b[1:], (name, kwargs, a, b)
    if a[1] is None:
        x, y = a[0], b[0]
        assert x[1:] == y[1:], (name, kwargs, x, y)
        assert math.isclose(x[0], y[0], rel_tol=2e-12, abs_tol=2e-14) or (
            np.isnan(x[0]) and np.isnan(y[0])
        ), (name, kwargs, x, y)
        assert all(isinstance(value, (int, np.integer)) for value in y[1:])


def survival(event, times):
    result = np.empty(len(event), dtype=[("event", "?"), ("time", "f8")])
    result["event"] = event
    result["time"] = times
    return result


def random_checks(rng):
    tolerances = [
        1e-8, 0, -1.0, np.inf, -np.inf, np.nan, 0.2, 1e100, -1e-100,
        np.float64(0.2), np.float32(0.2), np.float16(0.2), np.array(0.2),
    ]
    for dtype in [np.float16, np.float32, np.float64, np.int8, np.int64, np.uint64]:
        for n in [2, 63, 64, 65, 100, 300]:
            for i in range(39):
                event = rng.random(n) < 0.6
                times = rng.integers(0, n // 3 + 1, n).astype(float)
                risk = rng.normal(size=n).astype(dtype)
                if i % 3 == 0:
                    risk = rng.integers(0, 10, n).astype(dtype)
                    if dtype in (np.float16, np.float32, np.float64):
                        risk /= dtype(3)
                compare("concordance_index_censored", event, times, risk, tied_tol=tolerances[i % len(tolerances)])

    for n in [1, 2, 63, 64, 65, 200, 1000]:
        for i in range(35):
            train = survival(rng.random(300) < 0.7, rng.integers(0, 100, 300))
            test = survival(rng.random(n) < 0.7, rng.integers(0, 110, n))
            risk = rng.integers(-10, 10, n).astype([np.float16, np.float32, np.float64][i % 3])
            compare(
                "concordance_index_ipcw", train, test, risk,
                tau=[None, 30, 99, 100, 0, -1, np.inf][i % 7],
                tied_tol=tolerances[i % len(tolerances)],
            )


def boundary_checks(rng):
    for dtype in [np.float16, np.float32, np.float64]:
        finfo = np.finfo(dtype)
        values = np.array([
            -finfo.max, -finfo.max / 2, -1, -finfo.tiny,
            -finfo.smallest_subnormal, -0.0, 0.0, finfo.smallest_subnormal,
            finfo.tiny, 1, finfo.max / 2, finfo.max,
        ], dtype=dtype)
        for i in range(48):
            n = 128
            event = rng.random(n) < 0.7
            times = rng.integers(0, 10, n).astype(float)
            risk = rng.choice(values, n)
            tol = [0., 1e-8, np.inf, -np.inf, np.nan, -1e-100, finfo.max, -1.][i % 8]
            compare("concordance_index_censored", event, times, risk, tied_tol=tol)

        # Adjacent representable differences, exact and nextafter tolerances,
        # and cancellation in the commonly used risk +/- tolerance shortcut.
        for anchor in [-1e3, -1., -finfo.tiny, 0., finfo.tiny, 1., 1e3]:
            anchor = dtype(anchor)
            neighborhood = [anchor]
            for direction in [dtype(-np.inf), dtype(np.inf)]:
                value = anchor
                for _ in range(10):
                    value = np.nextafter(value, direction)
                    neighborhood.append(value)
            risk = np.resize(np.array(neighborhood, dtype=dtype), 128)
            for tol in [dtype(0), dtype(finfo.eps), dtype(1), np.float64(finfo.eps), float(finfo.eps)]:
                event = rng.random(128) < 0.8
                times = rng.integers(0, 10, 128).astype(float)
                compare("concordance_index_censored", event, times, risk, tied_tol=tol)

        # Extrema whose overflowing difference is never actually comparable.
        for event in [np.arange(128) == 0, np.arange(128) == 127]:
            risk = np.zeros(128, dtype=dtype)
            risk[:2] = [-finfo.max, finfo.max]
            times = np.ones(128)
            times[:2] = 0
            compare("concordance_index_censored", event, times, risk)


def degeneracy_checks(rng):
    for n in [1, 2, 63, 64, 128]:
        for event in [np.ones(n, dtype=bool), np.arange(n) == 0, np.arange(n) == n - 1, np.zeros(n, dtype=bool)]:
            for times in [np.zeros(n), np.arange(n), -np.arange(n), np.arange(n) // 3]:
                risk = rng.normal(size=n)
                compare("concordance_index_censored", event, times, risk)
                train = survival([True, True, False, True], [0, 1, n + 2, n + 3])
                test = survival(event, times)
                for tau in [None, n // 2, n + 1, -1, np.nan]:
                    compare("concordance_index_ipcw", train, test, risk, tau=tau)


def invalid_checks():
    event = np.ones(64, dtype=bool)
    times = np.arange(64, dtype=float)
    risk = np.arange(64, dtype=float)
    for bad_event in [[], [True], event[:-1], event.astype(int), event[:, None], event.reshape(2, 32)]:
        compare("concordance_index_censored", bad_event, times, risk)
    for bad_times in [[], times[:-1], times[:, None], times.reshape(2, 32), times * np.nan, times + np.inf]:
        compare("concordance_index_censored", event, bad_times, risk)
    for bad_risk in [[], risk[:-1], risk[:, None], risk.reshape(2, 32), risk * np.nan, risk + np.inf, risk.astype(complex)]:
        compare("concordance_index_censored", event, times, bad_risk)
    for bad_tol in [None, "bad", [], [1.], [0., 1.], np.array([1., 2.]), complex(1), 10**1000]:
        compare("concordance_index_censored", event, times, risk, tied_tol=bad_tol)

    valid = survival(event, times)
    for invalid in [
        [], np.array([]), valid[:0], valid["time"], valid.astype([("x", "i8"), ("y", "f8")]),
        survival(np.zeros(64, dtype=bool), times), survival(event, -times), survival(event, times * np.nan),
    ]:
        compare("concordance_index_ipcw", invalid, valid, risk)
        compare("concordance_index_ipcw", valid, invalid, risk)
    for tau in ["bad", [], [1., 2.], np.array([1., 2.]), complex(1), np.nan, -np.inf]:
        compare("concordance_index_ipcw", valid, valid, risk, tau=tau)


def large_count_checks():
    n = 200_000
    event = np.ones(n, dtype=bool)
    times = np.arange(n, dtype=float)
    expected = n * (n - 1) // 2
    train = survival([True, True], [0., float(n)])
    test = survival(event, times)
    assert metrics.concordance_index_censored(event, times, -times) == (1., expected, 0, 0, 0)
    assert metrics.concordance_index_ipcw(train, test, -times) == (1., expected, 0, 0, 0)
    assert metrics.concordance_index_censored(event, times, times) == (0., 0, expected, 0, 0)
    event[n // 2:] = False
    times[:] = 1.
    risk = event.astype(float)
    expected = (n // 2) * (n - n // 2)
    assert metrics.concordance_index_censored(event, times, risk) == (1., expected, 0, 0, expected)
    print("Exact large-count checks passed.", flush=True)


def scale_checks(rng):
    n = 200_000
    for distribution in ["normal", "ties", "extreme", "all_equal"]:
        event = rng.random(n) < 0.7
        times = rng.uniform(0, 99, n) if distribution == "normal" else rng.integers(0, 100, n).astype(float)
        if distribution == "normal":
            risk = rng.normal(size=n)
        elif distribution == "ties":
            risk = rng.integers(0, 100, n).astype(float) * 1e-8
        elif distribution == "extreme":
            risk = rng.choice([-1e308, -1., 0., 1., 1e308], n)
        else:
            risk = np.zeros(n)
        train = survival([True, False, True, False, True], [0., 30., 50., 80., 100.])
        test = survival(event, times)
        for name, args, kwargs in [
            ("concordance_index_censored", (event, times, risk), {}),
            ("concordance_index_ipcw", (train, test, risk), {"tau": 90.}),
        ]:
            start = time.monotonic()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                answer = getattr(metrics, name)(*args, **kwargs)
            print(distribution, name, round(time.monotonic() - start, 3), answer, flush=True)
            # Check a prefix independently against the pairwise implementation.
            subset_args = tuple(a[:1000] if isinstance(a, np.ndarray) and len(a) == n else a for a in args)
            compare(name, *subset_args, **kwargs)


if __name__ == "__main__":
    rng = np.random.default_rng(10)
    random_checks(rng)
    boundary_checks(rng)
    degeneracy_checks(rng)
    invalid_checks()
    print("Differential compatibility checks passed.", flush=True)
    large_count_checks()
    scale_checks(rng)

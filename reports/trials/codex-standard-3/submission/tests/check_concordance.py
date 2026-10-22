"""Differential checks against the unmodified 0.28.0 pair scan."""
import importlib.util
import math
from pathlib import Path
import sys
import warnings
import numpy as np
from sksurv import metrics as fast
import sksurv

_spec = importlib.util.spec_from_file_location(
    "sksurv._metrics_reference", Path(sksurv.__file__).with_name("metrics.py")
)
ref = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ref
_spec.loader.exec_module(ref)


def outcome(fn, args, kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            result = fn(*args, **kw)
        except Exception as exc:
            return type(exc), None, {type(w.message).__name__ for w in caught}
    return None, result, {type(w.message).__name__ for w in caught}


def compare(name, args, **kw):
    a = outcome(getattr(ref, name), args, kw)
    b = outcome(getattr(fast, name), args, kw)
    assert a[0] == b[0], (args, kw, a, b)
    if a[0] is None:
        assert a[2] == b[2], (args, kw, a, b)
        assert a[1][1:] == b[1][1:], (args, kw, a, b)
        x, y = a[1][0], b[1][0]
        assert (math.isnan(x) and math.isnan(y)) or math.isclose(x, y, rel_tol=2e-12, abs_tol=2e-14), (args, kw, a, b)
    return b


def structured(e, t):
    y = np.empty(len(e), dtype=[('event', '?'), ('time', 'f8')])
    y['event'], y['time'] = e, t
    return y


def main():
    rng = np.random.default_rng(1243)
    tol_values = [0., 1e-8, .25, 1., -1., np.nan, np.inf, -np.inf]
    for i in range(4000):
        n = int(rng.integers(2, 90))
        e = rng.random(n) < rng.random()
        t = rng.integers(0, rng.integers(1, n+1), n).astype(float)
        r = rng.normal(size=n) if i % 2 else rng.integers(-8, 8, n).astype(float)/4
        compare('concordance_index_censored', (e, t, r), tied_tol=tol_values[i%len(tol_values)])
    for dtype in [np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint64, bool]:
        for i in range(50):
            n=30
            e=rng.random(n)<.6
            t=rng.integers(0, 8, n)
            r=rng.integers(-8, 8, n).astype(dtype)
            compare('concordance_index_censored', (e,t,r), tied_tol=tol_values[i%len(tol_values)])
    # Subtraction rounding at the inclusive tie boundary, and signed zero.
    for dtype in [np.float16, np.float32, np.float64]:
        for tol in [0., 1e-8, 1e-4, 0.5, 1., np.finfo(dtype).max, 1e40]:
            for base in [0., -0., 1., -1., 100., -100.]:
                with np.errstate(over='ignore', invalid='ignore'):
                    r = np.array([base, base+tol, base-tol], dtype=dtype)
                    r = np.concatenate((r, np.nextafter(r, dtype(np.inf)), np.nextafter(r, dtype(-np.inf))))
                e = rng.random(r.size)<.8
                t = rng.integers(0, 5, r.size)
                compare('concordance_index_censored', (e,t,r), tied_tol=tol)
    for i in range(1000):
        n=int(rng.integers(1,80))
        tr=structured(rng.random(50)<.7,rng.integers(0,30,50))
        te=structured(rng.random(n)<.7,rng.integers(0,35,n))
        r=rng.normal(size=n) if i%2 else rng.integers(-3,4,n).astype(float)
        tau=[None, 5., 20., 30., 100., -1., 0., np.nan, np.inf][i%9]
        compare('concordance_index_ipcw', (tr,te,r), tau=tau, tied_tol=tol_values[i%len(tol_values)])
    print('Differential checks passed')

if __name__ == '__main__':
    main()

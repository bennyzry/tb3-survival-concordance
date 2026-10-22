"""Compatibility and scaling checks against the supplied 0.28.0 implementation."""
import importlib.util
import math
from pathlib import Path
import unittest
import warnings

import numpy as np
import sksurv
from sksurv import metrics

_spec = importlib.util.spec_from_file_location(
    'sksurv._reference_metrics', Path(sksurv.__file__).with_name('metrics.py')
)
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)


def outcome(function, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            value = function(*args, **kwargs)
        except Exception as exc:
            return ('error', type(exc))
    return ('ok', value, {w.category.__name__ for w in caught})


def survival(event, time):
    y = np.empty(len(event), dtype=[('event', bool), ('time', float)])
    y['event'] = event
    y['time'] = time
    return y


class ConcordanceTest(unittest.TestCase):
    def compare(self, api, *args, **kwargs):
        expected = outcome(getattr(reference, api), *args, **kwargs)
        actual = outcome(getattr(metrics, api), *args, **kwargs)
        self.assertEqual(actual[0], expected[0], (args, kwargs, actual, expected))
        if expected[0] == 'error':
            self.assertTrue(issubclass(actual[1], expected[1]), (args, kwargs, actual, expected))
        else:
            self.assertEqual(actual[1][1:], expected[1][1:], (args, kwargs, actual, expected))
            self.assertTrue(all(isinstance(x, (int, np.integer)) for x in actual[1][1:]))
            a, b = actual[1][0], expected[1][0]
            self.assertTrue((math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=2e-12, abs_tol=2e-14), (args, kwargs, actual, expected))
            self.assertEqual(actual[2], expected[2], (args, kwargs, actual, expected))

    def test_random(self):
        rng = np.random.default_rng(725)
        for iteration in range(1500):
            n = int(rng.integers(1, 90))
            event = rng.random(n) < rng.uniform(0.01, 1)
            time = rng.integers(0, max(2, n//2), n).astype(float)
            risk = rng.normal(size=n)
            if iteration % 2:
                risk = np.round(risk, int(rng.integers(0, 4)))
            tol = rng.choice([1e-8, 0., -1., np.inf, -np.inf, np.nan, .1, 1., 10.])
            self.compare('concordance_index_censored', event, time, risk, tied_tol=tol)
            train = survival([True,False,True,False,True], [0,1,3,8,100])
            self.compare('concordance_index_ipcw', train, survival(event, time), risk,
                         tied_tol=tol, tau=rng.choice([None, -1., 0., 2., 15., 100., np.nan, np.inf]))

    def test_dtype_arithmetic(self):
        rng = np.random.default_rng(923)
        for dtype in [np.float16, np.float32, np.float64, np.longdouble,
                      np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint64, bool]:
            for iteration in range(100):
                n = 30
                event = rng.random(n) < .65
                time = rng.integers(0, 5, n)
                with np.errstate(all='ignore'):
                    if np.dtype(dtype).kind == 'f':
                        info = np.finfo(dtype)
                        choices = np.array([-info.max, info.max, -info.tiny, info.tiny, 0., -0., 1., -1., 1e-8], dtype=dtype)
                    else:
                        info = np.iinfo(dtype) if dtype is not bool else None
                        choices = np.array([0, 1, info.max, info.min], dtype=dtype) if info else np.array([True,False])
                risk = rng.choice(choices, n)
                tol = rng.choice([1e-8, 0., -1., np.inf, -np.inf, np.nan, 1., 1e308])
                self.compare('concordance_index_censored', event, time, risk, tied_tol=tol)
                self.compare('concordance_index_ipcw', survival([True,False,True], [0,2,10]),
                             survival(event, time), risk, tied_tol=tol)

    def test_boundaries(self):
        rng = np.random.default_rng(516)
        for iteration in range(500):
            tol = 10. ** rng.uniform(-320, 300)
            center = rng.choice([0., rng.uniform(-1, 1) * tol, rng.normal(), tol])
            values = np.array([center, center - tol, center + tol])
            values = np.r_[values, np.nextafter(values, np.inf), np.nextafter(values, -np.inf)]
            for dtype in [np.float16, np.float32, np.float64]:
                with np.errstate(all='ignore'):
                    risk = values.astype(dtype)
                self.compare('concordance_index_censored', np.ones(9, bool), np.arange(9), risk, tied_tol=tol)

    def test_extreme_float64_pairs(self):
        rng = np.random.default_rng(92742)
        for iteration in range(600):
            bits = rng.bit_generator.random_raw(100)
            risks = bits.view(np.float64)
            risks = risks[np.isfinite(risks)]
            risks = np.r_[risks, -0., 0., np.finfo(float).max, -np.finfo(float).max]
            n = len(risks)
            with np.errstate(all='ignore'):
                tol = np.abs(risks[rng.integers(n)] - risks[rng.integers(n)])
                tol = np.nextafter(tol, rng.choice([0., np.inf]))
            if iteration % 3 == 0:
                tol = float(tol)
            elif iteration % 3 == 1:
                tol = np.longdouble(tol)
            self.compare('concordance_index_censored', rng.random(n) < .7,
                         rng.integers(0, 12, n), risks, tied_tol=tol)

    def test_ties_and_truncation(self):
        # Risk ties are pairwise and non-transitive, and same-time events must
        # never enter each other's comparison sets.
        event = np.array([True,False,True,True,False,True,False,True])
        time = np.array([1,1,1,2,2,3,3,4])
        risks = np.array([0., .75, 1.5, 0., .75, 1.5, 0., .75])
        train = survival([True,False,False,True,False,True], [0,1,1,2,3,5])
        self.compare('concordance_index_censored', event, time, risks, tied_tol=1.)
        for tau in [None, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5]:
            self.compare('concordance_index_ipcw', train, survival(event,time), risks,
                         tau=tau, tied_tol=1.)

    def test_invalid_and_degenerate(self):
        for event in [[True], [False], [True,True], [True,False], [False,True], [False,False]]:
            n = len(event)
            for time in [np.ones(n), np.arange(n), -np.arange(n), np.zeros((n,1))]:
                for tol in [1e-8, None, 'x', [0], [0,1], np.array(0), 1j, np.float32(0), 2**2000]:
                    self.compare('concordance_index_censored', np.array(event), time, np.zeros(n), tied_tol=tol)
        for tau in [None, 0, -1, np.nan, np.inf, -np.inf, 'x', [], [1], [1,2], [1,2,3], np.array(2)]:
            for train in [survival([True,True], [1,2]), survival([True,False], [1,2]),
                          survival([False,False], [1,2]), survival([True,False,True], [1,1,3])]:
                for test in [survival([True,False], [1,1]), survival([True,True], [2,2]),
                             survival([False,True], [2,3]), survival([True], [1])]:
                    self.compare('concordance_index_ipcw', train, test, np.zeros(len(test)), tau=tau)
        for event, time, estimate in [([],[],[]), ([1,0],[1,2],[2.,1.]),
                                      ([True,False],[1,np.nan],[2.,1.]),
                                      ([True,False],[1,2],[np.inf,1.]),
                                      ([True,False],[1,2],[[1],[2]]),
                                      ([[True],[False]],[1,2],[1.,2.])]:
            self.compare('concordance_index_censored', event, time, estimate)

    def test_public_keywords(self):
        self.compare('concordance_index_censored', event_indicator=[True,False], event_time=[1,2], estimate=[1.,0.])
        self.compare('concordance_index_ipcw', survival_train=survival([True,False,True], [0,2,5]),
                     survival_test=survival([True,False], [1,3]), estimate=[1.,0.])

    def test_wide_counts(self):
        n = 100000
        result = metrics.concordance_index_censored(np.ones(n, bool), np.arange(n), -np.arange(n, dtype=float))
        self.assertEqual(result, (1., n*(n-1)//2, 0, 0, 0))
        y = survival(np.r_[np.ones(n//2, bool), np.zeros(n//2, bool)], np.ones(n))
        result = metrics.concordance_index_ipcw(survival([True,False,True], [0,2,3]), y, np.zeros(n))
        self.assertEqual(result, (.5, 0, 0, n*n//4, n*n//4))


if __name__ == '__main__':
    unittest.main()

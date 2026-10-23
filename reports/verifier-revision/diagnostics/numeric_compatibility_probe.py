"""Offline dtype/scalar-promotion diagnostic. No model inference or grading."""
from pathlib import Path
import importlib.util
import json
import math
import sys
import warnings
import numpy as np
import sksurv.metrics as baseline

sys.path.insert(0, '/snapshot/tasks/survival-concordance/tests')
from cases import ordinary, ipcw, independent_pairs
from worker import capture


def promotion_cases():
    cases = []
    for dtype in ('float16', 'float32', 'float64'):
        smallest = float(np.nextafter(np.dtype(dtype).type(0), np.dtype(dtype).type(1)))
        # These ordinary public scalars straddle rounding to signed zero.
        for factor in (-1., -.75, -.5, -.25, .25, .5, .75, 1.):
            tol = smallest * factor
            for structure in ('equal', 'subnormal'):
                event = [True, False, True, False, True, False]
                times = [1., 1., 2., 3., 3., 4.]
                risk = [0.] * 6 if structure == 'equal' else [0., smallest, -smallest, 0., 2*smallest, -2*smallest]
                name = f'promotion-{dtype}-{factor}-{structure}'
                cases.append(ordinary(name+'-ordinary', event, times, risk, dtype=dtype, tol=tol))
                cases.append(ipcw(name+'-ipcw', [False, True, False, True, True, True],
                                  [.5, 1., 2., 3., 5., 6.], event, times, risk, dtype=dtype, tol=tol, tau=3.5))
    return cases


def compatible(actual, expected):
    if actual['status'] != expected['status']:
        return False
    if expected['status'] == 'error':
        return set(expected['error_types']) <= set(actual['error_types'])
    a, b = actual['result'], expected['result']
    return (a[1:] == b[1:] and actual['warnings'] == expected['warnings']
            and ((math.isnan(a[0]) and math.isnan(b[0]))
                 or math.isclose(a[0], b[0], rel_tol=2e-12, abs_tol=2e-14)))


def load(name, path):
    spec = importlib.util.spec_from_file_location('sksurv._probe_'+name.replace('-', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    cases = promotion_cases()
    expected = {c['id']: capture(baseline, c) for c in cases}
    independent_mismatches = []
    for case in cases:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            independent = independent_pairs(case)
        if not compatible(dict(expected[case['id']], result=independent), expected[case['id']]):
            independent_mismatches.append({'case':case['id'], 'independent':independent, 'pristine':expected[case['id']]})
    results = {}
    names = ['codex-standard-1-replacement', 'codex-standard-2', 'codex-standard-3',
             'deepseek-standard-1', 'deepseek-standard-2', 'deepseek-standard-3']
    for name in names:
        path = Path('/snapshot/reports/trials')/name/'submission/sksurv/metrics.py'
        module = load(name, path)
        failures = []
        for case in cases:
            actual = capture(module, case)
            if not compatible(actual, expected[case['id']]):
                failures.append({'case':case['id'], 'actual':actual, 'pristine':expected[case['id']]})
        results[name] = {'checked':len(cases), 'matched':len(cases)-len(failures), 'mismatches':failures}
    report = {'diagnostic_only':True, 'new_model_inference':False,
              'purpose':'NumPy dtype scalar-promotion at subnormal tolerance boundaries, across both published APIs.',
              'numpy':np.__version__, 'cases':cases, 'case_count':len(cases),
              'independent_reference_mismatches':independent_mismatches, 'implementations':results}
    print(json.dumps(report, indent=2, allow_nan=True))


if __name__ == '__main__':
    main()

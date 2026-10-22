"""Fresh-process large-cohort checks; input generation is included in timings."""
import sys
import time
import numpy as np
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw

case = sys.argv[1]
n = 200000
rng = np.random.default_rng(6301)
event = rng.random(n) < .7
times = rng.integers(1, 1000, n).astype(float)
risks = rng.normal(size=n)
expected = None
if case == 'ordered':
    event[:] = True
    times = np.arange(n, dtype=float)
    risks = -times
    expected = (1., n*(n-1)//2, 0, 0, 0)
elif case == 'all_risk_ties':
    event[:] = True
    times = np.arange(n, dtype=float)
    risks[:] = 0
    expected = (.5, 0, 0, n*(n-1)//2, 0)
elif case == 'all_time_ties':
    event[:] = False
    event[:n//2] = True
    times[:] = 1
    risks[:] = 0
    expected = (.5, 0, 0, n*n//4, n*n//4)
elif case == 'extreme':
    event[:] = True
    times = np.arange(n, dtype=float)
    risks = np.tile([np.finfo(float).max, -np.finfo(float).max], n//2)
    half = n//2
    # Positive and negative risks each tie; positive before negative concords.
    con = half*(half+1)//2
    dis = half*(half-1)//2
    ties = half*(half-1)
    expected = ((con + .5*ties)/(n*(n-1)//2), con, dis, ties, 0)
elif case == 'few_risks':
    risks = rng.integers(0, 5, n).astype(float)
elif case == 'near_ties':
    risks = 1 + rng.normal(size=n)*1e-8
elif case not in ['random', 'ipcw', 'ipcw_truncated']:
    raise ValueError(case)

start = time.monotonic()
if case.startswith('ipcw'):
    test = np.empty(n, dtype=[('event',bool),('time',float)])
    test['event'] = event
    test['time'] = times
    train = test.copy()
    train['time'] *= 2
    train['event'][np.argmax(train['time'])] = True
    result = concordance_index_ipcw(train, test, risks, tau=500 if case.endswith('truncated') else None)
else:
    result = concordance_index_censored(event, times, risks)
if expected is not None:
    assert result == expected, (result, expected)
print(case, round(time.monotonic() - start, 3), result)

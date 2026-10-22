"""Check a source-only transfer under the verifier's unprivileged UID."""
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import tempfile
import time

import numpy as np

SCRIPT = r'''
import json
import os
from pathlib import Path
import pickle
import warnings
import sksurv
sksurv.__path__ = [str(Path.cwd() / 'sksurv'), str(Path(sksurv.__file__).parent)]
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
assert Path(concordance_index_censored.__code__.co_filename).parent == Path.cwd() / 'sksurv'
with open('inputs.pkl', 'rb') as stream:
    event, times, risk, train, test, expected = pickle.load(stream)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    ordinary = concordance_index_censored(event_indicator=event, event_time=times, estimate=risk)
    ipcw = concordance_index_ipcw(survival_train=train, survival_test=test, estimate=risk)
assert not caught
for result in [ordinary, ipcw]:
    assert result[0] == 1.0
    assert result[1:] == (expected, 0, 0, 0)
    assert all(isinstance(v, (int, __import__('numpy').integer)) for v in result[1:])
print(json.dumps([[float(v[0]), *map(int, v[1:])] for v in [ordinary, ipcw]]))
'''

n = 200_000
event = np.ones(n, dtype=bool)
times = np.arange(1, n+1, dtype=float)
risk = -times
survival = np.empty(n, dtype=[('event', '?'), ('time', 'f8')])
survival['event'], survival['time'] = event, times
root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as temp:
    target = Path(temp)
    target.chmod(0o755)
    (target / 'sksurv').mkdir()
    shutil.copyfile(root / 'sksurv' / 'metrics.py', target / 'sksurv' / 'metrics.py')
    with (target / 'inputs.pkl').open('wb') as stream:
        pickle.dump((event, times, risk, survival, survival, n*(n-1)//2), stream)
    (target / 'run.py').write_text(SCRIPT)
    cmd = ['python', '-B', 'run.py']
    if os.getuid() == 0:
        cmd = ['setpriv', '--reuid=10001', '--regid=10001', '--clear-groups', *cmd]
    start = time.monotonic()
    proc = subprocess.run(cmd, cwd=target, capture_output=True, text=True, check=True, timeout=60)
    print(proc.stdout.strip())
    print(f'Fresh process, both APIs, {n:,} records: {time.monotonic()-start:.3f}s')

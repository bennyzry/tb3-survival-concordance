import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_concordance import compare, structured
import numpy as np

rng = np.random.default_rng(9229)
for dtype, uint in [(np.float16, np.uint16), (np.float32, np.uint32), (np.float64, np.uint64)]:
    info = np.finfo(dtype)
    with np.errstate(all='ignore'):
        special = np.array([info.max, -info.max, info.max/2, -info.max/2,
                            0., -0., info.tiny, -info.tiny, info.smallest_subnormal,
                            -info.smallest_subnormal, 1., -1., 16., -16.], dtype=dtype)
        special = np.r_[special, np.nextafter(special, dtype(np.inf)), np.nextafter(special, dtype(-np.inf))]
        special = special[np.isfinite(special)]
    for i in range(1500):
        n = int(rng.integers(2,60))
        e = rng.random(n)<.6
        t = rng.integers(0, max(1,n//2),n).astype(float)
        if i%2:
            r = rng.choice(special, n)
        else:
            r = rng.bytes(n*np.dtype(dtype).itemsize)
            r = np.frombuffer(r, dtype=dtype).copy()
            r[~np.isfinite(r)] = 0
        with np.errstate(all='ignore'):
            diff = np.absolute(r[0]-r[1])
        with np.errstate(over='ignore', invalid='ignore'):
            tol = [0., 1e-8, -1., np.nan, np.inf, -np.inf,
               float(diff), dtype(diff), np.float64(diff),
               np.nextafter(dtype(diff), dtype(-np.inf)),
               np.nextafter(dtype(diff), dtype(np.inf)), 1e300, -1e300,
               10**400, -10**400][i%15]
        compare('concordance_index_censored', (e,t,r), tied_tol=tol)
        if i%20==0:
            train=structured([True,False,True,True,False,True], [1.,2.,2.,3.,4.,100.])
            test=structured(e,t)
            compare('concordance_index_ipcw',(train,test,r),tau=10.,tied_tol=tol)
print('Numeric checks passed')

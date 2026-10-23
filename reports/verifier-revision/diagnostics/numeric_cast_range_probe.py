"""Check the other scalar-promotion boundary: finite tolerance cast overflow."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/probe-source')
from numeric_compatibility_probe import capture, compatible, load, baseline
from cases import ordinary, ipcw

cases=[]
for dtype in ('float16','float32'):
    limit=float(np.finfo(dtype).max)*2
    for sign in (-1,1):
        for repeat in (1,257):
            event=[True,False,True,False,True,False]
            times=[1.,1.,2.,3.,3.,4.]
            risk=[0.,1.,-1.,0.,2.,-2.]
            event,times,risk=([x for x in row for _ in range(repeat)] for row in (event,times,risk))
            name=f'cast-range-{dtype}-{sign}-n{len(event)}'
            cases.append(ordinary(name+'-ordinary',event,times,risk,dtype=dtype,tol=limit*sign))
            cases.append(ipcw(name+'-ipcw',[False,True,False,True,True,True],[.5,1.,2.,3.,5.,6.],
                              event,times,risk,dtype=dtype,tol=limit*sign,tau=3.5))
expected={c['id']:capture(baseline,c) for c in cases}
results={}
for name in ['oracle','alternative','codex-standard-1-replacement','codex-standard-2','codex-standard-3',
             'deepseek-standard-1','deepseek-standard-2','deepseek-standard-3']:
    path=(Path('/reference')/name/'sksurv/metrics.py' if name in ('oracle','alternative')
          else Path('/snapshot/reports/trials')/name/'submission/sksurv/metrics.py')
    module=load(name,path)
    records=[]
    for case in cases:
        actual=capture(module,case)
        records.append({'case':case['id'],'actual':actual,'pristine':expected[case['id']],
                        'matches':compatible(actual,expected[case['id']])})
    results[name]={'checked':len(records),'matched':sum(x['matches'] for x in records),'records':records}
print(json.dumps({'diagnostic_only':True,'new_model_inference':False,'results':results},indent=2))

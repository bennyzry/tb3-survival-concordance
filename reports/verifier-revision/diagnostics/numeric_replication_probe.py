"""Probe the same promotion behavior at an intermediate size via replication."""
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0, '/probe-source')
from numeric_compatibility_probe import promotion_cases, capture, compatible, load, baseline

cases = [c for c in promotion_cases() if c['dtype'] in ('float16','float32')
         and '-0.25-' in c['id']]
results = {}
for name in ['codex-standard-1-replacement','codex-standard-2','codex-standard-3',
             'deepseek-standard-1','deepseek-standard-2','deepseek-standard-3', 'oracle', 'alternative']:
    path = (Path('/reference')/name/'sksurv/metrics.py' if name in ('oracle','alternative')
            else Path('/snapshot/reports/trials')/name/'submission/sksurv/metrics.py')
    module = load(name, path)
    records=[]
    for original in cases:
        case=copy.deepcopy(original)
        factor=257
        for key in ('event','time','risk'):
            case[key]=[v for v in original[key] for _ in range(factor)]
        small=capture(baseline, original)
        expected=copy.deepcopy(small)
        expected['result'][1:]=[int(v)*factor*factor for v in small['result'][1:]]
        pristine=capture(baseline, case)
        assert compatible(pristine, expected),(case['id'],pristine,expected)
        actual=capture(module, case)
        records.append({'case':case['id'],'n':len(case['time']), 'replication_factor':factor,
                        'pristine':pristine,'analytical_replication':expected,
                        'actual':actual,'matches':compatible(actual, pristine)})
    results[name]={'checked':len(records),'matched':sum(r['matches'] for r in records),'records':records}
print(json.dumps({'diagnostic_only':True,'new_model_inference':False,
                  'derivation':'Replicating each subject r times multiplies every eligible pair category by r squared and leaves the weighted score invariant. Training data is unchanged.',
                  'results':results},indent=2))

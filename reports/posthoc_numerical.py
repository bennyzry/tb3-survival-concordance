"""One ungraded analytical check discovered during submitted-source review."""
import importlib.util,json,warnings
from pathlib import Path
import numpy as np
import sksurv
def load(name,path):
    spec=importlib.util.spec_from_file_location('sksurv.'+name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
baseline=load('_pristine_check',Path(sksurv.__file__).parent/'metrics.py')
inputs=(np.array([True,False]),np.array([1.,2.]),np.array([0.,0.],dtype=np.float16))
results={}
for name,path in [('pristine',None),('codex-standard-2','/models/codex-standard-2.py'),
                  ('codex-standard-3','/models/codex-standard-3.py'),('deepseek-standard-1','/models/deepseek-standard-1.py')]:
    module=baseline if path is None else load('_'+name.replace('-','_'),path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result=module.concordance_index_censored(*inputs,tied_tol=-1e-100)
    results[name]={'value':[float(result[0])]+[int(v) for v in result[1:]],
                   'warning_categories':sorted({type(v.message).__name__ for v in caught})}
assert results['pristine']['value']==[.5,0,0,1,0],results['pristine']
for name in results:results[name]['matches_pristine']=results[name]['value']==results['pristine']['value']
print(json.dumps({'graded':False,'fixture':{'event':[True,False],'time':[1.,2.],'risk':[0.,0.],
    'risk_dtype':'float16','tied_tol':-1e-100},'analytical_result':[.5,0,0,1,0],
    'explanation':'One comparable event/censor pair. NumPy converts the tiny negative tolerance to float16 negative zero; zero risk difference is an inclusive tie.',
    'results':results},indent=2))

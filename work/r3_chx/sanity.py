import sys
sys.dont_write_bytecode = True
import time
import numpy as np
sys.argv = ['chx.py', 'none']
exec(open('./work/r3_chx/chx.py').read())
import chx_lib
from chx_lib import SEED0
k = 0
r, fit, val, elig = splits[k]
S = SEED0 + k
ref = dict(np.load(f'{RCACHE}/chfull_std_k{k}.npz'))
for variant in ('0', 'abc'):
    t0 = time.time()
    R = build(variant, 3)
    Xf, (Xv,), Xt = R.matrices(fit, [val], True, lf_seed=k)
    inner = R.inner_sets(fit, S)
    print(variant, 'shapes', Xf.shape, Xv.shape, Xt.shape, [ (I.shape, J.shape) for _, _, I, J in inner], f'{time.time()-t0:.1f}s', flush=True)
    t0 = time.time()
    (rv,), rt, ip = R.run_family('extra', fit, Xf, [Xv], [val], Xt, inner, S)
    ev = R.correct('extra', fit, ip, rv, tr.iloc[val])
    print(variant, 'ET', f'{time.time()-t0:.1f}s', 'max|raw_v - cached|', np.abs(rv - ref['et_raw_v']).max(), 'max|ip-cached|', np.abs(ip - ref['et_ip']).max(), 'max|et_v - cached|', np.abs(ev - ref['et_v']).max())
    from sklearn.metrics import roc_auc_score
    print('   ET auc elig', roc_auc_score(y[val[elig]], ev[elig]), 'cached', roc_auc_score(y[val[elig]], ref['et_v'][elig]))
# tiny catboost with album categorical
chx_lib.CFG['cat'] = dict(chx_lib.CFG['cat'], rounds=20)
t0 = time.time()
(rv,), rt, ip = R.run_family('cat', fit, Xf, [Xv], [val], Xt, inner, S)
print('tiny cat ok', f'{time.time()-t0:.1f}s', rv[:5], rt[:3])

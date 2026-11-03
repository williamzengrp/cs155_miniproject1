import sys, time
sys.path.insert(0, './work/gbm')
import numpy as np
import lib as L
sp = L.H.make_splits(L.tr, L.y, n_rep=1)
r, fit_idx, val_idx, elig = sp[0]
for name in ['lgbm', 'xgb', 'cat', 'et', 'rf']:
    t = time.time()
    m = L.fit_one(name, L.BASE[name], fit_idx, L.y[fit_idx], None, seed=0)
    p = m.predict_proba(L.DATA[name][0].iloc[val_idx])[:, 1]
    print(name, f'{time.time()-t:.1f}s', flush=True)

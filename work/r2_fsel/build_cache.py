"""Cache label-free features + per-fold out-of-fold label features (same seeds as main/final_cv.py).
python -I build_cache.py SEED0 NREP     -> cache/lf_<SEED0>_<NREP>.pkl   (and cache/base.pkl, cache/full.pkl)
Label features: label_features(..., seed=k, spec=BEST_SPEC) for fold k (exactly as final_cv.py),
full refit: seeds 0,1,2 on all training rows."""
import os, sys, time, pickle
import numpy as np, pandas as pd
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/main'); sys.path.insert(0, WORK + '/te')
import harness as H
from features_final import build_features_final
from label_features import label_features, BEST_SPEC
C = WORK + '/r2_fsel/cache/'
SEED0, NREP = int(sys.argv[1]), int(sys.argv[2])
assert SEED0 != 11000, 'c11 is reserved for verification'
tr, te, y = H.load()
t0 = time.time()
if not os.path.exists(C + 'base.pkl'):
    X, Xt = build_features_final(tr, te)
    pickle.dump(dict(X=X, Xt=Xt, y=y), open(C + 'base.pkl', 'wb'))
else:
    b = pickle.load(open(C + 'base.pkl', 'rb')); X, Xt = b['X'], b['Xt']
XA = pd.concat([X, Xt], ignore_index=True)
n, m = len(tr), len(te)
test_idx = np.arange(n, n + m)
print('features', X.shape, f'{time.time()-t0:.0f}s', flush=True)
if not os.path.exists(C + 'full.pkl'):
    full = []
    for s in range(3):
        F_fit, (F_te,) = label_features(XA, y, np.arange(n), [test_idx], tr, te, seed=s, spec=BEST_SPEC)
        full.append((F_fit, F_te))
    pickle.dump(full, open(C + 'full.pkl', 'wb'))
    print('full done', f'{time.time()-t0:.0f}s', flush=True)
splits = H.make_splits(tr, y, n_rep=NREP, seed0=SEED0)
folds = []
for k, (r, fit_idx, val_idx, elig) in enumerate(splits):
    F_fit, (F_val, F_te) = label_features(XA, y, fit_idx, [val_idx, test_idx], tr, te, seed=k, spec=BEST_SPEC)
    folds.append(dict(r=r, fit_idx=fit_idx, val_idx=val_idx, elig=elig, F_fit=F_fit, F_val=F_val, F_te=F_te))
pickle.dump(folds, open(C + f'lf_{SEED0}_{NREP}.pkl', 'wb'))
print('folds done', len(folds), f'{time.time()-t0:.0f}s', flush=True)

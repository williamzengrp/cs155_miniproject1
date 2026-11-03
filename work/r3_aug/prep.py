"""r3_aug: cache Charlie's 115-col label-free view (ch_feats.charlie_pool PROJECT) and the per-fold out-of-fold label
features (te/label_features.py BEST_SPEC, exactly as main/final_cv.py: seed=k for std fold k, fit-fold labels only;
full refit seeds 0,1,2) so that the augmentation variants can reuse them."""
import sys, time, pickle
import numpy as np, pandas as pd
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/ch'); sys.path.insert(0, WORK + '/te')
import harness as H
from ch_feats import charlie_pool
from label_features import label_features, BEST_SPEC
HERE = WORK + '/r3_aug/cache/'
t0 = time.time()
tr, te, y = H.load()
n, m = len(tr), len(te)
F, BL = charlie_pool(tr, te)
XP = F[BL['PROJECT']].astype(np.float32)
X, Xt = XP.iloc[:n].reset_index(drop=True), XP.iloc[n:].reset_index(drop=True)
print('view', X.shape, f'{time.time()-t0:.0f}s', flush=True)
XA = pd.concat([X, Xt], ignore_index=True)
test_idx = np.arange(n, n + m)
splits = H.make_splits(tr, y, n_rep=5, seed0=1000)
LFc = {}
for k, (r, fit_idx, val_idx, elig) in enumerate(splits):
    F_fit, (F_val, F_te) = label_features(XA, y, fit_idx, [val_idx, test_idx], tr, te, seed=k, spec=BEST_SPEC)
    LFc[('cv', k)] = (F_fit.to_numpy(np.float32), F_val.to_numpy(np.float32), F_te.to_numpy(np.float32))
for s in range(3):
    F_fit, (F_te,) = label_features(XA, y, np.arange(n), [test_idx], tr, te, seed=s, spec=BEST_SPEC)
    LFc[('full', s)] = (F_fit.to_numpy(np.float32), None, F_te.to_numpy(np.float32))
pickle.dump(dict(cols=list(XP.columns), X=X.to_numpy(np.float32), Xt=Xt.to_numpy(np.float32), LF=LFc,
                 lf_cols=list(BEST_SPEC)), open(HERE + 'prep.pkl', 'wb'))
print('done', f'{time.time()-t0:.0f}s', flush=True)

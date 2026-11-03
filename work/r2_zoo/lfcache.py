"""Cache the exact per-fold inputs used by work/main/final_cv.py (58 label-free feats + 3 OOF label features),
so that many learners can be screened without recomputing label features.
python -I lfcache.py SEED0 NREP   -> r2_zoo/cache/lf_<SEED0>_<NREP>.pkl
Fold k uses label_features(..., seed=k) exactly as final_cv.py; full refit uses seeds 0,1,2 on all train rows.
Also stores label-free categorical keys (date string, year-month, album-cluster id) from te/label_features.Ctx."""
import sys, time, pickle, warnings
import numpy as np, pandas as pd
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/main'); sys.path.insert(0, WORK + '/te')
import harness as H
from features_final import build_features_final
from label_features import label_features, BEST_SPEC, get_ctx
warnings.filterwarnings('ignore')
SEED0, NREP = int(sys.argv[1]), int(sys.argv[2])
assert SEED0 != 11000, 'c11 is reserved for verification'
t0 = time.time()
tr, te, y = H.load()
X, Xt = build_features_final(tr, te)
XA = pd.concat([X, Xt], ignore_index=True)
n, m = len(tr), len(te)
test_idx = np.arange(n, n + m)
splits = H.make_splits(tr, y, n_rep=NREP, seed0=SEED0)
folds = []
for k, (r, fit_idx, val_idx, elig) in enumerate(splits):
    F_fit, (F_val, F_te) = label_features(XA, y, fit_idx, [val_idx, test_idx], tr, te, seed=k, spec=BEST_SPEC)
    folds.append(dict(r=r, fit_idx=fit_idx, val_idx=val_idx, elig=elig, F_fit=F_fit, F_val=F_val, F_te=F_te))
full = []
for s in range(3):
    F_fit, (F_te,) = label_features(XA, y, np.arange(n), [test_idx], tr, te, seed=s, spec=BEST_SPEC)
    full.append(dict(F_fit=F_fit, F_te=F_te))
ctx = get_ctx(tr, te)
raw = pd.concat([tr, te], ignore_index=True)['track_album_release_date'].astype(str).to_numpy()
cats = dict(date=ctx.key['date'], ym=ctx.key['ym'], year=ctx.key['year'], alb=ctx.key['alb'], raw=raw, t=ctx.t,
            uri=ctx.uri)
pickle.dump(dict(X=X, Xt=Xt, y=y, folds=folds, full=full, cats=cats, seed0=SEED0, nrep=NREP),
            open(f'{WORK}/r2_zoo/cache/lf_{SEED0}_{NREP}.pkl', 'wb'))
print(f'cached {len(folds)} folds in {time.time()-t0:.0f}s', flush=True)

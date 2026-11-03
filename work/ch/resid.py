"""ch agent: re-implementation of the charlie branch's "same-release-date residual correction" on our harness.

Idea (charlie, tune_project.cv_run): after fitting a model on the fit rows, get inner out-of-fold predictions for
the fit rows (3-fold StratifiedGroupKFold, groups = uri joined with identical audio), residual r = y - p_inner,
averaged per song (uri). For a query row (validation / test) with a full YYYY-MM-DD date, look at fit songs with
the SAME date, weight them by a Gaussian kernel of standardized 13-dim audio distance, and add
    alpha * sum_j W_ij r_j / (shrink + sum_j W_ij)
to the model's probability (clipped to [0, 1]). Labels used: fit rows only (inner OOF for their residuals).

Stage 1 (this script, heavy): per fold, base LightGBM (teammate params, 58 features, random_state = k + off) on
the fit rows -> val / test predictions, and inner-OOF predictions of the fit rows. Cached to cache/.
Stage 2 (resid_eval.py, cheap): corrections for a grid of (alpha, bw, shrink), scored with H.run_cv.

usage: python -I resid.py {std|tune} OFF [model]
"""
import json, os, pickle, sys, time, warnings
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from lightgbm import LGBMClassifier
WORK = './work'
sys.path.insert(0, WORK)
sys.path.insert(0, WORK + '/main')
sys.path.insert(0, WORK + '/ch')
import harness as H
from features_final import build_features_final
from ch_common import connected_groups
warnings.filterwarnings('ignore')

kind, OFF = sys.argv[1], int(sys.argv[2])
MODEL = sys.argv[3] if len(sys.argv) > 3 else 'lgbm'
CACHE = WORK + '/ch/cache'
os.makedirs(CACHE, exist_ok=True)
out_path = f'{CACHE}/resid_{MODEL}_{kind}_off{OFF}.pkl'

tr, te, y = H.load()
X, Xt = build_features_final(tr, te)
assert X.shape[1] == 58, X.shape
P = json.load(open(WORK + '/best_params_v3.json'))
groups = connected_groups(tr)
splits = H.make_splits(tr, y, n_rep=5) if kind == 'std' else H.make_splits(tr, y, n_rep=2, seed0=7000)


def mk(seed):
    if MODEL == 'lgbm':
        return LGBMClassifier(**P['lgbm'], random_state=seed, n_jobs=2, verbose=-1)
    raise ValueError(MODEL)


folds = []
t0 = time.time()
for k, (r, fit_idx, val_idx, elig) in enumerate(splits):
    m = mk(k + OFF).fit(X.iloc[fit_idx], y[fit_idx])
    pv = m.predict_proba(X.iloc[val_idx])[:, 1]
    pt = m.predict_proba(Xt)[:, 1]
    inner = np.full(len(fit_idx), np.nan)
    sgk = StratifiedGroupKFold(3, shuffle=True, random_state=OFF + 100 + k)
    for it, iv in sgk.split(fit_idx, y[fit_idx], groups[fit_idx]):
        mi = mk(OFF + 200 + k).fit(X.iloc[fit_idx[it]], y[fit_idx[it]])
        inner[iv] = mi.predict_proba(X.iloc[fit_idx[iv]])[:, 1]
    assert np.isfinite(inner).all()
    folds.append(dict(pv=pv, pt=pt, inner=inner))
    print(f'fold {k} done {time.time()-t0:.0f}s', flush=True)
pickle.dump(dict(kind=kind, off=OFF, model=MODEL, folds=folds), open(out_path, 'wb'))
print('saved', out_path, flush=True)

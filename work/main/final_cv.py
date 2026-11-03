"""Final tree models: 58 label-free features + out-of-fold label features (te agent's BEST_SPEC).
python -I final_cv.py NJ PREFIX PARAMS_JSON SPLITSEED model1 model2 ...
Saves results/<PREFIX>_<model>.npz with OOF (eligible rows), fold-averaged test preds, and test preds from a
full refit on all training rows (3 seeds; label features recomputed with matching inner-fold seeds)."""
import json, sys, time, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/main'); sys.path.insert(0, WORK + '/te')
import harness as H
from features_final import build_features_final
from label_features import label_features, BEST_SPEC
warnings.filterwarnings('ignore')

NJ, PREFIX, PJSON, SPLITSEED = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
MODELS = sys.argv[5:]
P = json.load(open(PJSON))
tr, te, y = H.load()
X, Xt = build_features_final(tr, te)
XA = pd.concat([X, Xt], ignore_index=True)
n, m = len(tr), len(te)
test_idx = np.arange(n, n + m)
fill = X.median()


def mk(name, seed):
    p = dict(P[name])
    if name == 'xgb': return XGBClassifier(**p, tree_method='hist', random_state=seed, n_jobs=NJ)
    if name == 'lgbm': return LGBMClassifier(**p, random_state=seed, n_jobs=NJ, verbose=-1)
    if name == 'cat': return CatBoostClassifier(**p, random_seed=seed, verbose=0, thread_count=NJ, allow_writing_files=False)
    if name == 'et': return ExtraTreesClassifier(**p, random_state=seed, n_jobs=NJ)
    if name == 'rf': return RandomForestClassifier(**p, random_state=seed, n_jobs=NJ)


def mats(name, A, F):
    M = pd.concat([A.reset_index(drop=True), F.reset_index(drop=True)], axis=1)
    return M.fillna(pd.concat([fill, F.median()])) if name in ('et', 'rf') else M


import os
splits = H.make_splits(tr, y, n_rep=int(os.environ.get("NREP", "5")), seed0=SPLITSEED)
DO_FULL = os.environ.get("DO_FULL", "1") == "1"
R = max(s[0] for s in splits) + 1
oof = {nm: np.full((R, n), np.nan) for nm in MODELS}
tst = {nm: np.zeros(m) for nm in MODELS}
t0 = time.time()
for k, (r, fit_idx, val_idx, elig) in enumerate(splits):
    F_fit, (F_val, F_te) = label_features(XA, y, fit_idx, [val_idx, test_idx], tr, te, seed=k, spec=BEST_SPEC)
    for nm in MODELS:
        mdl = mk(nm, k).fit(mats(nm, X.iloc[fit_idx], F_fit), y[fit_idx])
        pv = mdl.predict_proba(mats(nm, X.iloc[val_idx], F_val))[:, 1]
        oof[nm][r, val_idx[elig]] = pv[elig]
        tst[nm] += mdl.predict_proba(mats(nm, Xt, F_te))[:, 1] / len(splits)
    print(f'fold {k} done {time.time()-t0:.0f}s', flush=True)

full = {nm: np.zeros(m) for nm in MODELS}
for s in (range(3) if DO_FULL else []):
    F_fit, (F_te,) = label_features(XA, y, np.arange(n), [test_idx], tr, te, seed=s, spec=BEST_SPEC)
    for nm in MODELS:
        mdl = mk(nm, s).fit(mats(nm, X, F_fit), y)
        full[nm] += mdl.predict_proba(mats(nm, Xt, F_te))[:, 1] / 3

from sklearn.metrics import roc_auc_score
for nm in MODELS:
    aucs = np.array([roc_auc_score(y[~np.isnan(oof[nm][r])], oof[nm][r][~np.isnan(oof[nm][r])]) for r in range(R)])
    res = dict(oof=oof[nm], aucs=aucs, auc=float(aucs.mean()), test_foldavg=tst[nm])
    H.save(f'{PREFIX}_{nm}', res, full[nm])
    print(f'[{PREFIX}_{nm}] AUC {aucs.mean():.5f} per-rep {np.round(aucs, 4)}  corr(foldavg,full)={np.corrcoef(tst[nm], full[nm])[0,1]:.4f}', flush=True)

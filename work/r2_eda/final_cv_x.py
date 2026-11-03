"""r2_eda: final tree models (same as main/final_cv.py) + optional EDA-derived extras.
python -I final_cv_x.py NJ PREFIX PARAMS_JSON SPLITSEED model1 model2 ...
env: NREP (default 5), DO_FULL (default 1; full refit on all training rows, 3 seeds, saved as test_full),
     XBLOCKS  '+'-joined label-free blocks from eda_feats.py (e.g. 'mdup'), default none
     XSPEC    '+'-joined extra label-feature specs from eda_label.py (e.g. 'sum_date+neg_date'), default none
     MONO     '1' -> LightGBM/XGBoost monotone-increasing constraints on knnw_yr_40, teh_date_m10 and XSPEC sum_/te_ cols
Label features: BEST_SPEC (+ XSPEC) computed exactly as in main/final_cv.py (fit rows out-of-fold via inner
StratifiedGroupKFold grouped by uri, seed = fold number; val/test rows from all fit_idx labels only).
Saves results/<PREFIX>_<model>.npz.
"""
import json, os, sys, time, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/main'); sys.path.insert(0, WORK + '/te'); sys.path.insert(0, WORK + '/r2_eda')
import harness as H
from features_final import build_features_final
from label_features import BEST_SPEC
from eda_label import label_features_ext
from eda_feats import extra_block
warnings.filterwarnings('ignore')

NJ, PREFIX, PJSON, SPLITSEED = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
MODELS = sys.argv[5:]
XBLOCKS = [b for b in os.environ.get('XBLOCKS', '').split('+') if b]
XSPEC = tuple(s for s in os.environ.get('XSPEC', '').split('+') if s)
MONO = os.environ.get('MONO', '0') == '1'
SPEC = tuple(BEST_SPEC) + XSPEC
P = json.load(open(PJSON))
tr, te, y = H.load()
X, Xt = build_features_final(tr, te)
if XBLOCKS:
    EX, EXt = extra_block(tr, te, XBLOCKS)
    X = pd.concat([X, EX], axis=1)
    Xt = pd.concat([Xt, EXt], axis=1)
XA = pd.concat([X, Xt], ignore_index=True)
n, m = len(tr), len(te)
test_idx = np.arange(n, n + m)
fill = X.median()
INC = {'knnw_yr_40', 'teh_date_m10'} | {s for s in XSPEC if s.startswith(('sum_', 'te_', 'dsum_', 'mhi_'))}


def mk(name, seed, cols):
    p = dict(P[name])
    if name == 'xgb':
        if MONO:
            p['monotone_constraints'] = '(' + ','.join('1' if c in INC else '0' for c in cols) + ')'
        return XGBClassifier(**p, tree_method='hist', random_state=seed, n_jobs=NJ)
    if name == 'lgbm':
        if MONO:
            p.update(monotone_constraints=[1 if c in INC else 0 for c in cols], monotone_constraints_method='advanced')
        return LGBMClassifier(**p, random_state=seed, n_jobs=NJ, verbose=-1)
    if name == 'cat': return CatBoostClassifier(**p, random_seed=seed, verbose=0, thread_count=NJ, allow_writing_files=False)
    if name == 'et': return ExtraTreesClassifier(**p, random_state=seed, n_jobs=NJ)
    if name == 'rf': return RandomForestClassifier(**p, random_state=seed, n_jobs=NJ)


def mats(name, A, F):
    M = pd.concat([A.reset_index(drop=True), F.reset_index(drop=True)], axis=1)
    return M.fillna(pd.concat([fill, F.median()])) if name in ('et', 'rf') else M


splits = H.make_splits(tr, y, n_rep=int(os.environ.get('NREP', '5')), seed0=SPLITSEED)
DO_FULL = os.environ.get('DO_FULL', '1') == '1'
R = max(s[0] for s in splits) + 1
oof = {nm: np.full((R, n), np.nan) for nm in MODELS}
tst = {nm: np.zeros(m) for nm in MODELS}
t0 = time.time()
for k, (r, fit_idx, val_idx, elig) in enumerate(splits):
    F_fit, (F_val, F_te) = label_features_ext(XA, y, fit_idx, [val_idx, test_idx], tr, te, seed=k, spec=SPEC)
    for nm in MODELS:
        Mf = mats(nm, X.iloc[fit_idx], F_fit)
        mdl = mk(nm, k, list(Mf.columns)).fit(Mf, y[fit_idx])
        pv = mdl.predict_proba(mats(nm, X.iloc[val_idx], F_val))[:, 1]
        oof[nm][r, val_idx[elig]] = pv[elig]
        tst[nm] += mdl.predict_proba(mats(nm, Xt, F_te))[:, 1] / len(splits)
    print(f'fold {k} done {time.time()-t0:.0f}s', flush=True)

full = {nm: np.zeros(m) for nm in MODELS}
for s in (range(3) if DO_FULL else []):
    F_fit, (F_te,) = label_features_ext(XA, y, np.arange(n), [test_idx], tr, te, seed=s, spec=SPEC)
    for nm in MODELS:
        Mf = mats(nm, X, F_fit)
        mdl = mk(nm, s, list(Mf.columns)).fit(Mf, y)
        full[nm] += mdl.predict_proba(mats(nm, Xt, F_te))[:, 1] / 3

for nm in MODELS:
    aucs = np.array([roc_auc_score(y[~np.isnan(oof[nm][r])], oof[nm][r][~np.isnan(oof[nm][r])]) for r in range(R)])
    res = dict(oof=oof[nm], aucs=aucs, auc=float(aucs.mean()), test_foldavg=tst[nm])
    H.save(f'{PREFIX}_{nm}', res, full[nm] if DO_FULL else None)
    print(f'[{PREFIX}_{nm}] AUC {aucs.mean():.5f} per-rep {np.round(aucs, 4)}  '
          f'corr(foldavg,full)={np.corrcoef(tst[nm], full[nm])[0,1] if DO_FULL else float("nan"):.4f}', flush=True)

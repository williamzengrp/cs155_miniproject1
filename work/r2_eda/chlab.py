"""r2_eda: Charlie's 115-col label-free view (ch/ch_feats.py PROJECT, same as ch_et_proj / ch_lgbm_proj) PLUS the
te agent's out-of-fold label features (BEST_SPEC = knnw_yr_40, teh_date_m10, cnt_date [+ optional XSPEC]).
Motivation (error analysis, r2_eda/eda5.out): the label-free ch members lose ~0.015-0.02 AUC vs the label-feature GBMs
on rows with >= 4 labelled date-mates; this gives them the date-mate label signal.

python -I chlab.py SEED0 NREP PREFIX model1[,model2]      models: et (charlie ET, 1/count weights), lgbm (teammate params)
env: DO_FULL=1 -> also full refit on all training rows (3 seeds, label features recomputed with inner seeds 0/1/2)
     NJ (threads, default 2), XSPEC ('+'-joined extra label specs from eda_label.py)
Saves results/<PREFIX>_chl_<model>.npz (OOF on eligible rows, fold-avg test preds, test_full).
Leakage: label features exactly as main/final_cv.py (fit rows out-of-fold via inner StratifiedGroupKFold grouped by uri,
seed = fold number; val/test rows from fit_idx labels only).
"""
import json, os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/ch'); sys.path.insert(0, WORK + '/te'); sys.path.insert(0, WORK + '/r2_eda')
import harness as H
from ch_feats import charlie_pool
from label_features import BEST_SPEC
from eda_label import label_features_ext
warnings.filterwarnings('ignore')

SEED0, NREP, PREFIX = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
MODELS = sys.argv[4].split(',')
DO_FULL = os.environ.get('DO_FULL', '0') == '1'
NJ = int(os.environ.get('NJ', '2'))
SPEC = tuple(BEST_SPEC) + tuple(s for s in os.environ.get('XSPEC', '').split('+') if s)
tr, te, y = H.load()
n, m = len(tr), len(te)
F, BL = charlie_pool(tr, te)
XP = F[BL['PROJECT']].astype(np.float32)
X, Xt = XP.iloc[:n].reset_index(drop=True), XP.iloc[n:].reset_index(drop=True)
XA = pd.concat([X, Xt], ignore_index=True)
test_idx = np.arange(n, n + m)
P = json.load(open(WORK + '/best_params_v3.json'))
uri = tr['uri'].to_numpy()


def weights(idx):
    u = pd.Series(uri[idx])
    w = 1.0 / u.map(u.value_counts()).to_numpy(float)
    return w / w.mean()


def fit_predict(model, fit_idx, F_fit, frames, seed):
    A = pd.concat([X.iloc[fit_idx].reset_index(drop=True), F_fit.reset_index(drop=True)], axis=1)
    if model == 'et':
        med = A.median()
        mdl = ExtraTreesClassifier(n_estimators=900, max_depth=15, min_samples_leaf=2, max_features=0.1827077108658931,
                                   n_jobs=NJ, random_state=seed)
        mdl.fit(A.fillna(med), y[fit_idx], sample_weight=weights(fit_idx))
        return [mdl.predict_proba(B.fillna(med))[:, 1] for B in frames]
    mdl = LGBMClassifier(**P['lgbm'], random_state=seed, n_jobs=int(os.environ.get('NJL', '1')), verbose=-1).fit(A, y[fit_idx])
    return [mdl.predict_proba(B)[:, 1] for B in frames]


splits = H.make_splits(tr, y, n_rep=NREP, seed0=SEED0)
oof = {md: np.full((NREP, n), np.nan) for md in MODELS}
tst = {md: np.zeros(m) for md in MODELS}
t0 = time.time()
for k, (r, fit_idx, val_idx, elig) in enumerate(splits):
    F_fit, (F_val, F_te) = label_features_ext(XA, y, fit_idx, [val_idx, test_idx], tr, te, seed=k, spec=SPEC)
    Bv = pd.concat([X.iloc[val_idx].reset_index(drop=True), F_val.reset_index(drop=True)], axis=1)
    Bt = pd.concat([Xt.reset_index(drop=True), F_te.reset_index(drop=True)], axis=1)
    for md in MODELS:
        pv, pt = fit_predict(md, fit_idx, F_fit, [Bv, Bt], k)
        oof[md][r, val_idx[elig]] = pv[elig]
        tst[md] += pt / len(splits)
    if k % 5 == 4:
        print(f'fold {k} {time.time()-t0:.0f}s', flush=True)
full = {md: np.zeros(m) for md in MODELS}
for s in (range(3) if DO_FULL else []):
    F_fit, (F_te,) = label_features_ext(XA, y, np.arange(n), [test_idx], tr, te, seed=s, spec=SPEC)
    Bt = pd.concat([Xt.reset_index(drop=True), F_te.reset_index(drop=True)], axis=1)
    for md in MODELS:
        full[md] += fit_predict(md, np.arange(n), F_fit, [Bt], s)[0] / 3
for md in MODELS:
    aucs = np.array([roc_auc_score(y[~np.isnan(oof[md][r])], oof[md][r][~np.isnan(oof[md][r])]) for r in range(NREP)])
    res = dict(oof=oof[md], aucs=aucs, auc=float(aucs.mean()), test_foldavg=tst[md])
    H.save(f'{PREFIX}_chl_{md}', res, full[md] if DO_FULL else None)
    print(f'[{PREFIX}_chl_{md}] AUC {aucs.mean():.5f} per-rep {np.round(aucs, 4)}'
          + (f'  corr(foldavg,full)={np.corrcoef(tst[md], full[md])[0, 1]:.4f}' if DO_FULL else ''), flush=True)

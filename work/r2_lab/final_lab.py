"""Tree members with a redesigned label-feature block (copy of work/main/final_cv.py using r2_lab/labx.py).
python -I final_lab.py NJ PREFIX PARAMS_JSON SPLITSEED model1 model2 ...
env: NREP (5), DO_FULL (1), SPEC (comma list, default te BEST_SPEC), DEDUP (0/1), NINNER (5)
Saves results/<PREFIX>_<model>.npz: OOF over eligible rows (1 model seed = fold no. k, label inner seed = k),
fold-averaged test preds, and test_full = full refit on ALL training rows averaged over 3 seeds (s = 0,1,2;
label features recomputed with inner seed s). Same protocol as final_cv.py, only the label block differs."""
import json, os, sys, time, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/main'); sys.path.insert(0, WORK + '/r2_lab')
import harness as H
from features_final import build_features_final
import labx as LX
warnings.filterwarnings('ignore')

NJ, PREFIX, PJSON, SPLITSEED = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
MODELS = sys.argv[5:]
SPEC = [s for s in os.environ.get('SPEC', ','.join(LX.BEST_SPEC)).split(',') if s]
DEDUP = os.environ.get('DEDUP', '0') == '1'
NINNER = int(os.environ.get('NINNER', '5'))
P = json.load(open(PJSON))
tr, te, y = H.load()
X, Xt = build_features_final(tr, te)
XA = pd.concat([X, Xt], ignore_index=True)
n, m = len(tr), len(te)
test_idx = np.arange(n, n + m)
fill = X.median()
print(f'SPEC={SPEC} DEDUP={DEDUP} NINNER={NINNER}', flush=True)


def lab(fit_idx, apply_list, seed):
    return LX.label_features(XA, y, fit_idx, apply_list, tr, te, seed=seed, spec=SPEC, n_inner=NINNER, dedup=DEDUP)


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


splits = H.make_splits(tr, y, n_rep=int(os.environ.get('NREP', '5')), seed0=SPLITSEED)
DO_FULL = os.environ.get('DO_FULL', '1') == '1'
R = max(s[0] for s in splits) + 1
oof = {nm: np.full((R, n), np.nan) for nm in MODELS}
tst = {nm: np.zeros(m) for nm in MODELS}
t0 = time.time()
for k, (r, fit_idx, val_idx, elig) in enumerate(splits):
    F_fit, (F_val, F_te) = lab(fit_idx, [val_idx, test_idx], k)
    for nm in MODELS:
        mdl = mk(nm, k).fit(mats(nm, X.iloc[fit_idx], F_fit), y[fit_idx])
        pv = mdl.predict_proba(mats(nm, X.iloc[val_idx], F_val))[:, 1]
        oof[nm][r, val_idx[elig]] = pv[elig]
        tst[nm] += mdl.predict_proba(mats(nm, Xt, F_te))[:, 1] / len(splits)
    print(f'fold {k} done {time.time()-t0:.0f}s', flush=True)

full = {nm: np.zeros(m) for nm in MODELS}
for s in (range(3) if DO_FULL else []):
    F_fit, (F_te,) = lab(np.arange(n), [test_idx], s)
    for nm in MODELS:
        mdl = mk(nm, s).fit(mats(nm, X, F_fit), y)
        full[nm] += mdl.predict_proba(mats(nm, Xt, F_te))[:, 1] / 3

for nm in MODELS:
    aucs = np.array([roc_auc_score(y[~np.isnan(oof[nm][r])], oof[nm][r][~np.isnan(oof[nm][r])]) for r in range(R)])
    res = dict(oof=oof[nm], aucs=aucs, auc=float(aucs.mean()), test_foldavg=tst[nm])
    H.save(f'{PREFIX}_{nm}', res, full[nm] if DO_FULL else None)
    c = np.corrcoef(tst[nm], full[nm])[0, 1] if DO_FULL else float('nan')
    print(f'[{PREFIX}_{nm}] AUC {aucs.mean():.5f} per-rep {np.round(aucs, 4)}  corr(foldavg,full)={c:.4f}', flush=True)

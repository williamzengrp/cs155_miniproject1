"""Tree members with distribution-matched apply-time label features (agent `shift`).
python -I match_cv.py NJ PREFIX PARAMS_JSON SPLITSEED model1 model2 ...
env: NREP (5), DO_FULL (1 -> full refit on all train rows, seeds 0..2, saved as test_full), NDS (1: apply-time
     draws from NDS inner-split seeds x 5 folds)
Training is IDENTICAL to main/final_cv.py (same features, label features, seeds; with main/final_params.json the
'full' outputs reproduce fkd_lgbm / fkt_xgb / fkt_cat / fkd_et / fkd_rf). Only the apply-time label features differ:
  <PREFIX>_full_<model>  : val/test label features from all fit labels (== final_cv.py)
  <PREFIX>_match_<model> : mean prediction over the 5 inner-fold source sets (fit-time distribution, matched_lf.py)
  <PREFIX>_mix_<model>   : 0.5 * full + 0.5 * match (probabilities)
Saves results/<PREFIX>_{full,match,mix}_<model>.npz."""
import json, os, sys, time, pickle, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
W = './work'
sys.path.insert(0, W); sys.path.insert(0, W + '/main'); sys.path.insert(0, W + '/te'); sys.path.insert(0, W + '/r2_shift')
import harness as H
from matched_lf import label_features_matched
warnings.filterwarnings('ignore')
C = W + '/r2_shift/cache/'

NJ, PREFIX, PJSON, SPLITSEED = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
MODELS = sys.argv[5:]
NREP = int(os.environ.get('NREP', '5'))
DO_FULL = os.environ.get('DO_FULL', '1') == '1'
NDS = int(os.environ.get("NDS", "1"))   # apply-time draw seeds (5 draws each)
P = json.load(open(PJSON))
tr, te, y = H.load()
if os.path.exists(C + 'base.pkl'):
    b = pickle.load(open(C + 'base.pkl', 'rb')); X, Xt = b['X'], b['Xt']
else:
    from features_final import build_features_final
    X, Xt = build_features_final(tr, te)
XA = pd.concat([X, Xt], ignore_index=True)
n, m = len(tr), len(te)
test_idx = np.arange(n, n + m)
fill = X.median()
VARS = ('full', 'match', 'mix')


def mk(name, seed):
    p = dict(P[name])
    if name == 'xgb': return XGBClassifier(**p, tree_method='hist', random_state=seed, n_jobs=NJ)
    if name == 'lgbm': return LGBMClassifier(**p, random_state=seed, n_jobs=NJ, verbose=-1)
    if name == 'cat': return CatBoostClassifier(**p, random_seed=seed, verbose=0, thread_count=NJ, allow_writing_files=False)
    if name == 'et': return ExtraTreesClassifier(**p, random_state=seed, n_jobs=NJ)
    if name == 'rf': return RandomForestClassifier(**p, random_state=seed, n_jobs=NJ)


def mats(name, A, F, Ffit):
    """Ffit: the fit-row label features (median fill for et/rf uses the fit-row medians, as final_cv.py)."""
    M = pd.concat([A.reset_index(drop=True), F.reset_index(drop=True)], axis=1)
    return M.fillna(pd.concat([fill, Ffit.median()])) if name in ('et', 'rf') else M


def predict_variants(mdl, nm, A, F_full, F_inner, F_fit):
    pf = mdl.predict_proba(mats(nm, A, F_full, F_fit))[:, 1]
    pm = np.mean([mdl.predict_proba(mats(nm, A, Fj, F_fit))[:, 1] for Fj in F_inner], axis=0)
    return {'full': pf, 'match': pm, 'mix': 0.5 * (pf + pm)}


splits = H.make_splits(tr, y, n_rep=NREP, seed0=SPLITSEED)
R = max(s[0] for s in splits) + 1
oof = {(v, nm): np.full((R, n), np.nan) for v in VARS for nm in MODELS}
tst = {(v, nm): np.zeros(m) for v in VARS for nm in MODELS}
t0 = time.time()
for k, (r, fit_idx, val_idx, elig) in enumerate(splits):
    F_fit, (Fv, Ft), (Fv_in, Ft_in) = label_features_matched(XA, y, fit_idx, [val_idx, test_idx], tr, te, seed=k, n_draw_seeds=NDS)
    for nm in MODELS:
        mdl = mk(nm, k).fit(mats(nm, X.iloc[fit_idx], F_fit, F_fit), y[fit_idx])
        pv = predict_variants(mdl, nm, X.iloc[val_idx], Fv, Fv_in, F_fit)
        pt = predict_variants(mdl, nm, Xt, Ft, Ft_in, F_fit)
        for v in VARS:
            oof[(v, nm)][r, val_idx[elig]] = pv[v][elig]
            tst[(v, nm)] += pt[v] / len(splits)
    if k % 5 == 4:
        print(f'fold {k} done {time.time()-t0:.0f}s', flush=True)

full = {(v, nm): np.zeros(m) for v in VARS for nm in MODELS}
if DO_FULL:
    for s in range(3):
        F_fit, (Ft,), (Ft_in,) = label_features_matched(XA, y, np.arange(n), [test_idx], tr, te, seed=s, n_draw_seeds=NDS)
        for nm in MODELS:
            mdl = mk(nm, s).fit(mats(nm, X, F_fit, F_fit), y)
            pt = predict_variants(mdl, nm, Xt, Ft, Ft_in, F_fit)
            for v in VARS:
                full[(v, nm)] += pt[v] / 3

for nm in MODELS:
    for v in VARS:
        o = oof[(v, nm)]
        aucs = np.array([roc_auc_score(y[~np.isnan(o[r])], o[r][~np.isnan(o[r])]) for r in range(R)])
        res = dict(oof=o, aucs=aucs, auc=float(aucs.mean()), test_foldavg=tst[(v, nm)])
        H.save(f'{PREFIX}_{v}_{nm}', res, full[(v, nm)] if DO_FULL else None)
        cf = np.corrcoef(tst[(v, nm)], full[(v, nm)])[0, 1] if DO_FULL else float('nan')
        print(f'[{PREFIX}_{v}_{nm}] AUC {aucs.mean():.5f} per-rep {np.round(aucs, 4)}  corr(foldavg,full)={cf:.4f}',
              flush=True)
print(f'done {time.time()-t0:.0f}s', flush=True)

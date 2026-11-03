"""Tree members with optional training-row weights / variants (agent `shift`). Mirrors main/final_cv.py exactly
when WS=none and NSEED=1 (same features, label features, seeds, median fill for et/rf).
python -I shift_cv.py NJ PREFIX PARAMS_JSON SPLITSEED model1 model2 ...
env: NREP (5), DO_FULL (1: full refit on all train rows, 3 seeds -> test_full), NSEED (1: seeds per fold,
     seed = k + 100*s; full refit seeds s + 100*j), WS (training-row weight scheme, default none):
       none  : unweighted (== final_cv.py)
       adv   : adversarial density ratio p/(1-p) (train-vs-test LightGBM on the 58 label-free features,
               grouped by uri, OOF, cache/adv.pkl), clipped to [0.1, 10], normalised to mean 1 on the fit rows
       advs  : same but from the singletons-only adversarial model; rows with a within-train copy keep weight 1
       advT  : sqrt of adv (tempered)
Label features: cached per fold (cache/lf_<SEED0>_<NREP>.pkl, identical to final_cv.py) when available, else
computed on the fly with label_features(..., seed=k) exactly as final_cv.py (fit rows inner-OOF by uri).
Saves results/<PREFIX>_<model>.npz."""
import json, os, sys, time, pickle, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
W = './work'
sys.path.insert(0, W); sys.path.insert(0, W + '/main'); sys.path.insert(0, W + '/te')
import harness as H
from label_features import label_features, BEST_SPEC
warnings.filterwarnings('ignore')
C = W + '/r2_shift/cache/'

NJ, PREFIX, PJSON, SPLITSEED = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
MODELS = sys.argv[5:]
NREP = int(os.environ.get('NREP', '5'))
DO_FULL = os.environ.get('DO_FULL', '1') == '1'
NSEED = int(os.environ.get('NSEED', '1'))
WS = os.environ.get('WS', 'none')
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
cnt_tr = tr['uri'].map(tr['uri'].value_counts()).to_numpy()

# ---------------- training-row weights (label-free) ----------------
w_row = np.ones(n)
if WS != 'none':
    adv = pickle.load(open(C + 'adv.pkl', 'rb'))
    if WS in ('adv', 'advT'):
        p = np.clip(adv['oof_all'][:n], 0.01, 0.99)
        w_row = np.clip(p / (1 - p) * (n / m), 0.1, 10)
        if WS == 'advT':
            w_row = np.sqrt(w_row)
    elif WS == 'advs':
        rows_s, oof_s = adv['rows_s'], adv['oof_s']
        tr_s = rows_s < n
        p = np.clip(oof_s[tr_s], 0.01, 0.99)
        w_row = np.ones(n)
        w_row[rows_s[tr_s]] = np.clip(p / (1 - p) * (tr_s.sum() / (~tr_s).sum()), 0.1, 10)
    else:
        raise ValueError(WS)
print(f'WS={WS} weights: mean {w_row.mean():.3f} sd {w_row.std():.3f} min {w_row.min():.3f} max {w_row.max():.3f}; '
      f'mean by multiplicity ' + str({k: round(float(w_row[cnt_tr == k].mean()), 3) for k in (1, 2, 3)}), flush=True)


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


def fit(name, seed, Xf, yf, wf):
    mdl = mk(name, seed)
    if WS == 'none':
        return mdl.fit(Xf, yf)
    return mdl.fit(Xf, yf, sample_weight=wf / wf.mean())


splits = H.make_splits(tr, y, n_rep=NREP, seed0=SPLITSEED)
cache_f = C + f'lf_{SPLITSEED}_{NREP}.pkl'
LF = pickle.load(open(cache_f, 'rb')) if os.path.exists(cache_f) else None
R = max(s[0] for s in splits) + 1
oof = {nm: np.full((R, n), np.nan) for nm in MODELS}
tst = {nm: np.zeros(m) for nm in MODELS}
t0 = time.time()
for k, (r, fit_idx, val_idx, elig) in enumerate(splits):
    if LF is not None:
        f = LF[k]
        assert np.array_equal(f['fit_idx'], fit_idx) and np.array_equal(f['val_idx'], val_idx)
        F_fit, F_val, F_te = f['F_fit'], f['F_val'], f['F_te']
    else:
        F_fit, (F_val, F_te) = label_features(XA, y, fit_idx, [val_idx, test_idx], tr, te, seed=k, spec=BEST_SPEC)
    for nm in MODELS:
        Xf, Xv, Xtt = mats(nm, X.iloc[fit_idx], F_fit), mats(nm, X.iloc[val_idx], F_val), mats(nm, Xt, F_te)
        pv = np.zeros(len(val_idx))
        for s in range(NSEED):
            mdl = fit(nm, k + 100 * s, Xf, y[fit_idx], w_row[fit_idx])
            pv += mdl.predict_proba(Xv)[:, 1] / NSEED
            tst[nm] += mdl.predict_proba(Xtt)[:, 1] / (len(splits) * NSEED)
        oof[nm][r, val_idx[elig]] = pv[elig]
    if k % 5 == 4:
        print(f'fold {k} done {time.time()-t0:.0f}s', flush=True)

full = {nm: np.zeros(m) for nm in MODELS}
if DO_FULL:
    fc = C + 'full.pkl'
    FULL = pickle.load(open(fc, 'rb')) if os.path.exists(fc) else None
    for s in range(3):
        if FULL is not None:
            F_fit, F_te = FULL[s]
        else:
            F_fit, (F_te,) = label_features(XA, y, np.arange(n), [test_idx], tr, te, seed=s, spec=BEST_SPEC)
        for nm in MODELS:
            Xf, Xtt = mats(nm, X, F_fit), mats(nm, Xt, F_te)
            for j in range(NSEED):
                mdl = fit(nm, s + 100 * j, Xf, y, w_row)
                full[nm] += mdl.predict_proba(Xtt)[:, 1] / (3 * NSEED)

for nm in MODELS:
    aucs = np.array([roc_auc_score(y[~np.isnan(oof[nm][r])], oof[nm][r][~np.isnan(oof[nm][r])]) for r in range(R)])
    res = dict(oof=oof[nm], aucs=aucs, auc=float(aucs.mean()), test_foldavg=tst[nm])
    H.save(f'{PREFIX}_{nm}', res, full[nm] if DO_FULL else None)
    cf = np.corrcoef(tst[nm], full[nm])[0, 1] if DO_FULL else float('nan')
    print(f'[{PREFIX}_{nm}] AUC {aucs.mean():.5f} per-rep {np.round(aucs, 4)}  corr(foldavg,full)={cf:.4f}  '
          f'{time.time()-t0:.0f}s', flush=True)

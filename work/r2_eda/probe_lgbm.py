"""r2_eda probe: LightGBM (teammate params) on 58 final feats + BEST_SPEC label features, with/without extra blocks.
python -I probe_lgbm.py SEED0 NREP NSEEDS variant1 variant2 ...
variant syntax: name=block1+block2[:inner<N>]   e.g. base=  mdup=mdup  inner10=:inner10
Prints per-variant AUC and paired diff vs the first variant. Saves OOF to r2_eda/cache/probe_<seed0>_<name>.npz.
"""
import json, os, sys, time, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/main'); sys.path.insert(0, WORK + '/te'); sys.path.insert(0, WORK + '/r2_eda')
import harness as H
from features_final import build_features_final
from label_features import label_features, BEST_SPEC
from eda_feats import extra_block
warnings.filterwarnings('ignore')

SEED0, NREP, NSEEDS = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
VARS = []
for a in sys.argv[4:]:
    name, rest = a.split('=')
    blocks, inner = rest, 5
    if ':' in rest:
        blocks, opt = rest.split(':')
        inner = int(opt.replace('inner', ''))
    VARS.append((name, [b for b in blocks.split('+') if b], inner))
P = json.load(open(WORK + '/best_params_v3.json'))['lgbm']
tr, te, y = H.load()
X, Xt = build_features_final(tr, te)
XA = pd.concat([X, Xt], ignore_index=True)
n, m = len(tr), len(te)
test_idx = np.arange(n, n + m)
EX = {}
for name, blocks, inner in VARS:
    EX[name] = extra_block(tr, te, blocks)[0]
splits = H.make_splits(tr, y, n_rep=NREP, seed0=SEED0)
oof = {v[0]: np.full((NREP, n), np.nan) for v in VARS}
t0 = time.time()
for k, (r, fit_idx, val_idx, elig) in enumerate(splits):
    LF = {}
    for name, blocks, inner in VARS:
        if inner not in LF:
            LF[inner] = label_features(XA, y, fit_idx, [val_idx], tr, te, seed=k, spec=BEST_SPEC, n_inner=inner)
        F_fit, (F_val,) = LF[inner]
        A_fit = [X.iloc[fit_idx].reset_index(drop=True), F_fit.reset_index(drop=True)]
        A_val = [X.iloc[val_idx].reset_index(drop=True), F_val.reset_index(drop=True)]
        if EX[name] is not None:
            A_fit.append(EX[name].iloc[fit_idx].reset_index(drop=True))
            A_val.append(EX[name].iloc[val_idx].reset_index(drop=True))
        Mf, Mv = pd.concat(A_fit, axis=1), pd.concat(A_val, axis=1)
        pv = 0
        for s in range(NSEEDS):
            mdl = LGBMClassifier(**P, random_state=k + 100 * s, n_jobs=2, verbose=-1).fit(Mf, y[fit_idx])
            pv = pv + mdl.predict_proba(Mv)[:, 1] / NSEEDS
        oof[name][r, val_idx[elig]] = pv[elig]
    if k % 5 == 4:
        print(f'fold {k} {time.time()-t0:.0f}s', flush=True)
os.makedirs(WORK + '/r2_eda/cache', exist_ok=True)
A = {}
for name, *_ in VARS:
    A[name] = np.array([roc_auc_score(y[~np.isnan(oof[name][r])], oof[name][r][~np.isnan(oof[name][r])]) for r in range(NREP)])
    np.savez(WORK + f'/r2_eda/cache/probe_{SEED0}_{name}.npz', oof=oof[name], aucs=A[name])
b = VARS[0][0]
for name, *_ in VARS:
    d = A[name] - A[b]
    print(f'{name:14s} AUC {A[name].mean():.5f} per-rep {np.round(A[name], 4)}  diff vs {b} {d.mean():+.5f} pos {(d > 0).sum()}/{NREP} {np.round(d, 4)}')

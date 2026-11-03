"""r2_eda probe v2: LightGBM (teammate params) on 58 final feats + label features (BEST_SPEC + optional extra specs from
eda_label.py) + optional label-free blocks (eda_feats.py) + optional LightGBM monotone constraints on label features.
python -I probe2.py SEED0 NREP NSEEDS 'name|blocks|inner|extraspecs|opts' ...
  blocks / extraspecs: '+'-joined or '-' ; opts: '-' or 'mono' (monotone increasing in knnw_yr_40, teh_date_m10, and
  any sum_/te_ extra spec).
Prints per-variant AUC and paired diff vs the first variant; OOF saved to r2_eda/cache/p2_<seed0>_<name>.npz.
"""
import json, os, sys, time, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/main'); sys.path.insert(0, WORK + '/te'); sys.path.insert(0, WORK + '/r2_eda')
import harness as H
from features_final import build_features_final
from label_features import BEST_SPEC
from eda_label import label_features_ext
from eda_feats import extra_block
warnings.filterwarnings('ignore')

SEED0, NREP, NSEEDS = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
VARS = []
for a in sys.argv[4:]:
    name, blocks, inner, xs, opts = a.split('|')
    VARS.append(dict(name=name, blocks=[b for b in blocks.split('+') if b and b != '-'], inner=int(inner),
                     spec=tuple(BEST_SPEC) + tuple(s for s in xs.split('+') if s and s != '-'), opts=opts))
P = json.load(open(WORK + '/best_params_v3.json'))['lgbm']
tr, te, y = H.load()
X, Xt = build_features_final(tr, te)
XA = pd.concat([X, Xt], ignore_index=True)
n, m = len(tr), len(te)
for v in VARS:
    v['EX'] = extra_block(tr, te, v['blocks'])[0]
splits = H.make_splits(tr, y, n_rep=NREP, seed0=SEED0)
oof = {v['name']: np.full((NREP, n), np.nan) for v in VARS}
t0 = time.time()
for k, (r, fit_idx, val_idx, elig) in enumerate(splits):
    cache = {}
    for v in VARS:
        key = (v['inner'], v['spec'])
        if key not in cache:
            cache[key] = label_features_ext(XA, y, fit_idx, [val_idx], tr, te, seed=k, spec=v['spec'], n_inner=v['inner'])
        F_fit, (F_val,) = cache[key]
        A_fit = [X.iloc[fit_idx].reset_index(drop=True), F_fit.reset_index(drop=True)]
        A_val = [X.iloc[val_idx].reset_index(drop=True), F_val.reset_index(drop=True)]
        if v['EX'] is not None:
            A_fit.append(v['EX'].iloc[fit_idx].reset_index(drop=True))
            A_val.append(v['EX'].iloc[val_idx].reset_index(drop=True))
        Mf, Mv = pd.concat(A_fit, axis=1), pd.concat(A_val, axis=1)
        kw = {}
        if v['opts'] == 'mono':
            inc = {'knnw_yr_40', 'teh_date_m10'} | {s for s in v['spec'] if s.startswith(('sum_', 'te_', 'dsum_', 'mhi_'))}
            kw = dict(monotone_constraints=[1 if c in inc else 0 for c in Mf.columns], monotone_constraints_method='advanced')
        pv = 0
        for s in range(NSEEDS):
            mdl = LGBMClassifier(**P, **kw, random_state=k + 100 * s, n_jobs=int(os.environ.get("NJ", "1")), verbose=-1).fit(Mf, y[fit_idx])
            pv = pv + mdl.predict_proba(Mv)[:, 1] / NSEEDS
        oof[v['name']][r, val_idx[elig]] = pv[elig]
    if k % 5 == 4:
        print(f'fold {k} {time.time()-t0:.0f}s', flush=True)
os.makedirs(WORK + '/r2_eda/cache', exist_ok=True)
A = {}
for v in VARS:
    nm = v['name']
    A[nm] = np.array([roc_auc_score(y[~np.isnan(oof[nm][r])], oof[nm][r][~np.isnan(oof[nm][r])]) for r in range(NREP)])
    np.savez(WORK + f'/r2_eda/cache/p2_{SEED0}_{nm}.npz', oof=oof[nm], aucs=A[nm])
b = VARS[0]['name']
for v in VARS:
    nm = v['name']
    d = A[nm] - A[b]
    print(f'{nm:14s} AUC {A[nm].mean():.5f} per-rep {np.round(A[nm], 4)}  diff vs {b} {d.mean():+.5f} pos {(d > 0).sum()}/{NREP} {np.round(d, 4)}', flush=True)

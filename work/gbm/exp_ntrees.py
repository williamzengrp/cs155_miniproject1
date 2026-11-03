"""Robustness of the tree count (is the final full-data refit well defined without early stopping?).

usage: python -I exp_ntrees.py <model> <base|rec> [scheme]
Fits 1.5x the trees once per fold of the standard 5x5 splits and scores truncated ensembles
(0.5x .. 1.5x of the chosen n_estimators / iterations). Flat curve around 1.0x => a full-data refit
(25% more rows) does not need re-tuning of the tree count.
"""
import sys
sys.path.insert(0, './work/gbm')
import numpy as np
from sklearn.metrics import roc_auc_score
import lib as L

name, which = sys.argv[1], sys.argv[2]
scheme = sys.argv[3] if len(sys.argv) > 3 else 'none'
P = dict(L.BASE[name] if which == 'base' else L.M.load_params()[name]['params'])
key = 'iterations' if name == 'cat' else 'n_estimators'
n0 = int(P[key])
fr = [0.5, 0.67, 0.8, 1.0, 1.25, 1.5]
ns = [max(1, int(round(n0 * f))) for f in fr]
P[key] = ns[-1]
A, _ = L.DATA[name]
splits = L.H.make_splits(L.tr, L.y, n_rep=5)
oof = np.full((len(ns), 5, len(L.y)), np.nan)
for k, (r, fit_idx, val_idx, elig) in enumerate(splits):
    rows, lab, w = L.fit_rows(fit_idx, scheme)
    m = L.fit_one(name, P, rows, lab, w, seed=k)
    ve = val_idx[elig]
    for i, n in enumerate(ns):
        if name == 'lgbm':
            p = m.predict_proba(A.iloc[ve], num_iteration=n)[:, 1]
        elif name == 'xgb':
            p = m.predict_proba(A.iloc[ve], iteration_range=(0, n))[:, 1]
        else:
            p = m.predict_proba(A.iloc[ve], ntree_end=n)[:, 1]
        oof[i, r, ve] = p
au = np.array([[roc_auc_score(L.y[~np.isnan(oof[i, r])], oof[i, r][~np.isnan(oof[i, r])]) for r in range(5)]
               for i in range(len(ns))])
ref = au[fr.index(1.0)]
L.log(f'\n### tree-count robustness: {name} ({which} params, {key}={n0}, scheme {scheme}, feats {L.M.FEATS})')
for i, n in enumerate(ns):
    d = au[i] - ref
    L.log(f'- {name} x{fr[i]:<4} ({n:5d}) AUC {au[i].mean():.5f} vs x1.0 {d.mean():+.5f} (pos {int((d > 0).sum())}/5)')

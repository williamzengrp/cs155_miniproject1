"""Items 3+4: final-fit strategy and seed bagging, simulated honestly inside each outer CV fold.

usage: python -I exp_finalfit.py <model> <params: base|rec (= best_params_newcv.json)> <scheme> <n_rep_outer> <n_seeds> <n_inner_rep>

For every outer split (standard H.make_splits seeds; first n_rep_outer repeats) on the outer fit part:
  full_s{j}: refit on the WHOLE outer fit part with seed k+1000*j (j < n_seeds)  -> (b) / (c)
  inner{j}:  models from StratifiedGroupKFold(5, by uri) on the outer fit part (teammates' scheme,
             n_inner_rep repeats x 5 folds; seed = k + 100*(fold+5*rep) + 50)  -> (a)
All scored on the outer validation rows (eligible only). Raw predictions saved to gbm/ff_<model>_<tag>.npz.
"""
import sys
import os
import time
sys.path.insert(0, './work/gbm')
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import lib as L

name, which, scheme = sys.argv[1], sys.argv[2], sys.argv[3]
n_rep, n_seeds, n_inner_rep = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
params = L.BASE[name] if which == 'base' else L.M.load_params()[name]['params']
tag = f'{which}_{scheme}_{L.M.FEATS}'
A, At = L.DATA[name]
splits = L.H.make_splits(L.tr, L.y, n_rep=n_rep)
n = len(L.y)
FULL = np.full((n_seeds, n_rep, n), np.nan)             # full refit, per seed
INNER = np.full((5 * n_inner_rep, n_rep, n), np.nan)    # inner-fold models
TFULL = np.zeros((n_seeds, len(At)))                     # test preds of fold models (for test_foldavg)
groups = L.URI
t0 = time.time()
for k, (r, fit_idx, val_idx, elig) in enumerate(splits):
    ve = val_idx[elig]
    rows, lab, w = L.fit_rows(fit_idx, scheme)
    for j in range(n_seeds):
        m = L.fit_one(name, params, rows, lab, w, seed=k + 1000 * j)
        FULL[j, r, ve] = m.predict_proba(A.iloc[ve])[:, 1]
        TFULL[j] += m.predict_proba(At)[:, 1] / len(splits)
    for ir in range(n_inner_rep):
        sgk = StratifiedGroupKFold(5, shuffle=True, random_state=k + 7919 * ir)
        for f, (itr, _) in enumerate(sgk.split(fit_idx, L.y[fit_idx], groups[fit_idx])):
            sub = fit_idx[itr]
            rows, lab, w = L.fit_rows(sub, scheme)
            m = L.fit_one(name, params, rows, lab, w, seed=k + 100 * (f + 5 * ir) + 50)
            INNER[f + 5 * ir, r, ve] = m.predict_proba(A.iloc[ve])[:, 1]
    print(f'split {k} done {time.time() - t0:.0f}s', flush=True)

np.savez(os.path.join(L.M.GBM, f'ff_{name}_{tag}.npz'), FULL=FULL, INNER=INNER, TFULL=TFULL)

from sklearn.metrics import roc_auc_score


def aucs_of(P):          # P: (n_rep, n) with NaN outside eligible
    return np.array([roc_auc_score(L.y[~np.isnan(P[r])], P[r][~np.isnan(P[r])]) for r in range(n_rep)])


V = {'full_s1': FULL[0]}
for s in (3, 5, 10):
    if n_seeds >= s:
        V[f'full_s{s}'] = FULL[:s].mean(0)
V['inner5'] = INNER[:5].mean(0)
if n_inner_rep >= 3:
    V['inner15'] = INNER[:15].mean(0)
if n_seeds >= 5:
    V['full_s5+inner5'] = 0.5 * FULL[:5].mean(0) + 0.5 * INNER[:5].mean(0)
ref = aucs_of(V['full_s1'])
L.log(f'\n### final-fit / seed bagging: {name} params={which} scheme={scheme} feats={L.M.FEATS} '
      f'({n_rep} outer reps, {n_seeds} seeds, {n_inner_rep}x5 inner)')
for kname, P in V.items():
    a = aucs_of(P)
    d = a - ref
    L.log(f'- {name} {kname:15s} AUC {a.mean():.5f} per-rep {np.round(a, 4)}  vs full_s1 {d.mean():+.5f} '
          f'(pos {int((d > 0).sum())}/{len(d)})')
# spread of single-seed AUCs (how much the seed alone moves the score)
single = np.array([aucs_of(FULL[j]).mean() for j in range(n_seeds)])
L.log(f'- {name} single-seed mean AUCs over seeds: {np.round(single, 5)} (sd {single.std(ddof=1) if n_seeds > 1 else 0:.5f})')

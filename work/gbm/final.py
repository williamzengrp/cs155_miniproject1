"""Deliverable: H.save('gbm_<model>', res, test_full) for each model with its recommended setting.

usage: python -I final.py <model> [<model> ...]
res       = standard 5x5 test-mimicking CV where each fold uses the recommended final-fit strategy
            (refit on the whole fit part, n_seeds seeds k+1000*s averaged, recommended weighting), so the
            OOF matches how test_full is produced. Re-uses gbm/ff_<model>_<tag>.npz (exp_finalfit.py, same
            seeds) when it covers all 5 repeats and >= n_seeds seeds.
test_full = the same strategy on ALL training rows: n_seeds refits (seeds 0..n-1), mean probability.
"""
import sys
import os
sys.path.insert(0, './work/gbm')
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import lib as L

CFG = L.M.load_params()
splits = L.H.make_splits(L.tr, L.y, n_rep=5)
for name in sys.argv[1:]:
    c = CFG[name]
    params, scheme, S = c['params'], c['weighting'], int(c['n_seeds'])
    ff = os.path.join(L.M.GBM, f'ff_{name}_{c.get("ff_tag", "rec_" + scheme) + "_" + L.M.FEATS}.npz')
    res = None
    if os.path.exists(ff):
        d = np.load(ff)
        if d['FULL'].shape[0] >= S and d['FULL'].shape[1] == 5:
            oof = d['FULL'][:S].mean(0)
            aucs = np.array([roc_auc_score(L.y[~np.isnan(o)], o[~np.isnan(o)]) for o in oof])
            res = dict(oof=oof, aucs=aucs, auc=float(aucs.mean()), test_foldavg=d['TFULL'][:S].mean(0))
            print(f'reused {os.path.basename(ff)}: AUC {res["auc"]:.5f} {np.round(aucs, 4)}')
    if res is None:
        res = L.run(name, params, splits, scheme=scheme, seeds=tuple(range(S)), test=True,
                    tag=f'gbm_{name} ({S} seeds, {scheme})')
    rows, lab, w = L.fit_rows(np.arange(len(L.y)), scheme)
    At = L.DATA[name][1]
    test_full = np.mean([L.fit_one(name, params, rows, lab, w, seed=s).predict_proba(At)[:, 1]
                         for s in range(S)], axis=0)
    L.H.save(f'gbm_{name}', res, test_full)
    rho = spearmanr(test_full, res['test_foldavg']).statistic
    L.log(f'- saved results/gbm_{name}.npz: 5x5 AUC {res["auc"]:.5f} {np.round(res["aucs"], 4)}; '
          f'test_full = {S}-seed full-train refit, weighting {scheme}, feats {L.M.FEATS} '
          f'(Spearman vs 25-fold-model test avg {rho:.4f})')

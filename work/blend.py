"""Blend saved results (work/results/*.npz) on the shared test-mimicking CV.

Usage: python -I blend.py name1 name2 ...   (names without .npz)
Per repeat: rank-transform each model's OOF over that repeat's eligible rows, then
  - equal-weight rank average
  - nested optimised weights (fit non-negative weights on 4/5 of the rows, score on 1/5)
"""
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, './work')
import harness as H


def fit_weights(O, y):
    k = O.shape[1]

    def neg_auc(w):
        w = np.abs(w) / np.abs(w).sum()
        return -roc_auc_score(y, O @ w)

    starts = [np.ones(k) / k] + [np.eye(k)[i] * 0.5 + 0.5 / k for i in range(k)]
    best = min((minimize(neg_auc, s, method='Nelder-Mead', options=dict(maxiter=1500, xatol=1e-4, fatol=1e-6))
                for s in starts), key=lambda r: r.fun)
    w = np.abs(best.x)
    return w / w.sum()


def rank_matrix(results, r, mask):
    return np.column_stack([rankdata(res['oof'][r, mask]) / mask.sum() for res in results])


def evaluate(names, y, weights=None, nested=True, verbose=True):
    results = [H.load_res(n) for n in names]
    R = results[0]['oof'].shape[0]
    out = {'single': {n: res['aucs'].mean() for n, res in zip(names, results)}}
    eq, nst, fixed = [], [], []
    for r in range(R):
        m = ~np.isnan(results[0]['oof'][r])
        for res in results:
            assert np.array_equal(m, ~np.isnan(res['oof'][r])), 'results come from different splits'
        O = rank_matrix(results, r, m)
        yr = y[m]
        eq.append(roc_auc_score(yr, O.mean(axis=1)))
        if weights is not None:
            fixed.append(roc_auc_score(yr, O @ np.asarray(weights)))
        if nested and len(names) > 1:
            p = np.zeros(len(yr))
            for a, b in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(O, yr):
                p[b] = O[b] @ fit_weights(O[a], yr[a])
            nst.append(roc_auc_score(yr, p))
    out['equal'] = np.array(eq)
    if fixed:
        out['fixed'] = np.array(fixed)
    if nst:
        out['nested'] = np.array(nst)
    if verbose:
        for n, a in out['single'].items():
            print(f'  {n:28s} {a:.5f}')
        for k in ('equal', 'fixed', 'nested'):
            if k in out:
                print(f'  [{k:6s} blend] {out[k].mean():.5f}  per-rep {np.round(out[k], 4)}')
    return out


if __name__ == '__main__':
    tr, te, y = H.load()
    evaluate(sys.argv[1:], y)

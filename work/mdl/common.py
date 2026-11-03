"""Shared helpers for the `mdl` agent (non-tree model families).

- data(): train/test frames, labels, the teammates' 67 label-free features (FEATURE_GROUPS).
- prep(): per-fold preprocessing (median fill + quantile-normal / standard scaling), fit on fit rows only.
- blend_report(): marginal value of a model in a per-repeat rank-average blend with the 5 baseline trees.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer, StandardScaler

WORK = './work'
MDL = os.path.join(WORK, 'mdl')
sys.path.insert(0, WORK)
import harness as H                                   # noqa: E402
from time_features import build_features, FEATURE_GROUPS   # noqa: E402

BASE = ['lgbm', 'xgb', 'cat', 'et', 'rf']


def data():
    tr, te, y = H.load()
    X, Xt = build_features(tr, te, FEATURE_GROUPS)
    return tr, te, y, X, Xt


def prep(Xfit, Xother_list, scaler='quantile', seed=0):
    """Median-fill + scale, fitted on Xfit only. Returns (A_fit, [A_other...]) as float arrays."""
    med = Xfit.median()
    A = Xfit.fillna(med).to_numpy(float)
    others = [Xo.fillna(med).to_numpy(float) for Xo in Xother_list]
    if scaler == 'quantile':
        sc = QuantileTransformer(n_quantiles=min(1000, len(A)), output_distribution='normal',
                                 subsample=10**9, random_state=seed)
    elif scaler == 'standard':
        sc = StandardScaler()
    elif scaler == 'none':
        return A, others
    else:
        raise ValueError(scaler)
    A = sc.fit_transform(A)
    others = [sc.transform(o) for o in others]
    return A, others


def load_base():
    return {n: H.load_res('base_' + n) for n in BASE}


def _rank01(v):
    return (rankdata(v) - 0.5) / len(v)


def blend_aucs(y, oofs, weights):
    """Per-repeat AUC of a weighted rank-average of OOF arrays (R x n, NaN = not scored)."""
    R = oofs[0].shape[0]
    out = []
    for r in range(R):
        m = ~np.isnan(oofs[0][r])
        s = sum(w * _rank01(o[r, m]) for o, w in zip(oofs, weights))
        out.append(roc_auc_score(y[m], s))
    return np.array(out)


def blend_report(y, new_oof, base=None, label='new', ws=(None, 0.1, 0.2, 0.3), verbose=True):
    """Compare the 5-tree rank-average with / without `new_oof` (same splits). w=None -> equal weights.
    Returns dict w -> (mean diff, per-repeat diffs)."""
    base = base or load_base()
    b = [base[n]['oof'] for n in BASE]
    for o in b:
        assert np.array_equal(np.isnan(o), np.isnan(new_oof)), 'eligible masks differ: different splits?'
    a0 = blend_aucs(y, b, [1 / len(b)] * len(b))
    out = {'base_blend': a0}
    for w in ws:
        if w is None:
            wts = [1 / (len(b) + 1)] * (len(b) + 1)
            key = 'eq'
        else:
            wts = [(1 - w) / len(b)] * len(b) + [w]
            key = f'w{w}'
        a1 = blend_aucs(y, b + [new_oof], wts)
        d = a1 - a0
        out[key] = (float(d.mean()), d)
        if verbose:
            print(f'  blend[{label}] {key:5s}: {a1.mean():.5f} vs base {a0.mean():.5f}  diff {d.mean():+.5f}'
                  f'  per-rep {np.round(d, 4)}  pos {int((d > 0).sum())}/{len(d)}', flush=True)
    return out


def standalone(y, oof):
    R = oof.shape[0]
    a = []
    for r in range(R):
        m = ~np.isnan(oof[r])
        a.append(roc_auc_score(y[m], oof[r, m]))
    return np.array(a)


class Timer:
    def __init__(self):
        self.t = time.time()

    def __call__(self):
        return time.time() - self.t


def tune_gain(y, oof, ws=(None, 0.1, 0.2)):
    """Blend gain on the TUNING splits (n_rep=2, seed0=7000) vs the 5 trees cached by base_tune.py.
    Returns a short string; '' if the cache is missing."""
    fs = [os.path.join(MDL, 'cache', f'tune_base_{n}.npz') for n in BASE]
    if not all(os.path.exists(f) for f in fs):
        return ''
    b = [np.load(f)['oof'] for f in fs]
    a0 = blend_aucs(y, b, [1 / len(b)] * len(b))
    out = []
    for w in ws:
        wts = [1 / (len(b) + 1)] * (len(b) + 1) if w is None else [(1 - w) / len(b)] * len(b) + [w]
        d = blend_aucs(y, b + [oof], wts) - a0
        out.append(f'{"eq" if w is None else w}:{d.mean():+.4f}({",".join(f"{v:+.4f}" for v in d)})')
    return 'tune-blend-gain ' + ' '.join(out)


# redundant date encodings / seasonality extras (dropped by the 'compact'/'core' recipes)
DATE_REDUNDANT = ['release_year', 'days_since_release', 't_log_age', 't_decade', 's_doy', 's_week', 's_quarter',
                  's_mon_sin', 's_mon_cos', 's_dow_sin', 's_dow_cos', 's_doy_sin', 's_doy_cos', 's_is_friday',
                  's_is_tuesday', 's_holiday', 's_summer']


def data_v2():
    """fe agent's confirmed v2 set (55 cols): trend, seasonality, density, albumnn2, detrended."""
    sys.path.insert(0, os.path.join(WORK, 'fe'))
    from features_v2 import build_features_v2
    tr, te, y = H.load()
    X, Xt = build_features_v2(tr, te)
    assert X.shape[1] == 55
    return X, Xt

"""Distribution-matched apply-time label features (agent `shift`).

Problem (train/apply shift): label features of the FIT rows are out-of-fold over n_inner=5 inner folds, i.e. built
from 80% of the fit labels, while validation / test rows get features built from 100% of the fit labels (cnt_date
mean 3.40 fit-OOF vs 4.16 CV-val vs 4.98 test). The model is trained on one distribution and applied on another.

Fix: for every inner fold j (same StratifiedGroupKFold(5, groups=uri, random_state=seed) as te/label_features.py),
encode the apply rows from exactly the source set used for inner fold j's fit rows (fit_idx[a_j]); predict with
each of the 5 versions and average. Apply-time features then have exactly the fit-time distribution.
Leakage: apply features read labels of fit_idx[a_j] (subset of fit_idx) only; validation labels are never read.
F_fit is identical to te/label_features.label_features (same seed, same inner split)."""
import os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
W = './work'
sys.path.insert(0, W); sys.path.insert(0, W + '/te')
from label_features import get_ctx, encode, BEST_SPEC


def label_features_matched(XA, y, fit_idx, apply_idx_list, tr, te, seed=0, spec=BEST_SPEC, n_inner=5,
                           n_draw_seeds=1):
    """Returns (F_fit, F_full_list, F_inner_list):
    F_fit       : fit-row OOF features (identical to label_features)
    F_full_list : per apply array, features from all fit labels (identical to label_features)
    F_inner_list: per apply array, a list of n_inner frames, frame j built from inner-fold-j sources fit_idx[a_j].
    n_draw_seeds > 1: extra apply-time draws from the inner splits of seeds seed+1000*d (d=1..n_draw_seeds-1),
    same fold size (fit-time distribution), more draws to average (still only fit_idx labels)."""
    ctx = get_ctx(tr, te)
    assert len(XA) == ctx.N and len(y) == ctx.n
    fit_idx = np.asarray(fit_idx)
    vals = np.zeros(ctx.N)
    vals[:ctx.n] = y
    F_fit = None
    F_inner = [[] for _ in apply_idx_list]
    sgkf = StratifiedGroupKFold(n_inner, shuffle=True, random_state=seed)
    for a, b in sgkf.split(fit_idx, y[fit_idx], ctx.uri[fit_idx]):
        wts = np.zeros(ctx.N)
        wts[fit_idx[a]] = 1.0
        e = encode(ctx, vals, wts, fit_idx[b], spec)
        if F_fit is None:
            F_fit = pd.DataFrame(np.nan, index=range(len(fit_idx)), columns=e.columns)
        F_fit.iloc[b] = e.to_numpy()
        for i, ai in enumerate(apply_idx_list):
            F_inner[i].append(encode(ctx, vals, wts, np.asarray(ai), spec))
    for d in range(1, n_draw_seeds):
        sg2 = StratifiedGroupKFold(n_inner, shuffle=True, random_state=seed + 1000 * d)
        for a, b in sg2.split(fit_idx, y[fit_idx], ctx.uri[fit_idx]):
            wts = np.zeros(ctx.N)
            wts[fit_idx[a]] = 1.0
            for i, ai in enumerate(apply_idx_list):
                F_inner[i].append(encode(ctx, vals, wts, np.asarray(ai), spec))
    wts = np.zeros(ctx.N)
    wts[fit_idx] = 1.0
    F_full = [encode(ctx, vals, wts, np.asarray(a), spec) for a in apply_idx_list]
    return F_fit, F_full, F_inner

"""r2_eda: extended label features = te agent's label_features (same Ctx, same inner StratifiedGroupKFold grouped by
uri, same leakage rules) plus a few extra encodings suggested by the error analysis:
  sum_date      number of High labelled date-mates (source rows sharing the release date, own uri excluded)
  neg_date      number of Low labelled date-mates
  te_date_raw   unsmoothed date-mate High rate (NaN when no labelled date-mates)
  dsum_date     number of distinct High date-mate SONGS (copies of one uri counted once)
  mhi_date      number of High date-mate rows that belong to multi-copy songs (pool count >= 2)
Only rows with wts > 0 (fit rows / other inner folds) are ever read as label sources (see te/label_features.py).
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
WORK = './work'
sys.path.insert(0, WORK + '/te')
import label_features as LF

_UCNT = {}


def _ucnt(ctx):
    if id(ctx) not in _UCNT:
        _UCNT[id(ctx)] = np.bincount(ctx.uri)[ctx.uri]
    return _UCNT[id(ctx)]


def encode_ext(ctx, vals, wts, dst, spec):
    base = [s for s in spec if s.split('_')[0] in ('cnt', 'te', 'teh', 'albsize', 'nbd', 'knn', 'knnw')]
    out = LF.encode(ctx, vals, wts, dst, base) if base else pd.DataFrame(index=range(len(dst)))
    for name in spec:
        if name in base:
            continue
        if name in ('sum_date', 'neg_date', 'te_date_raw'):
            C, S = LF._key_stats(ctx, 'date', vals, wts, dst)
            v = {'sum_date': S, 'neg_date': C - S, 'te_date_raw': np.where(C > 0, S / np.maximum(C, 1e-12), np.nan)}[name]
        elif name == 'dsum_date':
            # count each High source SONG once: weight each source row by 1/(#source copies of its uri)
            k = ctx.uri
            cpy = np.bincount(k, weights=(wts > 0).astype(float), minlength=k.max() + 1)[k]
            w1 = np.where(wts > 0, 1.0 / np.maximum(cpy, 1), 0.0)
            v = LF._key_stats(ctx, 'date', vals, w1, dst)[1]
        elif name == 'mhi_date':
            w2 = wts * (_ucnt(ctx) >= 2)
            v = LF._key_stats(ctx, 'date', vals, w2, dst)[1]
        else:
            raise ValueError(name)
        out[name] = np.asarray(v, float)
    return out[list(spec)]


def label_features_ext(XA, y, fit_idx, apply_idx_list, tr, te, seed=0, spec=LF.BEST_SPEC, n_inner=5):
    """Same contract as te/label_features.label_features (fit rows out-of-fold via inner folds grouped by uri)."""
    ctx = LF.get_ctx(tr, te)
    assert len(XA) == ctx.N and len(y) == ctx.n
    fit_idx = np.asarray(fit_idx)
    vals = np.zeros(ctx.N)
    vals[:ctx.n] = y
    F_fit = None
    sgkf = StratifiedGroupKFold(n_inner, shuffle=True, random_state=seed)
    for a, b in sgkf.split(fit_idx, y[fit_idx], ctx.uri[fit_idx]):
        wts = np.zeros(ctx.N)
        wts[fit_idx[a]] = 1.0
        e = encode_ext(ctx, vals, wts, fit_idx[b], spec)
        if F_fit is None:
            F_fit = pd.DataFrame(np.nan, index=range(len(fit_idx)), columns=e.columns)
        F_fit.iloc[b] = e.to_numpy()
    wts = np.zeros(ctx.N)
    wts[fit_idx] = 1.0
    return F_fit, [encode_ext(ctx, vals, wts, np.asarray(a), spec) for a in apply_idx_list]

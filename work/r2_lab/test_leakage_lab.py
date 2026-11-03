"""Leakage checks for r2_lab/labx.py (all feature kinds, dedup on/off, n_inner 5/10)."""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/main'); sys.path.insert(0, WORK + '/r2_lab')
import harness as H
from features_final import build_features_final
import labx as LX

tr, te, y = H.load()
X, Xt = build_features_final(tr, te)
XA = pd.concat([X, Xt], ignore_index=True)
n, m = len(tr), len(te)
TEST = np.arange(n, n + m)
ctx = LX.get_ctx(tr, te)
SPEC = ['knnw_yr_40', 'teh_date_m10', 'cnt_date', 'knnw_yr_10', 'knnw_yr_150', 'knnw_yr05_40', 'knnw_yr3_40',
        'knnw_yrA_40', 'knnw_yrB_40', 'knn_glob_20', 'knnwx_yr_40', 'knnwx_yr3_40', 'dmin_hi', 'dmin_lo', 'tek_date_m10',
        'chain_year-ym-date-alb_m10', 'teh_yc8_m10', 'teh_alb_m10', 'cnt_alb', 'nbd_s1_m1', 'te_ym_m10']
splits = H.make_splits(tr, y, n_rep=1)
_, a, b, e = splits[3]
vals = np.zeros(ctx.N); vals[:n] = y

# 1. inner folds: never split a uri; fit-row features invariant to own-fold + validation labels
for dedup in (False, True):
    for ia, ib in StratifiedGroupKFold(5, shuffle=True, random_state=3).split(a, y[a], ctx.uri[a]):
        assert not set(ctx.uri[a[ia]]) & set(ctx.uri[a[ib]]), 'same uri split across inner folds'
        v1, w1 = LX.sources(ctx, vals, a[ia], dedup)
        f1 = LX.encode(ctx, v1, w1, a[ib], SPEC).to_numpy()
        v2 = vals.copy(); v2[a[ib]] = 1 - v2[a[ib]]; v2[b] = 1 - v2[b]
        v2s, w2 = LX.sources(ctx, v2, a[ia], dedup)
        f2 = LX.encode(ctx, v2s, w2, a[ib], SPEC).to_numpy()
        assert np.array_equal(w1, w2) and np.abs(f1 - f2).max() == 0, 'own / val labels leak into fit-row features'

# 2. a source row's OWN label never enters its own features: swap the labels of two singleton source rows
#    (one High, one Low -> prior unchanged) that are > 3.5 y apart, in different date/alb/yc8 groups and not in
#    each other's global neighbour lists; their own features must not change.
wts_src = np.zeros(ctx.N); wts_src[a] = 1
cnt_u = np.bincount(ctx.uri[a], minlength=ctx.uri.max() + 1)
single = a[cnt_u[ctx.uri[a]] == 1]
rng = np.random.default_rng(0)
hi = rng.permutation(single[y[single] == 1])
lo = rng.permutation(single[y[single] == 0])
gl = ctx.nbr['glob'][0]
checked = 0
for i in hi[:400]:
    cand = lo[(np.abs(ctx.t[lo] - ctx.t[i]) > 3.5) & (ctx.key['yc8'][lo] != ctx.key['yc8'][i])]
    cand = [j for j in cand[:50] if j not in gl[i] and i not in gl[j]]
    if not cand:
        continue
    j = cand[0]
    for dedup in (False, True):
        v1, w1 = LX.sources(ctx, vals, a, dedup)
        v2 = vals.copy(); v2[i], v2[j] = 0, 1
        v2s, w2 = LX.sources(ctx, v2, a, dedup)
        f1 = LX.encode(ctx, v1, w1, np.array([i, j]), SPEC).to_numpy()
        f2 = LX.encode(ctx, v2s, w2, np.array([i, j]), SPEC).to_numpy()
        assert np.abs(f1 - f2).max() < 1e-12, f'own label leaks for rows {i},{j}'
    checked += 1
    if checked >= 60:
        break
assert checked >= 30
# 3. neighbour / pair lists never contain the row itself or a same-uri copy
for kind, (ind, dist) in ctx.nbr.items():
    assert not (ctx.uri[ind] == ctx.uri[:, None])[np.isfinite(dist)].any(), kind
assert not (ctx.uri[ctx.pi] == ctx.uri[ctx.pj]).any()
# 4. public API: validation labels never matter (all options); test rows have no labels to read
for dedup, ni in ((False, 5), (True, 5), (False, 10), (True, 10)):
    y2 = y.copy(); y2[b] = 1 - y2[b]
    F1 = LX.label_features(XA, y, a, [b, TEST], tr, te, seed=0, spec=SPEC, n_inner=ni, dedup=dedup)
    F2 = LX.label_features(XA, y2, a, [b, TEST], tr, te, seed=0, spec=SPEC, n_inner=ni, dedup=dedup)
    assert all(np.abs(p.to_numpy() - q.to_numpy()).max() == 0 for p, q in zip([F1[0]] + F1[1], [F2[0]] + F2[1]))
    assert not F1[0].isna().any().any()
print(f'all leakage checks passed ({checked} swapped singleton pairs, {len(SPEC)} feature kinds, dedup on/off, n_inner 5/10)')

"""Leakage checks for label_features.encode / label_features (agent te)."""
import sys; sys.path.insert(0, './work/te')
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from common import LF, XA, y, tr, te, n, m, splits, TEST

ctx = LF.get_ctx(tr, te)
spec = ['te_date_m3', 'cnt_date', 'teh_date_m3', 'te_ym_m10', 'te_year_m10', 'nbd_s1_m1',
        'knn_glob_20', 'knn_yr_20', 'knnw_yr_20', 'teh_alb_m10', 'te_alb_m3', 'cnt_alb', 'knnw_yr_80']
_, a, b, e = splits[3]
vals = np.zeros(ctx.N); vals[:n] = y
for ia, ib in StratifiedGroupKFold(5, shuffle=True, random_state=3).split(a, y[a], ctx.uri[a]):
    assert not set(ctx.uri[a[ia]]) & set(ctx.uri[a[ib]]), 'same uri split across inner folds'
    wts = np.zeros(ctx.N); wts[a[ia]] = 1
    f1 = LF.encode(ctx, vals, wts, a[ib], spec).to_numpy()
    v2 = vals.copy(); v2[a[ib]] = 1 - v2[a[ib]]          # flip own-fold labels
    v2[b] = 1 - v2[b]                                    # and validation labels
    f2 = LF.encode(ctx, v2, wts, a[ib], spec).to_numpy()
    assert np.abs(f1 - f2).max() == 0, 'own / val labels leak into fit-row features'
# apply rows overlapping the source set: own label must still be excluded
wts = np.zeros(ctx.N); wts[a] = 1
sub = np.concatenate([a[y[a] == 1][:150], a[y[a] == 0][:150]])   # balanced flip keeps the prior fixed
v2 = vals.copy(); v2[sub] = 1 - v2[sub]
# features of row i may change only through OTHER flipped rows; check a row whose date has no other flipped row
k = ctx.key['date']
lonely = [j for j in range(300) if np.sum(k[sub] == k[sub[j]]) == 1]
f2 = LF.encode(ctx, v2, wts, sub, ['te_date_m3', 'cnt_date', 'nbd_s1_m1', 'te_alb_m3', 'cnt_alb']).to_numpy()
f1b = LF.encode(ctx, vals, wts, sub, ['te_date_m3', 'cnt_date', 'nbd_s1_m1', 'te_alb_m3', 'cnt_alb']).to_numpy()
assert np.abs(f1b[lonely] - f2[lonely]).max() == 0, 'own label used for date features'
# kNN lists never contain the row itself or a same-uri copy
for kind in ('glob', 'yr'):
    ind, dist = ctx.nbr[kind]
    assert not (ctx.uri[ind] == ctx.uri[:, None])[np.isfinite(dist)].any()
assert not (ctx.uri[ctx.pi] == ctx.uri[ctx.pj]).any()
# validation-label invariance through the public API
y2 = y.copy(); y2[b] = 1 - y2[b]
F1 = LF.label_features(XA, y, a, [b, TEST], tr, te, seed=0, spec=spec)
F2 = LF.label_features(XA, y2, a, [b, TEST], tr, te, seed=0, spec=spec)
assert all(np.abs(p.to_numpy() - q.to_numpy()).max() == 0 for p, q in zip([F1[0]] + F1[1], [F2[0]] + F2[1]))
print('all leakage checks passed;', len(lonely), 'lonely rows checked')

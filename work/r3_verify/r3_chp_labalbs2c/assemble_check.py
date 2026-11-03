"""Auditor: (1) rebuild r3_chp_labalbs2c from the builder caches and compare with the saved npz;
(2) residual-correction inputs are invariant to flipping validation labels; (3) fold-0 full chain with
the auditor's re-run (orig vs flipped-val labels; and permuted fit labels)."""
import sys, os
sys.dont_write_bytecode = True
W = './work'
D = W + '/r3_verify/r3_chp_labalbs2c'
sys.path.insert(0, W + '/r3_chp')
import numpy as np
import chp_save as S
L = S.L
from sklearn.metrics import roc_auc_score
from corrlib import correction
cat, et, w, cc, ec, fc, fe = S.BLOCKS['labalbs2c']
print('block', cat, et, w, cc, ec, fc, fe)
oof, aucs, tfa, have = L.assemble(lambda k: L.blend(cat, et, k, w, cc, ec))
sv = dict(np.load(W + '/results/r3_chp_labalbs2c.npz'))
print('saved keys', list(sv), 'have', have)
print('oof NaN pattern equal:', np.array_equal(np.isnan(oof), np.isnan(sv['oof'])),
      ' max|oof diff|', float(np.nanmax(np.abs(oof - sv['oof']))))
print('aucs rebuilt', np.round(aucs, 5), ' saved', np.round(sv['aucs'], 5))
print('test_foldavg max diff', float(np.abs(tfa - sv['test_foldavg']).max()))
import chp_full as F
tf = F.blend_full(fc, fe, w, cc, ec)
print('test_full max diff', float(np.abs(tf - sv['test_full']).max()), 'len', len(sv['test_full']))
# eligibility: each repeat's OOF only on eligible val rows
for k, (r, fit, val, elig) in enumerate(L.splits):
    assert not set(fit) & set(val)
    assert len(L.G[k]['codes']) == len(fit)
    assert len(L.C[k]['cat_ip']) == len(fit)
    assert np.isnan(sv['oof'][r, val[~elig]]).all() and np.isfinite(sv['oof'][r, val[elig]]).all()
print('fit/val disjoint, geometry/ip aligned with fit, OOF only on eligible rows: OK')
# (2) correction residual refs with val labels flipped
mx = 0.
for k, (r, fit, val, elig) in enumerate(L.splits):
    yf = L.y.copy(); yf[val] = 1 - yf[val]
    g = L.G[k]
    for tag in ('cat', 'et'):
        res = yf[fit] - L.C[k][f'{tag}_ip']
        rr = np.bincount(g['codes'], weights=res, minlength=g['nref']) / np.bincount(g['codes'], minlength=g['nref'])
        mx = max(mx, float(np.abs(rr - L.RR[(k, tag)]).max()))
print('residual refs max diff after flipping val labels (all 25 folds):', mx)
# (3) fold 0 full chain from the auditor's re-runs
k = 0
r, fit, val, elig = L.splits[k]
def chain(d):
    c = d['cat_v_2000']; e = d['et_v']
    a, bw, sh = cc
    c = np.clip(c + a * correction(L.G[k]['pv'], L.RR[(k, 'cat')], bw, sh), 0, 1)
    a, bw, sh = L.CORR0['et']
    e = np.clip(e + a * correction(L.G[k]['pv'], L.RR[(k, 'et')], bw, sh), 0, 1)
    return (1 - w) * c + w * e
runs = {}
for mode in ('orig', 'flipval', 'permfit'):
    p = f'{D}/fold_{mode}_lab-alb_o0_k0.npz'
    if os.path.exists(p):
        runs[mode] = dict(np.load(p))
if 'orig' in runs and 'flipval' in runs:
    for key in ('Xf', 'Xv', 'Xt', 'cat_v_2000', 'cat_t_2000', 'et_v', 'et_t'):
        print(f'orig vs flipval {key}: identical={np.array_equal(runs["orig"][key], runs["flipval"][key])}  '
              f'max|d|={float(np.abs(runs["orig"][key] - runs["flipval"][key]).max())}')
for mode, d in runs.items():
    b = chain(d)
    raw = (1 - w) * d['cat_v_2000'] + w * d['et_v']
    print(f'{mode}: fold0 eligible-val AUC (true labels) raw blend {roc_auc_score(L.y[val][elig], raw[elig]):.5f}  '
          f'with residual corr {roc_auc_score(L.y[val][elig], b[elig]):.5f}')

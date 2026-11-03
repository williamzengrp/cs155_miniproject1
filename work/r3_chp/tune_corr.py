"""r3_chp: re-tune residual correction (alpha, bw, shrink per family) and Cat/ET mix on cached replica folds."""
import sys
sys.dont_write_bytecode = True
import os, pickle, time, itertools
import numpy as np
from sklearn.metrics import roc_auc_score
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/r3_chp')
import harness as H
from corrlib import correction
tr, te, y = H.load()
splits = H.make_splits(tr, y, n_rep=5)
G = pickle.load(open(WORK + '/r3_chp/cache/geo_std.pkl', 'rb'))
C = [dict(np.load(f'{WORK}/ch/cache/chfull_std_k{k}.npz')) for k in range(25)]


def rref(k, tag):
    g = G[k]; fit = splits[k][1]
    res = y[fit] - C[k][f'{tag}_ip']
    cnt = np.bincount(g['codes'], minlength=g['nref'])
    return np.bincount(g['codes'], weights=res, minlength=g['nref']) / cnt

RR = {(k, t): rref(k, t) for k in range(25) for t in ('cat', 'et')}


def fam_val(tag, a, bw, sh):
    """list over folds of corrected val preds"""
    return [np.clip(C[k][f'{tag}_raw_v'] + a * correction(G[k]['pv'], RR[(k, tag)], bw, sh), 0, 1) for k in range(25)]


def aucs(vals):
    oof = np.full((5, len(y)), np.nan)
    for k, (r, fit, val, elig) in enumerate(splits):
        oof[r, val[elig]] = vals[k][elig]
    return np.array([roc_auc_score(y[~np.isnan(o)], o[~np.isnan(o)]) for o in oof])

base_cat = fam_val('cat', .35, .75, 10.); base_et = fam_val('et', .15, 3., 3.)
mix = lambda cv, ev, w: [(1 - w) * a + w * b for a, b in zip(cv, ev)]
b0 = aucs(mix(base_cat, base_et, .4))
print('baseline blend', b0.mean().round(5), b0.round(4))
print('cat corr', aucs(base_cat).mean().round(5), ' et corr', aucs(base_et).mean().round(5))
rc = fam_val('cat', 0, 1, 1); re_ = fam_val('et', 0, 1, 1)
print('cat raw', aucs(rc).mean().round(5), ' et raw', aucs(re_).mean().round(5), ' blend raw', aucs(mix(rc, re_, .4)).mean().round(5))
for w in (0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6):
    a = aucs(mix(base_cat, base_et, w)); print(f'  w_et {w}: {a.mean():.5f}  d {np.round(a - b0, 4)}')
A = [0.2, 0.35, 0.5, 0.7, 1.0]; BW = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]; SH = [1., 3., 10., 30.]
for tag in ('cat', 'et'):
    rows = []
    for a, bw, sh in itertools.product(A, BW, SH):
        v = fam_val(tag, a, bw, sh)
        bl = mix(v, base_et, .4) if tag == 'cat' else mix(base_cat, v, .4)
        fa, ba = aucs(v), aucs(bl)
        rows.append((ba.mean(), fa.mean(), a, bw, sh, (ba - b0 > 0).sum()))
    rows.sort(reverse=True)
    print(f'--- {tag}: top settings (blend AUC, family AUC, alpha, bw, shrink, #reps>base)')
    for row in rows[:12]:
        print('   ', np.round(row[:2], 5), row[2:])

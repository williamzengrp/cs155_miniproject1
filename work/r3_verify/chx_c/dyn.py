"""Leakage audit of r3_chx_c (variant c: album id as CatBoost native categorical; ET = Charlie replica), std fold k.
Stage F: val-label FLIP invariance (y[val] -> 1 - y[val]) of the WHOLE variant-c pipeline (matrices, inner sets,
         cat + extra families, residual correction) at REDUCED rounds (cat 200 it lr .05, ET 100 trees): data flow is
         unchanged by the round count; val + test predictions must be bitwise identical.
Stage S: FIT-label shuffle (val labels kept true), same reduced pipeline -> val AUC on eligible rows must be ~0.5.
Stage R: reproduction of the cached main CatBoost raw val predictions (full CFG, 2 threads) + c_alb importance."""
import sys, os, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
WORK = './work'
for p in (WORK + '/r3_chx', WORK, WORK + '/ch', WORK + '/main', WORK + '/te'):
    sys.path.insert(0, p)
import harness as H
from sklearn.metrics import roc_auc_score
tr, te, y = H.load(); n, m = len(tr), len(te)
splits = H.make_splits(tr, y, n_rep=5)
from ch_common import connected_groups
from ch_feats import charlie_pool
import chx_lib
from chx_lib import ReplicaX, SEED0, dup_weights
from label_features import get_ctx
k = int(sys.argv[1]); stages = sys.argv[2]
T = 2
F, BL = charlie_pool(tr, te)
alb = get_ctx(tr, te).key['alb']
G = connected_groups(tr)
r, fit, val, elig = splits[k]
S = SEED0 + k
C = WORK + '/r3_chx/cache'
print(f'fold {k}: fit {len(fit)} val {len(val)} elig {elig.sum()}', flush=True)


def mk(yv):
    return ReplicaX(tr, te, yv, F[BL['PROJECT']], G, 'c', threads=T, alb=alb)


def md(a, b):
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def pipeline(yv):
    R = mk(yv)
    Xf, (Xv,), Xt = R.matrices(fit, [val], True, lf_seed=k)
    inner = R.inner_sets(fit, S)
    out = {}
    for fam, tag in (('cat', 'cat'), ('extra', 'et')):
        (rv,), rt, ip = R.run_family(fam, fit, Xf, [Xv], [val], Xt, inner, S)
        out[tag + '_raw_v'], out[tag + '_ip'], out[tag + '_raw_t'] = rv, ip, rt
        out[tag + '_v'] = R.correct(fam, fit, ip, rv, tr.iloc[val])
        out[tag + '_t'] = R.correct(fam, fit, ip, rt, te)
    out['bl_v'] = 0.6 * out['cat_v'] + 0.4 * out['et_v']
    out['bl_t'] = 0.6 * out['cat_t'] + 0.4 * out['et_t']
    return out, Xf, Xv, Xt


if 'R' in stages:
    t0 = time.time()
    R = mk(y)
    Xf, (Xv,), Xt = R.matrices(fit, [val], True, lf_seed=k)
    mdl = R._model('cat', S).fit(R._frame('cat', Xf, fit), y[fit], sample_weight=dup_weights(R.uri[fit], 0.0))
    rv = mdl.predict_proba(R._frame('cat', Xv, val))[:, 1]
    c = np.load(f'{C}/c_k{k}.npz')
    imp = mdl.get_feature_importance()
    names = mdl.feature_names_
    j = names.index('c_alb')
    print(f'[repro] cat raw val vs cache c_k{k}: maxdiff {md(rv, c["cat_raw_v"]):.3g}  corr {np.corrcoef(rv, c["cat_raw_v"])[0, 1]:.6f}'
          f'  AUC elig {roc_auc_score(y[val[elig]], rv[elig]):.5f} vs cache {roc_auc_score(y[val[elig]], c["cat_raw_v"][elig]):.5f}'
          f'  c_alb importance {imp[j]:.2f} (rank {int((imp > imp[j]).sum()) + 1}/{len(imp)})  ({time.time() - t0:.0f}s)', flush=True)

# reduced rounds for the data-flow tests
chx_lib.CFG['cat'].update(rounds=200, lr=0.05)
chx_lib.CFG['extra'].update(rounds=100)

if 'F' in stages:
    t0 = time.time()
    yf = y.copy(); yf[val] = 1 - y[val]
    o1, Xf1, Xv1, Xt1 = pipeline(y)
    print(f'  true-label reduced run: bl val AUC elig {roc_auc_score(y[val[elig]], o1["bl_v"][elig]):.5f} '
          f'cat {roc_auc_score(y[val[elig]], o1["cat_v"][elig]):.5f} et {roc_auc_score(y[val[elig]], o1["et_v"][elig]):.5f} '
          f'({time.time() - t0:.0f}s)', flush=True)
    o2, Xf2, Xv2, Xt2 = pipeline(yf)
    print(f'[flip] matrices maxdiff fit {md(Xf1, Xf2):.3g} val {md(Xv1, Xv2):.3g} test {md(Xt1, Xt2):.3g}', flush=True)
    for key in sorted(o1):
        print(f'[flip] {key:10s} maxdiff {md(o1[key], o2[key]):.3g}', flush=True)
    print(f'[flip] done ({time.time() - t0:.0f}s)', flush=True)

if 'S' in stages:
    t0 = time.time()
    res = []
    for sd in (12345, 777):
        ys = y.copy()
        ys[fit] = np.random.default_rng(sd).permutation(y[fit])
        o, *_ = pipeline(ys)
        a = {key: roc_auc_score(y[val[elig]], o[key][elig]) for key in ('cat_raw_v', 'cat_v', 'et_v', 'bl_v')}
        res.append(a['bl_v'])
        print(f'[shuffle seed {sd}] val AUC elig (true labels): ' + '  '.join(f'{kk} {vv:.4f}' for kk, vv in a.items())
              + f'  ({time.time() - t0:.0f}s)', flush=True)
    yv = y[val[elig]]
    print(f'[shuffle] null SE ~ {np.sqrt((len(yv) + 1) / (12 * yv.sum() * (1 - yv).sum())):.4f}', flush=True)

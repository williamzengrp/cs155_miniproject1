"""Auditor: re-run chp_var's lab-alb fold pipeline for fold K with (orig | flipval | permfit) labels.
usage: python -I -B fold_check.py MODE K VARIANT SOFF   -> OUT/fold_<MODE>_<VARIANT>_o<SOFF>_k<K>.npz"""
import sys, os, time
sys.dont_write_bytecode = True
W = './work'
OUT = W + '/r3_verify/r3_chp_labalbs2c'
mode, K, variant, soff = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
sys.path.insert(0, W + '/r3_chp')
sys.argv = ['chp_var.py', 'noop', variant]
t0 = time.time()
import chp_var as V
import numpy as np
from sklearn.metrics import roc_auc_score
splits = V.H.make_splits(V.tr, V.y, n_rep=5)
r, fit, val, elig = splits[K]
y0 = V.y.copy()
yy = y0.copy()
if mode == 'flipval':
    yy[val] = 1 - yy[val]
elif mode == 'permfit':
    rng = np.random.default_rng(123)
    yy[fit] = rng.permutation(yy[fit])
else:
    assert mode == 'orig'
V.y = yy
V.R.y = yy
print('setup', round(time.time() - t0, 1), 'toks', V.toks, 'REFIT_ET', V.REFIT_ET, 'ITERS', V.ITERS, 'THREADS', V.THREADS, flush=True)
mats, al = V.matrices(fit, [val], 42 + (soff if 'tev' in V.toks else 0), K + soff)
(Xf, (Xv,), Xt), (af, (av,), at) = mats, al
out = V.outer(fit, mats, al, V.SEED0 + K + soff)
np.savez(f'{OUT}/fold_{mode}_{variant}_o{soff}_k{K}.npz', Xf=Xf, Xv=Xv, Xt=Xt, **out)
for key in ('cat_v_2000', 'et_v'):
    print(mode, key, 'val AUC (eligible, TRUE labels)', round(roc_auc_score(y0[val][elig], out[key][elig]), 5), flush=True)
c = dict(np.load(f'{W}/r3_chp/cache/var_{variant}_o{soff}_k{K}.npz'))
for key in ('cat_v_2000', 'cat_t_2000', 'et_v', 'et_t'):
    print(mode, key, 'max|diff| vs builder cache', float(np.abs(out[key] - c[key]).max()), flush=True)
print('done', round(time.time() - t0, 1), flush=True)

"""Paired check of r3_chx variant c vs the replica on the confirmation split set (n_rep=3, seed0=9000; replica cache
work/ch/cache/chfull_conf_k*.npz).  python -I conf_eval.py VARIANT"""
import sys, os
sys.dont_write_bytecode = True
import numpy as np
WORK = './work'
sys.path.insert(0, WORK)
import harness as H
from sklearn.metrics import roc_auc_score
tr, te, y = H.load(); n = len(tr)
V = sys.argv[1]
cs = H.make_splits(tr, y, n_rep=3, seed0=9000)
ks = [k for k in range(len(cs)) if os.path.exists(f'{WORK}/r3_chx/cache/conf_{V}_k{k}.npz')
      and os.path.exists(f'{WORK}/ch/cache/chfull_conf_k{k}.npz')]
for key in ('bl', 'cat', 'et'):
    fd = []
    oa, ob = np.full(n, np.nan), np.full(n, np.nan)
    for k in ks:
        _, fit, val, elig = cs[k]
        a, b = np.load(f'{WORK}/r3_chx/cache/conf_{V}_k{k}.npz'), np.load(f'{WORK}/ch/cache/chfull_conf_k{k}.npz')
        pa = 0.6 * a['cat_v'] + 0.4 * a['et_v'] if key == 'bl' else a[f'{key}_v']
        pb = 0.6 * b['cat_v'] + 0.4 * b['et_v'] if key == 'bl' else b[f'{key}_v']
        fd.append(roc_auc_score(y[val[elig]], pa[elig]) - roc_auc_score(y[val[elig]], pb[elig]))
        if k < 5:
            oa[val[elig]], ob[val[elig]] = pa[elig], pb[elig]
    line = f'{key}: folds {ks} per-fold diff mean {np.mean(fd):+.5f} ({np.sum(np.array(fd) > 0)}/{len(fd)} > 0)'
    if all(k in ks for k in range(5)):
        mm = ~np.isnan(oa)
        line += f' | conf repeat 0 pooled AUC {roc_auc_score(y[mm], oa[mm]):.5f} vs replica {roc_auc_score(y[mm], ob[mm]):.5f}'
    print(line)

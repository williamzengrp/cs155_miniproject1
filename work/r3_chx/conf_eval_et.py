"""Paired check of the ET-only variant-a runs (cache/conf_et_<V>_k*.npz) vs the replica ET family on the confirmation
split set (n_rep=3, seed0=9000).  python -I conf_eval_et.py V"""
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
ks = [k for k in range(len(cs)) if os.path.exists(f'{WORK}/r3_chx/cache/conf_et_{V}_k{k}.npz')
      and os.path.exists(f'{WORK}/ch/cache/chfull_conf_k{k}.npz')]
fd = []
oa, ob, ocb, occ = (np.full(n, np.nan) for _ in range(4))
for k in ks:
    _, fit, val, elig = cs[k]
    a, b = np.load(f'{WORK}/r3_chx/cache/conf_et_{V}_k{k}.npz'), np.load(f'{WORK}/ch/cache/chfull_conf_k{k}.npz')
    fd.append(roc_auc_score(y[val[elig]], a['et_v'][elig]) - roc_auc_score(y[val[elig]], b['et_v'][elig]))
    oa[val[elig]], ob[val[elig]] = a['et_v'][elig], b['et_v'][elig]
    ocb[val[elig]] = (0.6 * b['cat_v'] + 0.4 * b['et_v'])[elig]
    cc = f'{WORK}/r3_chx/cache/conf_c_k{k}.npz'
    if os.path.exists(cc):
        c = np.load(cc)
        occ[val[elig]] = (0.6 * c['cat_v'] + 0.4 * a['et_v'])[elig]
mm = ~np.isnan(oa)
print(f'ET {V}: folds {ks} per-fold diff mean {np.mean(fd):+.5f} ({np.sum(np.array(fd) > 0)}/{len(fd)} > 0); '
      f'pooled ET AUC {roc_auc_score(y[mm], oa[mm]):.5f} vs replica ET {roc_auc_score(y[mm], ob[mm]):.5f}')
m2 = ~np.isnan(occ)
if m2.sum() == mm.sum():
    print(f'mix c(cat)+{V}(ET) pooled AUC {roc_auc_score(y[m2], occ[m2]):.5f} vs replica blend {roc_auc_score(y[m2], ocb[m2]):.5f}')

import sys
sys.dont_write_bytecode = True
import numpy as np, os, warnings
warnings.filterwarnings('ignore')
WORK = './work'
sys.path.insert(0, WORK)
import harness as H
from sklearn.metrics import roc_auc_score
tr, te, y = H.load(); n = len(tr)
cs = H.make_splits(tr, y, n_rep=3, seed0=9000)
ks = [k for k in range(15) if os.path.exists(f'{WORK}/r3_chx/cache/conf_c_k{k}.npz') and os.path.exists(f'{WORK}/ch/cache/chfull_conf_k{k}.npz')]
d = []
for k in ks:
    r, fit, val, elig = cs[k]
    a = np.load(f'{WORK}/r3_chx/cache/conf_c_k{k}.npz'); b = np.load(f'{WORK}/ch/cache/chfull_conf_k{k}.npz')
    pa = 0.6 * a['cat_v'] + 0.4 * a['et_v']; pb = 0.6 * b['cat_v'] + 0.4 * b['et_v']
    d.append(roc_auc_score(y[val[elig]], pa[elig]) - roc_auc_score(y[val[elig]], pb[elig]))
    print(k, round(d[-1], 5), 'ET maxdiff', float(np.max(np.abs(a['et_v'] - b['et_v']))))
reps = sorted({cs[k][0] for k in ks if all(kk in ks for kk in range(5 * cs[k][0], 5 * cs[k][0] + 5))})
for r in reps:
    oa = np.full(n, np.nan); ob = np.full(n, np.nan)
    for k in range(5 * r, 5 * r + 5):
        _, fit, val, elig = cs[k]
        a = np.load(f'{WORK}/r3_chx/cache/conf_c_k{k}.npz'); b = np.load(f'{WORK}/ch/cache/chfull_conf_k{k}.npz')
        oa[val[elig]] = (0.6 * a['cat_v'] + 0.4 * a['et_v'])[elig]; ob[val[elig]] = (0.6 * b['cat_v'] + 0.4 * b['et_v'])[elig]
    mm = ~np.isnan(oa)
    print(f'conf repeat {r}: c {roc_auc_score(y[mm], oa[mm]):.5f} vs replica {roc_auc_score(y[mm], ob[mm]):.5f}')
print('folds', ks, 'mean per-fold diff', round(float(np.mean(d)), 5), f'{sum(x > 0 for x in d)}/{len(d)} > 0')

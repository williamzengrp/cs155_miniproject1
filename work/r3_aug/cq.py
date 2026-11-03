"""quick per-repeat AUCs of chaug fold caches vs the replica (same folds). python -I cq.py FAM TAG NREP"""
import sys, numpy as np
from sklearn.metrics import roc_auc_score
WORK = './work'
sys.path.insert(0, WORK)
import harness as H
fam, tag, nrep = sys.argv[1], sys.argv[2], int(sys.argv[3])
tr, te, y = H.load()
sp = H.make_splits(tr, y, n_rep=5)
for r in range(nrep):
    o = {k: np.full(len(y), np.nan) for k in ('orig', 'new', 'orig_raw', 'new_raw', 'bl_orig', 'bl_new')}
    for k in range(5 * r, 5 * r + 5):
        _, fit, val, el = sp[k]
        C = dict(np.load(f'{WORK}/ch/cache/chfull_std_k{k}.npz'))
        N = dict(np.load(f'{WORK}/r3_aug/cache/chaug_{fam}_{tag}_k{k}.npz'))
        o['orig'][val[el]] = C[f'{fam}_v'][el]; o['new'][val[el]] = N['v'][el]
        o['orig_raw'][val[el]] = C[f'{fam}_raw_v'][el]; o['new_raw'][val[el]] = N['raw_v'][el]
        other = 'et' if fam == 'cat' else 'cat'
        wf = 0.6 if fam == 'cat' else 0.4
        o['bl_orig'][val[el]] = (wf * C[f'{fam}_v'] + (1 - wf) * C[f'{other}_v'])[el]
        o['bl_new'][val[el]] = (wf * N['v'] + (1 - wf) * C[f'{other}_v'])[el]
    m = ~np.isnan(o['orig'])
    print(f'rep {r}: ' + '  '.join(f'{k} {roc_auc_score(y[m], v[m]):.5f}' for k, v in o.items()), flush=True)

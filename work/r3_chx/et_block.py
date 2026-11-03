"""Save an ExtraTrees-family-only block from cached ET-only runs (cache/et_<V>_k*.npz, falling back to cache/<V>_k*.npz)
and ET-only full refits -> results/r3_chx_et_<V>.  python -I et_block.py V"""
import sys, os
sys.dont_write_bytecode = True
import numpy as np
WORK = './work'
C_ = WORK + '/r3_chx/cache'
sys.path.insert(0, WORK)
import harness as H
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
tr, te, y = H.load(); n, m = len(tr), len(te)
splits = H.make_splits(tr, y, n_rep=5)
V = sys.argv[1]
def ep(k):
    p = f'{C_}/et_{V}_k{k}.npz'
    return p if os.path.exists(p) else f'{C_}/{V}_k{k}.npz'
reps = [r for r in range(5) if all(os.path.exists(ep(k)) for k in range(5 * r, 5 * r + 5))]
oof = np.full((5, n), np.nan); tst = np.zeros(m)
for r in reps:
    for k in range(5 * r, 5 * r + 5):
        _, fit, val, elig = splits[k]
        d = np.load(ep(k))
        oof[r, val[elig]] = d['et_v'][elig]
        tst += d['et_t'] / (5 * len(reps))
aucs = np.array([roc_auc_score(y[~np.isnan(oof[r])], oof[r][~np.isnan(oof[r])]) if r in reps else np.nan for r in range(5)])
fs = [f'{C_}/et_{V}_full_s{s}.npz' if os.path.exists(f'{C_}/et_{V}_full_s{s}.npz') else f'{C_}/{V}_full_s{s}.npz' for s in range(10)]
fs = [f for f in fs if os.path.exists(f)]
full = np.mean([np.load(f)['et_t'] for f in fs], axis=0)
H.save(f'r3_chx_et_{V}', dict(oof=oof, aucs=aucs, test_foldavg=tst), full)
ref = H.load_res('chrep_et')['aucs']
print(f'r3_chx_et_{V}: reps {reps} AUC {np.nanmean(aucs):.5f} vs replica ET {ref[reps].mean():.5f} diff {np.round(aucs[reps] - ref[reps], 5)}; '
      f'full refits {len(fs)}; spearman(test_full, charlie file) {spearmanr(full, H.load_res("chrep_bl")["test_full"])[0]:.4f}')

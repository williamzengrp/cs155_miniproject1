"""ch agent: preliminary read of the replica from cached folds (first NREP repeats of a split set)."""
import sys
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
WORK = './work'
sys.path.insert(0, WORK)
import harness as H
kind, NREP = sys.argv[1], int(sys.argv[2])
tr, te, y = H.load()
splits = (H.make_splits(tr, y, n_rep=5) if kind == 'std' else H.make_splits(tr, y, n_rep=3, seed0=9000))[:5 * NREP]
C = [dict(np.load(f'{WORK}/ch/cache/chfull_{kind}_k{k}.npz')) for k in range(len(splits))]
def oof(fn):
    return H.run_cv(lambda fi, vi, k: (fn(C[k]), None), splits, y, verbose=False)
V = {'cat': oof(lambda c: c['cat_v']), 'et': oof(lambda c: c['et_v']),
     'blend': oof(lambda c: 0.6 * c['cat_v'] + 0.4 * c['et_v']),
     'cat_raw': oof(lambda c: c['cat_raw_v']), 'et_raw': oof(lambda c: c['et_raw_v']),
     'blend_raw': oof(lambda c: 0.6 * c['cat_raw_v'] + 0.4 * c['et_raw_v'])}
for k, v in V.items():
    print(f'{k:10s} {v["auc"]:.5f} {np.round(v["aucs"], 4)}')
F8 = ['fkd_lgbm', 'fkt_xgb', 'fkt_cat', 'fkd_et', 'fkd_rf', 'mdl_mlp_ple', 'ch_et_proj', 'ch_lgbm_proj']
if kind == 'conf':
    F8 = ['conf_' + x for x in ['fkd_lgbm', 'fkt_xgb', 'fkt_cat', 'fkd_et', 'fkd_rf', 'mlp_ple', 'ch_et', 'ch_lgbm']]
M = [H.load_res(m)['oof'] for m in F8]
b, g = [], []
for r in range(NREP):
    m = ~np.isnan(V['blend']['oof'][r])
    R8 = [rankdata(o[r, m]) for o in M]
    b.append(roc_auc_score(y[m], np.mean(R8, 0)))
    g.append(roc_auc_score(y[m], np.mean(R8 + [rankdata(V['blend']['oof'][r, m])], 0)))
b, g = np.array(b), np.array(g)
print(f'final8 first {NREP} reps {b.mean():.5f}; +replica blend (1/9) {g.mean():.5f} diff {(g-b).mean():+.5f} per-rep {np.round(g-b, 4)}')

"""Calibrate seed noise: baseline LGBM re-run with different seeds (same splits) -> paired diffs."""
import sys, json, time, warnings
import numpy as np
W = './work'
sys.path.insert(0, W)
import harness as H
from time_features import build_features, FEATURE_GROUPS
from lightgbm import LGBMClassifier
warnings.filterwarnings('ignore')
tr, te, y = H.load()
X, Xt = build_features(tr, te, FEATURE_GROUPS)
P = json.load(open(H.HERE + '/best_params_v3.json'))
splits = H.make_splits(tr, y, n_rep=5)
base = H.load_res('base_lgbm')
out = {}
for off in (0, 100, 200):
    def fp(a, b, k):
        m = LGBMClassifier(**P['lgbm'], random_state=k + off, n_jobs=2, verbose=-1).fit(X.iloc[a], y[a])
        return m.predict_proba(X.iloc[b])[:, 1], None
    t = time.time()
    out[off] = H.run_cv(fp, splits, y, tag=f'base seed+{off}')
    print('  vs base_lgbm', np.round(out[off]['aucs'] - base['aucs'], 4), f'{time.time()-t:.0f}s', flush=True)
    np.save(W + f'/te/out/base_seed{off}_oof.npy', out[off]['oof'])
for a, b in ((100, 0), (200, 0), (200, 100)):
    d = out[a]['aucs'] - out[b]['aucs']
    print(f'seed+{a} - seed+{b}: mean {d.mean():+.5f} per-rep {np.round(d, 4)}')
# 2-seed average vs others
for r in range(5):
    pass
from sklearn.metrics import roc_auc_score
avg = (out[100]['oof'] + out[200]['oof']) / 2
a2 = np.array([roc_auc_score(y[~np.isnan(avg[r])], avg[r][~np.isnan(avg[r])]) for r in range(5)])
print('avg(seed100,200) AUC', a2.mean().round(5), np.round(a2, 4), 'minus seed0:', np.round(a2 - out[0]['aucs'], 4))

"""Baseline = teammates' best label-free pipeline (FEATURE_GROUPS + best_params_v3) under the new CV."""
import json, sys, time, warnings
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
sys.path.insert(0, './work')
import harness as H
from time_features import build_features, FEATURE_GROUPS
warnings.filterwarnings('ignore')
NJ = int(sys.argv[1]) if len(sys.argv) > 1 else 8

tr, te, y = H.load()
X, Xt = build_features(tr, te, FEATURE_GROUPS)
fill = X.median()
Xf, Xtf = X.fillna(fill), Xt.fillna(fill)
P = json.load(open(H.HERE + '/best_params_v3.json'))
print('n features', X.shape[1])

def mk(name, seed):
    p = dict(P[name])
    if name == 'xgb': return XGBClassifier(**p, tree_method='hist', random_state=seed, n_jobs=NJ)
    if name == 'lgbm': return LGBMClassifier(**p, random_state=seed, n_jobs=NJ, verbose=-1)
    if name == 'cat': return CatBoostClassifier(**p, random_seed=seed, verbose=0, thread_count=NJ, allow_writing_files=False)
    if name == 'et': return ExtraTreesClassifier(**p, random_state=seed, n_jobs=NJ)
    if name == 'rf': return RandomForestClassifier(**p, random_state=seed, n_jobs=NJ)

splits = H.make_splits(tr, y, n_rep=5)
gsplits = H.make_group_splits(tr, y, n_rep=2)
out = {}
for name in ['lgbm', 'xgb', 'cat', 'et', 'rf']:
    A, At = (Xf, Xtf) if name in ('et', 'rf') else (X, Xt)
    def fp(fit_idx, val_idx, k):
        m = mk(name, k); m.fit(A.iloc[fit_idx], y[fit_idx])
        return m.predict_proba(A.iloc[val_idx])[:, 1], m.predict_proba(At)[:, 1]
    t = time.time()
    res = H.run_cv(fp, splits, y, tag=name + ' new-CV')
    H.save('base_' + name, res)
    out[name] = res
    g = H.run_cv(fp, gsplits, y, tag=name + ' grouped-CV(ref)')
    print(f'   {time.time()-t:.0f}s', flush=True)

# rank-average blend per repeat
R = out['lgbm']['oof'].shape[0]
for combo in [['lgbm', 'xgb', 'cat'], ['lgbm', 'xgb', 'cat', 'et', 'rf']]:
    aucs = []
    for r in range(R):
        m = ~np.isnan(out['lgbm']['oof'][r])
        s = sum(rankdata(out[n]['oof'][r, m]) for n in combo)
        aucs.append(roc_auc_score(y[m], s))
    print('rank-avg', combo, np.mean(aucs).round(5), np.round(aucs, 4))

"""Teammates' 5 tree models + params on the final 58-feature set; same splits/seeds as baseline.py."""
import json, sys, time, warnings
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
sys.path.insert(0, './work')
sys.path.insert(0, './work/main')
import harness as H
from features_final import build_features_final
warnings.filterwarnings('ignore')
NJ = int(sys.argv[1]); names = sys.argv[2:]
tr, te, y = H.load()
X, Xt = build_features_final(tr, te)
print('n features', X.shape[1], flush=True)
fill = X.median(); Xf, Xtf = X.fillna(fill), Xt.fillna(fill)
P = json.load(open(H.HERE + '/best_params_v3.json'))
def mk(name, seed):
    p = dict(P[name])
    if name == 'xgb': return XGBClassifier(**p, tree_method='hist', random_state=seed, n_jobs=NJ)
    if name == 'lgbm': return LGBMClassifier(**p, random_state=seed, n_jobs=NJ, verbose=-1)
    if name == 'cat': return CatBoostClassifier(**p, random_seed=seed, verbose=0, thread_count=NJ, allow_writing_files=False)
    if name == 'et': return ExtraTreesClassifier(**p, random_state=seed, n_jobs=NJ)
    if name == 'rf': return RandomForestClassifier(**p, random_state=seed, n_jobs=NJ)
splits = H.make_splits(tr, y, n_rep=5)
for name in names:
    A, At = (Xf, Xtf) if name in ('et', 'rf') else (X, Xt)
    def fp(fit_idx, val_idx, k):
        m = mk(name, k); m.fit(A.iloc[fit_idx], y[fit_idx])
        return m.predict_proba(A.iloc[val_idx])[:, 1], None
    t = time.time()
    res = H.run_cv(fp, splits, y, tag=name + ' v3feats')
    base = H.load_res('base_' + name)
    print(f'   vs base67 paired {H.paired(res, base)}  ({time.time()-t:.0f}s)', flush=True)
    H.save('v3_' + name, res)

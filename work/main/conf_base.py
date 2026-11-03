"""Teammates' pipeline (67 features + best_params_v3, 5 trees) on arbitrary splits, OOF only.
python -I conf_base.py NJ SEED0 NREP PREFIX"""
import json, sys, warnings
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
sys.path.insert(0, './work')
import harness as H
from time_features import build_features, FEATURE_GROUPS
warnings.filterwarnings('ignore')
NJ, SEED0, NREP, PREFIX = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
tr, te, y = H.load()
X, Xt = build_features(tr, te, FEATURE_GROUPS)
Xf = X.fillna(X.median())
P = json.load(open(H.HERE + '/best_params_v3.json'))
def mk(name, seed):
    p = dict(P[name])
    if name == 'xgb': return XGBClassifier(**p, tree_method='hist', random_state=seed, n_jobs=NJ)
    if name == 'lgbm': return LGBMClassifier(**p, random_state=seed, n_jobs=NJ, verbose=-1)
    if name == 'cat': return CatBoostClassifier(**p, random_seed=seed, verbose=0, thread_count=NJ, allow_writing_files=False)
    if name == 'et': return ExtraTreesClassifier(**p, random_state=seed, n_jobs=NJ)
    if name == 'rf': return RandomForestClassifier(**p, random_state=seed, n_jobs=NJ)
splits = H.make_splits(tr, y, n_rep=NREP, seed0=SEED0)
for name in ['lgbm', 'xgb', 'cat', 'et', 'rf']:
    A = Xf if name in ('et', 'rf') else X
    def fp(fi, vi, k):
        m = mk(name, k); m.fit(A.iloc[fi], y[fi])
        return m.predict_proba(A.iloc[vi])[:, 1], None
    H.save(f'{PREFIX}_{name}', H.run_cv(fp, splits, y, tag=f'{PREFIX}_{name}'))

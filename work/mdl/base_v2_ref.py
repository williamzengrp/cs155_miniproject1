"""Reference only: the 5 baseline tree models (teammates params, fixed, no tuning) on the v2 feature set
and the STANDARD splits, to measure blend gains of mdl_*_v2 models against a v2 tree blend.
Saved to work/mdl/cache/std_v2_base_<name>.npz (not to results/)."""
import json, os, sys, warnings
sys.path.insert(0, './work/mdl')
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import common as C
warnings.filterwarnings('ignore')
NJ = 2
H = C.H
tr, te, y = H.load(); X, Xt = C.data_v2()
Xf = X.fillna(X.median())
P = json.load(open(H.HERE + '/best_params_v3.json'))
os.makedirs(os.path.join(C.MDL, 'cache'), exist_ok=True)


def mk(name, seed):
    p = dict(P[name])
    if name == 'xgb': return XGBClassifier(**p, tree_method='hist', random_state=seed, n_jobs=NJ)
    if name == 'lgbm': return LGBMClassifier(**p, random_state=seed, n_jobs=NJ, verbose=-1)
    if name == 'cat': return CatBoostClassifier(**p, random_seed=seed, verbose=0, thread_count=NJ, allow_writing_files=False)
    if name == 'et': return ExtraTreesClassifier(**p, random_state=seed, n_jobs=NJ)
    if name == 'rf': return RandomForestClassifier(**p, random_state=seed, n_jobs=NJ)


splits = H.make_splits(tr, y, n_rep=5)
for name in C.BASE:
    A = Xf if name in ('et', 'rf') else X
    def fp(fit_idx, val_idx, k):
        m = mk(name, k); m.fit(A.iloc[fit_idx], y[fit_idx])
        return m.predict_proba(A.iloc[val_idx])[:, 1], None
    res = H.run_cv(fp, splits, y, tag=name + ' v2 std-splits')
    np.savez(os.path.join(C.MDL, 'cache', f'std_v2_base_{name}.npz'), oof=res['oof'], aucs=res['aucs'])
b = [np.load(os.path.join(C.MDL, 'cache', f'std_v2_base_{n}.npz'))['oof'] for n in C.BASE]
a = C.blend_aucs(y, b, [0.2] * 5)
print('5-tree rank blend (v2, std splits)', a.mean().round(5), np.round(a, 4))

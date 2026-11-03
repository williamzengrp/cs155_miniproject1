"""Teammates' ExtraTrees (67 features + best_params_v3, median fill) on any split set, as in baseline.py.
python -I base_et_run.py NJ SEED0 NREP PREFIX     env DO_FULL=1 -> also full refit on all rows (seeds 0,1,2) -> test_full
Saves results/<PREFIX>_et.npz."""
import json, os, sys, warnings
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
sys.path.insert(0, './work')
import harness as H
from time_features import build_features, FEATURE_GROUPS
warnings.filterwarnings('ignore')
NJ, SEED0, NREP, PREFIX = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
tr, te, y = H.load()
X, Xt = build_features(tr, te, FEATURE_GROUPS)
fill = X.median()
Xf, Xtf = X.fillna(fill), Xt.fillna(fill)
P = json.load(open(H.HERE + '/best_params_v3.json'))['et']
mk = lambda seed: ExtraTreesClassifier(**P, random_state=seed, n_jobs=NJ)


def fp(fit_idx, val_idx, k):
    m = mk(k); m.fit(Xf.iloc[fit_idx], y[fit_idx])
    return m.predict_proba(Xf.iloc[val_idx])[:, 1], m.predict_proba(Xtf)[:, 1]


res = H.run_cv(fp, H.make_splits(tr, y, n_rep=NREP, seed0=SEED0), y, tag=f'{PREFIX}_et')
tf = None
if os.environ.get('DO_FULL', '0') == '1':
    tf = np.mean([mk(s).fit(Xf, y).predict_proba(Xtf)[:, 1] for s in range(3)], 0)
H.save(f'{PREFIX}_et', res, tf)

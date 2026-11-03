"""r3_chp: ExtraTrees-only variants inside Charlie's pipeline (same fold matrices: project view + Charlie's date TE,
median imputation; same seeds S = SEED0 + k). Output in the chp_var cache format (et_v / et_t only), so chp_lib
reads them as 'var:et<NAME>:0' (CatBoost then comes from the replica).
python -I -B et_var.py K0 K1 NAME [NAME ...]      or      python -I -B et_var.py full S_IDX NAME [NAME ...]"""
import sys
sys.dont_write_bytecode = True
import os, time, warnings
import numpy as np
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/ch')
import harness as H
from ch_common import connected_groups
from ch_feats import charlie_pool
from chfull_lib import Replica, SEED0, CFG, dup_weights
from sklearn.ensemble import ExtraTreesClassifier
warnings.filterwarnings('ignore')
THREADS = int(os.environ.get('CHP_THREADS', '3'))
CACHE = WORK + '/r3_chp/cache'
BASE = dict(n_estimators=900, max_depth=15, min_samples_leaf=2, max_features=0.1827077108658931)
CONF = {'base': {}, 'mf3': dict(max_features=0.3), 'mf5': dict(max_features=0.5), 'dN': dict(max_depth=None),
        'd20': dict(max_depth=20), 'l1': dict(min_samples_leaf=1), 'l4': dict(min_samples_leaf=4),
        'n2500': dict(n_estimators=2500), 'ent': dict(criterion='entropy'), 'dp0': dict(_dp=0.0),
        'dp05': dict(_dp=0.5), 'mf3l4dN': dict(max_features=0.3, min_samples_leaf=4, max_depth=None),
        'l3d20': dict(min_samples_leaf=3, max_depth=20), 'mf3n2500': dict(max_features=0.3, n_estimators=2500)}
tr, te, y = H.load()
n = len(tr)
F, BL = charlie_pool(tr, te)
R = Replica(tr, te, y, F[BL['PROJECT']], connected_groups(tr), threads=THREADS)


def fit_et(name, Xf, yf, w_uri, S):
    p = dict(BASE); p.update(CONF[name]); dp = p.pop('_dp', 1.0)
    return ExtraTreesClassifier(n_jobs=THREADS, random_state=S, **p).fit(Xf, yf, sample_weight=dup_weights(w_uri, dp))


if sys.argv[1] == 'full':
    fit = np.arange(n)
    Xf, _, Xt = R.matrices(fit, [], True)
    for s in [int(x) for x in sys.argv[2].split(',')]:
        for name in sys.argv[3:]:
            path = f'{CACHE}/varfull_et{name}_s{s}.npz'
            if os.path.exists(path):
                continue
            m = fit_et(name, Xf, y, R.uri, SEED0 + 1000 * (s + 1))
            np.savez(path, et_t=m.predict_proba(Xt)[:, 1])
            print(f'[full et{name} s={s}]', flush=True)
else:
    k0, k1 = int(sys.argv[1]), int(sys.argv[2])
    splits = H.make_splits(tr, y, n_rep=5)
    for k in range(k0, k1):
        r, fit, val, elig = splits[k]
        t0 = time.time()
        Xf, (Xv,), Xt = R.matrices(fit, [val], True)
        for name in sys.argv[3:]:
            path = f'{CACHE}/var_et{name}_o0_k{k}.npz'
            if os.path.exists(path):
                continue
            m = fit_et(name, Xf, y[fit], R.uri[fit], SEED0 + k)
            np.savez(path, et_v=m.predict_proba(Xv)[:, 1], et_t=m.predict_proba(Xt)[:, 1])
        print(f'[et k={k}] {time.time() - t0:.1f}s', flush=True)

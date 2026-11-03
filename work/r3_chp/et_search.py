"""r3_chp: modest ExtraTrees hyper-parameter screen inside Charlie's pipeline (same fold matrices, same seeds).
Scores: ET alone (raw) and the Charlie blend 0.6*cat(cached, corrected) + 0.4*ET(raw).
python -I -B et_search.py NJOBS CONFIG_NAMES..."""
import sys
sys.dont_write_bytecode = True
import os, time, warnings, pickle
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/ch')
import harness as H
from ch_common import connected_groups
from ch_feats import charlie_pool
from chfull_lib import Replica, SEED0, dup_weights
warnings.filterwarnings('ignore')
NJ = int(sys.argv[1])
CACHE = WORK + '/r3_chp/cache'
tr, te, y = H.load()
splits = H.make_splits(tr, y, n_rep=5)
MP = CACHE + '/mats_std.pkl'
if os.path.exists(MP):
    M = pickle.load(open(MP, 'rb'))
else:
    F, BL = charlie_pool(tr, te)
    R = Replica(tr, te, y, F[BL['PROJECT']], connected_groups(tr), threads=NJ)
    M = []
    for k, (r, fit, val, elig) in enumerate(splits):
        Xf, (Xv,), Xt = R.matrices(fit, [val], True)
        M.append((Xf.astype(np.float32), Xv.astype(np.float32), Xt.astype(np.float32)))
    pickle.dump(M, open(MP, 'wb'))
uri = tr['uri'].astype(str).to_numpy()
C = [dict(np.load(f'{WORK}/ch/cache/chfull_std_k{k}.npz')) for k in range(25)]
BASE = dict(n_estimators=900, max_depth=15, min_samples_leaf=2, max_features=0.1827077108658931)
CONF = {
    'base': {}, 'mf10': dict(max_features=0.10), 'mf30': dict(max_features=0.30), 'mf50': dict(max_features=0.5),
    'dNone': dict(max_depth=None), 'd20': dict(max_depth=20), 'd12': dict(max_depth=12), 'l1': dict(min_samples_leaf=1),
    'l4': dict(min_samples_leaf=4), 'l8': dict(min_samples_leaf=8), 'n2500': dict(n_estimators=2500),
    'ent': dict(criterion='entropy'), 'nw': dict(_dp=0.0), 'dp05': dict(_dp=0.5),
    'mf30l4': dict(max_features=0.30, min_samples_leaf=4), 'mf30d20': dict(max_features=0.30, max_depth=20),
    'mf30l4dN': dict(max_features=0.30, min_samples_leaf=4, max_depth=None),
    'mf50l8dN': dict(max_features=0.5, min_samples_leaf=8, max_depth=None),
    'mf30l3d20': dict(max_features=0.30, min_samples_leaf=3, max_depth=20),
    'l3d20': dict(min_samples_leaf=3, max_depth=20), 'l4dN': dict(min_samples_leaf=4, max_depth=None),
    'mf40l6dN': dict(max_features=0.4, min_samples_leaf=6, max_depth=None),
    'mf25l5dN': dict(max_features=0.25, min_samples_leaf=5, max_depth=None),
}


def aucs(vals):
    oof = np.full((5, len(y)), np.nan)
    for k, (r, fit, val, elig) in enumerate(splits):
        oof[r, val[elig]] = vals[k][elig]
    return np.array([roc_auc_score(y[~np.isnan(o)], o[~np.isnan(o)]) for o in oof])

b_et = aucs([C[k]['et_raw_v'] for k in range(25)])
b_bl = aucs([0.6 * C[k]['cat_v'] + 0.4 * C[k]['et_raw_v'] for k in range(25)])
print(f'cached: et raw {b_et.mean():.5f}  blend(cat corr + et raw) {b_bl.mean():.5f}', flush=True)
for name in sys.argv[2:]:
    p = dict(BASE); p.update(CONF[name]); dp = p.pop('_dp', 1.0)
    t0 = time.time()
    V, T = [], []
    for k, (r, fit, val, elig) in enumerate(splits):
        Xf, Xv, Xt = M[k]
        m = ExtraTreesClassifier(n_jobs=NJ, random_state=SEED0 + k, **p).fit(Xf, y[fit], sample_weight=dup_weights(uri[fit], dp))
        V.append(m.predict_proba(Xv)[:, 1]); T.append(m.predict_proba(Xt)[:, 1])
    a = aucs(V); bl = aucs([0.6 * C[k]['cat_v'] + 0.4 * V[k] for k in range(25)])
    bl5 = aucs([0.5 * C[k]['cat_v'] + 0.5 * V[k] for k in range(25)])
    np.savez(f'{CACHE}/etscr_{name}.npz', v=np.array(V, dtype=object), t=np.array(T))
    print(f'{name:12s} et {a.mean():.5f} (d {a.mean() - b_et.mean():+.5f})  blend.6/.4 {bl.mean():.5f} '
          f'(d {bl.mean() - b_bl.mean():+.5f}, reps+ {(bl > b_bl).sum()})  blend.5/.5 {bl5.mean():.5f}  {time.time() - t0:.0f}s', flush=True)

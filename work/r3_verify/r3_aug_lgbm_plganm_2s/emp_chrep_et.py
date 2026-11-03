"""chrep_bl ET family, fold 0: true labels vs ALL val labels flipped, same thread count, bit comparison."""
import sys, time, warnings, numpy as np
warnings.filterwarnings('ignore')
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/ch')
import harness as H
from ch_common import connected_groups
from ch_feats import charlie_pool
from chfull_lib import Replica, SEED0
tr, te, y = H.load()
r, fit, val, elig = H.make_splits(tr, y, n_rep=5)[0]
out = {}
for MODE in ('true', 'flip'):
    yy = y.copy(); trm = tr.copy()
    if MODE == 'flip':
        yy[val] = 1 - yy[val]
        pt = trm['Popularity_Type'].to_numpy().copy(); pt[val] = np.where(pt[val] == 'High', 'Low', 'High'); trm['Popularity_Type'] = pt
    F, BL = charlie_pool(trm, te)
    R = Replica(trm, te, yy, F[BL['PROJECT']], connected_groups(trm), threads=1)
    Xf, (Xv,), Xt = R.matrices(fit, [val], True)
    inner = R.inner_sets(fit, SEED0)
    (rv,), rt, ip = R.run_family('extra', fit, Xf, [Xv], Xt, inner, SEED0)
    out[MODE] = dict(Xv=Xv, Xt=Xt, rv=rv, rt=rt, v=R.correct('extra', fit, ip, rv, trm.iloc[val]), t=R.correct('extra', fit, ip, rt, te))
for k in out['true']:
    print('ET', k, 'true vs flipped-val identical:', bool(np.array_equal(out['true'][k], out['flip'][k], equal_nan=True)), flush=True)

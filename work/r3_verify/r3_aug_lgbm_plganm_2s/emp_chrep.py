"""chrep_bl teacher, fold 0: re-run Charlie replica with ALL val labels flipped (both y and the Popularity_Type
column of the train frame) and compare with the cached fold-0 predictions (made with the true labels)."""
import sys, time, numpy as np
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/ch')
import harness as H
from ch_common import connected_groups
from ch_feats import charlie_pool
from chfull_lib import Replica, SEED0
t0 = time.time()
tr, te, y = H.load()
r, fit, val, elig = H.make_splits(tr, y, n_rep=5)[0]
MODE = sys.argv[1] if len(sys.argv) > 1 else 'flip'
yy = y.copy(); trm = tr.copy()
if MODE == 'flip':
    yy[val] = 1 - yy[val]
    pt = trm['Popularity_Type'].to_numpy().copy()
    pt[val] = np.where(pt[val] == 'High', 'Low', 'High'); trm['Popularity_Type'] = pt
F, BL = charlie_pool(trm, te)
R = Replica(trm, te, yy, F[BL['PROJECT']], connected_groups(trm), threads=2)
S = SEED0 + 0
Xf, (Xv,), Xt = R.matrices(fit, [val], True)
inner = R.inner_sets(fit, S)
c = dict(np.load(WORK + '/ch/cache/chfull_std_k0.npz'))
print(MODE, 'prep', f'{time.time()-t0:.0f}s', flush=True)
for fam, tag in (('extra', 'et'), ('cat', 'cat')):
    (rv,), rt, ip = R.run_family(fam, fit, Xf, [Xv], Xt, inner, S)
    v = R.correct(fam, fit, ip, rv, trm.iloc[val])
    print(MODE, tag, 'raw_v identical to cache:', bool(np.array_equal(rv, c[f'{tag}_raw_v'])), 'maxabs', np.abs(rv - c[f'{tag}_raw_v']).max(),
          '| corrected v identical:', bool(np.array_equal(v, c[f'{tag}_v'])), 'maxabs', np.abs(v - c[f'{tag}_v']).max(),
          '| inner ip identical:', bool(np.array_equal(ip, c[f'{tag}_ip'])), f'{time.time()-t0:.0f}s', flush=True)

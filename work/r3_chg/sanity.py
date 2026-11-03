"""Sanity checks: RX('none') == Replica matrices on fold 0; shapes / timing of every group; train AUC of new cols."""
import sys, time
sys.argv = ['chg.py', 'x']
import numpy as np, pandas as pd
sys.path.insert(0, './work/r3_chg')
import chg
from chg import H, build, lf_block, Replica, charlie_pool, connected_groups, SongGraph
from sklearn.metrics import roc_auc_score
tr, te, y = H.load()
sp = H.make_splits(tr, y, n_rep=5)
r, fit, val, elig = sp[0]
F, BL = charlie_pool(tr, te)
R0 = Replica(tr, te, y, F[BL['PROJECT']], connected_groups(tr), threads=1)
A = R0.matrices(fit, [val], True)
R1 = build(tr, te, y, 'none', 1)
B = R1.matrices(fit, [val], True, gseed=0)
print('none == replica:', np.array_equal(A[0], B[0]), np.array_equal(A[1][0], B[1][0]), np.array_equal(A[2], B[2]), A[0].shape)
t = time.time()
R2 = build(tr, te, y, 'hc2+hk2+dupm+spec8+dom+drank', 1)
print('build all', round(time.time() - t, 1), 's')
t = time.time()
C = R2.matrices(fit, [val], True, gseed=0)
print('matrices all', round(time.time() - t, 1), 's', C[0].shape, C[1][0].shape, C[2].shape)
print('prefix equal:', np.array_equal(C[0][:, :A[0].shape[1]], A[0]))
t = time.time(); inner = R2.inner_sets(fit, 123, gseed=0); print('inner sets', round(time.time() - t, 1), 's', [x[2].shape for x in inner])
names = list(pd.concat([lf_block(g, tr, te, R2.G) for g in ['dupm', 'spec8', 'dom', 'drank']], axis=1).columns) + ['g_hc2', 'g_hk2']
X = C[0][:, A[0].shape[1]:]
for j, nm in enumerate(names):
    a = roc_auc_score(y[fit], X[:, j]); v = roc_auc_score(y[val][elig], C[1][0][elig, A[0].shape[1] + j])
    print(f'{nm:24s} fitAUC {max(a, 1-a):.3f} valAUC {max(v, 1-v):.3f}  test mean {C[2][:, A[0].shape[1]+j].mean():.3f} fit mean {X[:, j].mean():.3f}')

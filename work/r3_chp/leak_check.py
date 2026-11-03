"""r3_chp: leakage self-check of the per-fold inputs used by the lab / alb variants: flipping the labels of the
validation rows (and of the test rows, which have none) must not change any fit / val / test input matrix."""
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/ch'); sys.path.insert(0, WORK + '/te')
import harness as H
from ch_common import connected_groups
from ch_feats import charlie_pool
from chfull_lib import Replica
from label_features import label_features, BEST_SPEC

tr, te, y = H.load()
n, m = len(tr), len(te)
F, BL = charlie_pool(tr, te)
splits = H.make_splits(tr, y, n_rep=5)
XA = pd.DataFrame(index=range(n + m))
for k in (0, 13):
    r, fit, val, elig = splits[k]
    yf = y.copy(); yf[val] = 1 - yf[val]
    outs = []
    for yy in (y, yf):
        R = Replica(tr, te, yy, F[BL['PROJECT']], connected_groups(tr), threads=1)
        Xf, (Xv,), Xt = R.matrices(fit, [val], True)
        Lf, (Lv, Lt) = label_features(XA, yy, fit, [val, np.arange(n, n + m)], tr, te, seed=k, spec=BEST_SPEC)
        outs.append([Xf, Xv, Xt, Lf.to_numpy(), Lv.to_numpy(), Lt.to_numpy()])
    same = [np.array_equal(a, b) for a, b in zip(*outs)]
    print(f'fold {k}: identical after flipping val labels (Xf, Xv, Xt, Lf, Lv, Lt): {same}')
    assert all(same)
print('OK')

"""Auditor: recompute the replica's inner-OOF predictions (cat_ip / et_ip, the residual-correction input) for fold K
with the VALIDATION labels flipped and compare with the cached chfull_std_k<K>.npz. usage: FAM(cat|extra) K"""
import sys, time, warnings
sys.dont_write_bytecode = True
warnings.filterwarnings('ignore')
W = './work'
sys.path.insert(0, W); sys.path.insert(0, W + '/ch')
fam, K = sys.argv[1], int(sys.argv[2])
t0 = time.time()
import numpy as np
import harness as H
from ch_common import connected_groups
from ch_feats import charlie_pool
from chfull_lib import Replica, SEED0, make_model, dup_weights, CFG
tr, te, y = H.load()
splits = H.make_splits(tr, y, n_rep=5)
r, fit, val, elig = splits[K]
yf = y.copy(); yf[val] = 1 - yf[val]
F, BL = charlie_pool(tr, te)
R = Replica(tr, te, yf, F[BL['PROJECT']], connected_groups(tr), threads=1)
S = SEED0 + K
inner = R.inner_sets(fit, S)
ip = np.full(len(fit), np.nan)
for it, iv, I, J in inner:
    mi = make_model(fam, S + 200, 1).fit(I, yf[fit[it]], sample_weight=dup_weights(R.uri[fit[it]], CFG[fam]['duplicate_power']))
    ip[iv] = mi.predict_proba(J)[:, 1]
tag = 'cat' if fam == 'cat' else 'et'
c = dict(np.load(f'{W}/ch/cache/chfull_std_k{K}.npz'))
print(f'{tag}_ip (val labels flipped) max|diff| vs cached replica: {float(np.abs(ip - c[tag + "_ip"]).max())}  '
      f'{time.time() - t0:.1f}s', flush=True)

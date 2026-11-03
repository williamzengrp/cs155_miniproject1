"""Reproduce a cvet-mode ET(a) fold from the audited code with VAL labels flipped; compare with cache et_a_k{k}."""
import sys, os
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
WORK = './work'
for p in (WORK + '/r3_chx', WORK, WORK + '/ch', WORK + '/main', WORK + '/te'):
    sys.path.insert(0, p)
import harness as H
from sklearn.metrics import roc_auc_score
tr, te, y = H.load(); n = len(tr)
splits = H.make_splits(tr, y, n_rep=5)
from ch_common import connected_groups
from ch_feats import charlie_pool
from chx_lib import ReplicaX, CD_COLS, SEED0
from features_final import build_features_final
F, BL = charlie_pool(tr, te)
X, Xt = build_features_final(tr, te)
XA = pd.concat([X, Xt], ignore_index=True)
k = int(sys.argv[1])
_, fit, val, elig = splits[k]
yf = y.copy(); yf[val] = 1 - y[val]
R = ReplicaX(tr, te, yf, F[BL['PROJECT']], connected_groups(tr), 'a', threads=2, XA=XA, CD=XA[CD_COLS].to_numpy(float))
S = SEED0 + k
Xf, Xa, Xt_ = R.matrices(fit, [val], True, lf_seed=k)
inner = R.inner_sets(fit, S)
ra, rt, ip = R.run_family('extra', fit, Xf, Xa, [val], Xt_, inner, S)
v = R.correct('extra', fit, ip, ra[0], tr.iloc[val]); t = R.correct('extra', fit, ip, rt, te)
c = np.load(f'{WORK}/r3_chx/cache/et_a_k{k}.npz')
print(f'[repro-et k={k}, val labels flipped] vs cache et_a_k{k}: v {np.abs(v - c["et_v"]).max():.3g} '
      f't {np.abs(t - c["et_t"]).max():.3g} ip {np.abs(ip - c["et_ip"]).max():.3g}; '
      f'AUC {roc_auc_score(y[val[elig]], v[elig]):.5f}')

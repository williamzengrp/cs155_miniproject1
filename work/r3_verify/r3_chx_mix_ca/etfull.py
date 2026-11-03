"""Re-run the ET full refit seed 0 (variant a, fit = all train rows) and compare with cache/et_a_full_s0.npz."""
import sys
sys.dont_write_bytecode = True
import time, warnings
import numpy as np, pandas as pd
WORK = './work'
HERE = WORK + '/r3_chx'
for p in (HERE, WORK, WORK + '/ch', WORK + '/main', WORK + '/te'):
    sys.path.insert(0, p)
import harness as H
warnings.filterwarnings('ignore')
from chx_lib import ReplicaX, SEED0
from ch_common import connected_groups
from ch_feats import charlie_pool
t0 = time.time()
tr, te, y = H.load(); n, m = len(tr), len(te)
F, BL = charlie_pool(tr, te)
s = 0; S = SEED0 + 1000 * (s + 1); fit = np.arange(n)
R = ReplicaX(tr, te, y, F[BL['PROJECT']], connected_groups(tr), 'a', threads=1, XA=pd.DataFrame(index=range(n + m)))
Xf, Xa, Xt = R.matrices(fit, [], True, lf_seed=s)
inner = R.inner_sets(fit, S)
ra, rt, ip = R.run_family('extra', fit, Xf, Xa, [], Xt, inner, S)
et_t = R.correct('extra', fit, ip, rt, te)
c = np.load(f'{HERE}/cache/et_a_full_s{s}.npz')
print('full refit fit rows', len(fit), '== n', n, '| max|et_t - cache|', np.abs(et_t - c['et_t']).max(),
      'max|raw_t - cache|', np.abs(rt - c['et_raw_t']).max(), 'max|ip - cache|', np.abs(ip - c['et_ip']).max(),
      f'{time.time() - t0:.1f}s')

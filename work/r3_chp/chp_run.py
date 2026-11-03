"""r3_chp: re-run Charlie replica pipeline (copied logic of work/ch/chfull.py; chfull_lib is imported read-only).

  python -I -B chp_run.py full SEEDS            full refit on all train rows for seeds (e.g. 0,1,2) -> cache/full_s<s>.npz
  python -I -B chp_run.py cv K0 K1 SOFF         std folds K0..K1-1 with seed offset SOFF (S = SEED0 + k + SOFF;
                                                inner split seed S+100, inner models S+200, as in chfull.py; SOFF=0
                                                reproduces the cached chfull_std_k*.npz) -> cache/cv_o<SOFF>_k<k>.npz
"""
import sys
sys.dont_write_bytecode = True
import os, time, warnings
import numpy as np
WORK = './work'
sys.path.insert(0, WORK)
sys.path.insert(0, WORK + '/ch')
import harness as H
from ch_common import connected_groups
from ch_feats import charlie_pool
from chfull_lib import Replica, SEED0
warnings.filterwarnings('ignore')
THREADS = int(os.environ.get('CHP_THREADS', '3'))
CACHE = WORK + '/r3_chp/cache'
mode = sys.argv[1]
tr, te, y = H.load()
n = len(tr)
F, BL = charlie_pool(tr, te)
R = Replica(tr, te, y, F[BL['PROJECT']], connected_groups(tr), threads=THREADS)

if mode == 'full':
    seeds = [int(s) for s in sys.argv[2].split(',')]
    fit = np.arange(n)
    for s in seeds:
        path = f'{CACHE}/full_s{s}.npz'
        if os.path.exists(path):
            continue
        S = SEED0 + 1000 * (s + 1)
        t0 = time.time()
        Xf, _, Xt = R.matrices(fit, [], True)
        inner = R.inner_sets(fit, S)
        out = {}
        for fam, tag in (('cat', 'cat'), ('extra', 'et')):
            _, rt, ip = R.run_family(fam, fit, Xf, [], Xt, inner, S)
            out[f'{tag}_raw_t'], out[f'{tag}_ip'] = rt, ip
            out[f'{tag}_t'] = R.correct(fam, fit, ip, rt, te)
        np.savez(path, **out)
        print(f'[full seed {s}] {time.time() - t0:.1f}s', flush=True)

elif mode == 'cv':
    k0, k1, soff = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    splits = H.make_splits(tr, y, n_rep=5)
    for k in range(k0, min(k1, len(splits))):
        path = f'{CACHE}/cv_o{soff}_k{k}.npz'
        if os.path.exists(path):
            continue
        r, fit, val, elig = splits[k]
        S = SEED0 + k + soff
        t0 = time.time()
        Xf, (Xv,), Xt = R.matrices(fit, [val], True)
        inner = R.inner_sets(fit, S)
        out = {}
        for fam, tag in (('cat', 'cat'), ('extra', 'et')):
            (rv,), rt, ip = R.run_family(fam, fit, Xf, [Xv], Xt, inner, S)
            out[f'{tag}_raw_v'], out[f'{tag}_ip'], out[f'{tag}_raw_t'] = rv, ip, rt
            out[f'{tag}_v'] = R.correct(fam, fit, ip, rv, tr.iloc[val])
            out[f'{tag}_t'] = R.correct(fam, fit, ip, rt, te)
        np.savez(path, **out)
        print(f'[cv soff={soff} k={k}] {time.time() - t0:.1f}s', flush=True)

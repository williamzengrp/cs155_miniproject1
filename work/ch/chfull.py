"""ch agent: driver for the faithful replica of Charlie's final model (see chfull_lib.py).

  python -I chfull.py cv {std|conf} K0 K1   folds K0..K1-1 (std = make_splits(n_rep=5); conf = n_rep=3, seed0=9000);
                                            per-fold cache in cache/chfull_<kind>_k<k>.npz (resumable)
  python -I chfull.py full SEEDS            test_full: refit on ALL train rows (TE of train rows cross-fitted,
                                            test rows from all train labels; residual correction from an inner
                                            3-fold over all train rows); SEEDS e.g. 0,1 -> mean over seeds
  python -I chfull.py save {std|conf} NREP  assemble cached folds of the first NREP repeats -> H.save
"""
import os, sys, time, warnings
import numpy as np
WORK = './work'
sys.path.insert(0, WORK)
sys.path.insert(0, WORK + '/ch')
import harness as H
from ch_common import connected_groups
from ch_feats import charlie_pool
from chfull_lib import Replica, SEED0, EXTRA_WEIGHT
warnings.filterwarnings('ignore')
THREADS = 4
CACHE = WORK + '/ch/cache'
mode = sys.argv[1]
tr, te, y = H.load()
n = len(tr)


def splits_for(kind):
    return H.make_splits(tr, y, n_rep=5) if kind == 'std' else H.make_splits(tr, y, n_rep=3, seed0=9000)


if mode in ('cv', 'full'):
    F, BL = charlie_pool(tr, te)
    R = Replica(tr, te, y, F[BL['PROJECT']], connected_groups(tr), threads=THREADS)

if mode == 'cv':
    kind, k0, k1 = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    splits = splits_for(kind)
    with_test = kind == 'std'
    for k in range(k0, min(k1, len(splits))):
        path = f'{CACHE}/chfull_{kind}_k{k}.npz'
        if os.path.exists(path):
            continue
        r, fit, val, elig = splits[k]
        S = SEED0 + k
        t0 = time.time()
        Xf, (Xv,), Xt = R.matrices(fit, [val], with_test)
        inner = R.inner_sets(fit, S)
        t1 = time.time()
        out, tim = {}, {'prep': t1 - t0}
        for fam, tag in (('cat', 'cat'), ('extra', 'et')):
            ta = time.time()
            (rv,), rt, ip = R.run_family(fam, fit, Xf, [Xv], Xt, inner, S)
            out[f'{tag}_raw_v'], out[f'{tag}_ip'] = rv, ip
            out[f'{tag}_v'] = R.correct(fam, fit, ip, rv, tr.iloc[val])
            if with_test:
                out[f'{tag}_raw_t'] = rt
                out[f'{tag}_t'] = R.correct(fam, fit, ip, rt, te)
            tim[tag] = time.time() - ta
        np.savez(path, **out)
        print(f'[{kind} k={k}] prep {tim["prep"]:.1f}s  cat(1 outer + 3 inner fits) {tim["cat"]:.1f}s  '
              f'et(1+3) {tim["et"]:.1f}s  total {time.time() - t0:.1f}s', flush=True)

elif mode == 'full':
    seeds = [int(s) for s in sys.argv[2].split(',')]
    fit = np.arange(n)
    for s in seeds:
        path = f'{CACHE}/chfull_full_s{s}.npz'
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

elif mode == 'save':
    kind, nrep = sys.argv[2], int(sys.argv[3])
    splits = splits_for(kind)[:5 * nrep]
    C = [dict(np.load(f'{CACHE}/chfull_{kind}_k{k}.npz')) for k in range(len(splits))]
    for c in C:
        c['bl_v'] = (1 - EXTRA_WEIGHT) * c['cat_v'] + EXTRA_WEIGHT * c['et_v']
        if kind == 'std':
            c['bl_t'] = (1 - EXTRA_WEIGHT) * c['cat_t'] + EXTRA_WEIGHT * c['et_t']
    full = None
    if kind == 'std':
        fs = sorted(f for f in os.listdir(CACHE) if f.startswith('chfull_full_s'))
        Fl = [dict(np.load(f'{CACHE}/{f}')) for f in fs]
        print('test_full seeds:', fs)
        full = {t: np.mean([f[f'{t}_t'] for f in Fl], axis=0) for t in ('cat', 'et')}
        full['bl'] = (1 - EXTRA_WEIGHT) * full['cat'] + EXTRA_WEIGHT * full['et']
    for tag, name in (('cat', 'cat'), ('et', 'et'), ('bl', 'blend')):
        res = H.run_cv(lambda fi, vi, k: (C[k][f'{tag}_v'], C[k][f'{tag}_t'] if kind == 'std' else None),
                       splits, y, tag=f'{kind} {name}')
        nm = ('ch_full_' if kind == 'std' else 'conf_ch_full_') + name
        H.save(nm, res, None if full is None else full[tag])
        print('saved', nm, flush=True)
    # uncorrected (no residual correction) variants, for reference only (not saved)
    for tag in ('cat', 'et'):
        H.run_cv(lambda fi, vi, k: (C[k][f'{tag}_raw_v'], None), splits, y, tag=f'{kind} {tag} WITHOUT residual corr')
    H.run_cv(lambda fi, vi, k: ((1 - EXTRA_WEIGHT) * C[k]['cat_raw_v'] + EXTRA_WEIGHT * C[k]['et_raw_v'], None),
             splits, y, tag=f'{kind} blend WITHOUT residual corr')

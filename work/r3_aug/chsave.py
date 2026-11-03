"""r3_aug: assemble a Charlie-blend block (0.6 cat + 0.4 et, residual-corrected) from chaug.py caches.
python -I chsave.py NAME CATTAG ETTAG NREP SEEDS
  TAG 'orig' = replica fold cache work/ch/cache/chfull_std_k<k>.npz (CV) and chaug '<fam>_base' full refit (test_full);
  otherwise cache/chaug_<fam>_<TAG>_k<k>.npz and cache/chaug_<fam>_<TAG>_full_s<s>.npz.
Repeats beyond NREP are left all-NaN (screening blocks)."""
import sys
import numpy as np
WORK = './work'
sys.path.insert(0, WORK)
import harness as H

name, ct, et, nrep, seeds = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5].split(',')
tr, te, y = H.load()
splits = H.make_splits(tr, y, n_rep=5)


def fold(fam, tag, k):
    if tag == 'orig':
        C = np.load(f'{WORK}/ch/cache/chfull_std_k{k}.npz')
        return C[f'{fam}_v'], C[f'{fam}_t']
    C = np.load(f'{WORK}/r3_aug/cache/chaug_{fam}_{tag}_k{k}.npz')
    return C['v'], C['t']


def full(fam, tag):
    tag = 'base' if tag == 'orig' else tag
    return np.mean([np.load(f'{WORK}/r3_aug/cache/chaug_{fam}_{tag}_full_s{s}.npz')['t'] for s in seeds], axis=0)


def fp(fi, vi, k):
    cv, ctt = fold('cat', ct, k)
    ev, ett = fold('et', et, k)
    return 0.6 * cv + 0.4 * ev, 0.6 * ctt + 0.4 * ett


res = H.run_cv(fp, splits[:5 * nrep], y, tag=name)
if nrep < 5:
    o = np.full((5, len(y)), np.nan)
    o[:nrep] = res['oof']
    res['oof'] = o
tf = 0.6 * full('cat', ct) + 0.4 * full('et', et)
assert np.isfinite(tf).all() and len(tf) == len(te)
H.save(name, res, tf)
print('saved', name, 'corr(full, foldavg)', np.corrcoef(tf, res['test_foldavg'])[0, 1].round(4))

"""r3_chp: test_full assembly from full-refit caches (all train rows).
Sources: 'full:S' (replica pipeline incl. its own inner fits, cache/full_s<S>.npz),
         'sd:S:ITERS:TEV[:IT]' (extra outer seed, cache/sdfull_s<S>_i<ITERS>_te<TEV>.npz),
         'var:V:S[:IT]' (outer-only variant, cache/varfull_<V>_s<S>.npz).
Per family: mean raw test prediction over the sources, then Charlie's residual correction with the mean inner-OOF
residuals of the replica full refits full_s0..2 (all-train labels only; test labels unknown)."""
import sys
sys.dont_write_bytecode = True
import os
import numpy as np
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/r3_chp')
import harness as H
from corrlib import Geo, correction
CACHE = WORK + '/r3_chp/cache'
tr, te, y = H.load()
FULL = [dict(np.load(f'{CACHE}/full_s{s}.npz')) for s in range(3) if os.path.exists(f'{CACHE}/full_s{s}.npz')]
GF = Geo(tr)
PT = GF.pairs(te)
CORR0 = {'cat': (0.35, 0.75, 10.0), 'et': (0.15, 3.0, 3.0)}
# self-check: the cached corrected full predictions are reproduced with each seed's own inner OOF
for d in FULL:
    for tag in ('cat', 'et'):
        a, bw, sh = CORR0[tag]
        v = np.clip(d[f'{tag}_raw_t'] + a * correction(PT, GF.ref_resid(y - d[f'{tag}_ip']), bw, sh), 0, 1)
        assert np.abs(v - d[f'{tag}_t']).max() < 1e-9
IP = {tag: np.mean([d[f'{tag}_ip'] for d in FULL], 0) for tag in ('cat', 'et')}


def raw(src, tag):
    p = src.split(':')
    if p[0] == 'full':
        return dict(np.load(f'{CACHE}/full_s{p[1]}.npz'))[f'{tag}_raw_t']
    if p[0] == 'sd':
        d = dict(np.load(f'{CACHE}/sdfull_s{p[1]}_i{p[2]}_te{p[3]}.npz'))
        return d[f'cat_t_{p[4] if len(p) > 4 else 2000}'] if tag == 'cat' else d['et_t']
    if p[0] == 'var':
        d = dict(np.load(f'{CACHE}/varfull_{p[1]}_s{p[2]}.npz'))
        if tag == 'cat':
            return d[f'cat_t_{p[3] if len(p) > 3 else 2000}']
        return d['et_t'] if 'et_t' in d else None
    raise ValueError(src)


def fam_full(srcs, tag, corr=None, fallback_et=None):
    rs = [raw(s, tag) for s in srcs]
    rs = [r for r in rs if r is not None] or [raw(s, tag) for s in fallback_et]
    t = np.mean(rs, 0)
    a, bw, sh = CORR0[tag] if corr is None else corr
    if a == 0:
        return t
    return np.clip(t + a * correction(PT, GF.ref_resid(y - IP[tag]), bw, sh), 0, 1)


def blend_full(cat_srcs, et_srcs, w=0.4, ccorr=None, ecorr=None):
    c = fam_full(cat_srcs, 'cat', ccorr)
    e = fam_full(et_srcs, 'et', ecorr, fallback_et=['full:0', 'full:1', 'full:2'])
    return (1 - w) * c + w * e

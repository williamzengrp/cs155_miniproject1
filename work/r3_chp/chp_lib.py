"""r3_chp: evaluation helpers on cached fold predictions (replica seed-0 cache + extra seeds + outer-only variants).
Residual correction = Charlie's same-date audio kernel on the replica's inner-OOF residuals of the same fold (fit-row
labels only), via the cached geometry (geo_std.pkl, verified to reproduce the replica's corrected preds exactly)."""
import sys
sys.dont_write_bytecode = True
import os, pickle
import numpy as np
from sklearn.metrics import roc_auc_score
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/r3_chp')
import harness as H
from corrlib import correction
CACHE = WORK + '/r3_chp/cache'
tr, te, y = H.load()
n = len(tr)
splits = H.make_splits(tr, y, n_rep=5)
G = pickle.load(open(CACHE + '/geo_std.pkl', 'rb'))
C = [dict(np.load(f'{WORK}/ch/cache/chfull_std_k{k}.npz')) for k in range(25)]
CORR0 = {'cat': (0.35, 0.75, 10.0), 'et': (0.15, 3.0, 3.0)}


def _rref(k, tag):
    g = G[k]; fit = splits[k][1]
    res = y[fit] - C[k][f'{tag}_ip']
    cnt = np.bincount(g['codes'], minlength=g['nref'])
    return np.bincount(g['codes'], weights=res, minlength=g['nref']) / cnt


RR = {(k, t): _rref(k, t) for k in range(25) for t in ('cat', 'et')}
_NPZ = {}


def _load(path):
    if path not in _NPZ:
        _NPZ[path] = dict(np.load(path)) if os.path.exists(path) else None
    return _NPZ[path]


def src_pred(src, tag, k):
    """src: 'rep' | 'sd:SOFF:ITERS:TEV[:IT]' | 'var:VARIANT:SOFF[:IT]' -> (raw_v, raw_t) or None if missing."""
    p = src.split(':')
    if p[0] == 'rep':
        return C[k][f'{tag}_raw_v'], C[k][f'{tag}_raw_t']
    if p[0] == 'sd':
        d = _load(f'{CACHE}/sd_o{p[1]}_i{p[2]}_te{p[3]}_k{k}.npz')
        if d is None:
            return None
        if tag == 'cat':
            it = p[4] if len(p) > 4 else '2000'
            return d[f'cat_v_{it}'], d[f'cat_t_{it}']
        return d['et_v'], d['et_t']
    if p[0] == 'var':
        d = _load(f'{CACHE}/var_{p[1]}_o{p[2]}_k{k}.npz')
        if d is None:
            return None
        if tag == 'cat':
            it = p[3] if len(p) > 3 else '2000'
            return d[f'cat_v_{it}'], d[f'cat_t_{it}']
        if 'et_v' in d:
            return d['et_v'], d['et_t']
        return C[k]['et_raw_v'], C[k]['et_raw_t']
    raise ValueError(src)


def fam(srcs, tag, k, corr=None):
    """Average raw preds of the sources, then the residual correction (corr=(alpha,bw,shrink); None -> Charlie's)."""
    ps = [src_pred(s, tag, k) for s in srcs]
    if any(p is None for p in ps):
        return None
    v = np.mean([p[0] for p in ps], 0); t = np.mean([p[1] for p in ps], 0)
    a, bw, sh = CORR0[tag] if corr is None else corr
    if a == 0:
        return v, t
    rr = RR[(k, tag)]
    return (np.clip(v + a * correction(G[k]['pv'], rr, bw, sh), 0, 1),
            np.clip(t + a * correction(G[k]['pt'], rr, bw, sh), 0, 1))


def blend(cat_srcs, et_srcs, k, w=0.4, ccorr=None, ecorr=None):
    c = fam(cat_srcs, 'cat', k, ccorr)
    e = fam(et_srcs, 'et', k, ecorr)
    if c is None or e is None:
        return None
    return (1 - w) * c[0] + w * e[0], (1 - w) * c[1] + w * e[1]


def assemble(fn):
    """fn(k) -> (val_pred, test_pred) or None. Returns oof (5 x n, NaN for missing repeats), aucs (NaN for missing
    repeats), test_foldavg (over available folds), list of complete repeats."""
    oof = np.full((5, n), np.nan)
    tsum, nt, have = 0, 0, []
    for r in range(5):
        preds = [fn(k) for k in range(5 * r, 5 * r + 5)]
        if any(p is None for p in preds):
            continue
        have.append(r)
        for k, p in zip(range(5 * r, 5 * r + 5), preds):
            _, fit, val, elig = splits[k]
            oof[r, val[elig]] = p[0][elig]
            tsum = tsum + p[1]; nt += 1
    aucs = np.full(5, np.nan)
    for r in have:
        mk = ~np.isnan(oof[r])
        aucs[r] = roc_auc_score(y[mk], oof[r, mk])
    return oof, aucs, (tsum / nt if nt else None), have


REF_AUCS = H.load_res('chrep_bl')['aucs']


def report(name, aucs, have, ref=REF_AUCS):
    d = aucs[have] - ref[have]
    print(f'{name:58s} AUC {np.nanmean(aucs):.5f} (reps {have})  vs chrep_bl {d.mean():+.5f}  '
          f'per-rep {np.round(d, 4)}  pos {(d > 0).sum()}/{len(d)}', flush=True)
    return d.mean()


def save(name, oof, aucs, tfa, test_full):
    assert name.startswith('r3_chp_')
    assert test_full is not None and np.size(test_full) == 863 and np.isfinite(test_full).all()
    H.save(name, dict(oof=oof, aucs=aucs, test_foldavg=tfa), test_full)

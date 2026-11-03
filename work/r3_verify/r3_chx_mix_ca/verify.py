"""Leakage audit for results/r3_chx_mix_ca (0.6 CatBoost variant c + 0.4 ExtraTrees variant a, r3_chx/chx_lib.py).
Re-runs ONE outer fold of the exact r3_chx code path (ReplicaX.matrices / inner_sets / run_family / correct) with
modified labels.  Modes:
  et    K THREADS : ET family, variant a.  (1) all VALIDATION labels flipped -> must equal cached a_k / et_a_k file
                    (2) FIT labels permuted -> validation AUC (eligible rows) should be ~0.5
  catsm K THREADS ROUNDS : CatBoost family, variant c, reduced iterations; baseline vs val-flipped (exact equality
                    expected) and fit-permuted AUC
  catfull K THREADS : CatBoost family, variant c, full 2000 it, val-flipped -> compare with cached c_k file
Writes nothing outside this directory.
"""
import sys
sys.dont_write_bytecode = True
import os, time, warnings
import numpy as np
import pandas as pd
WORK = './work'
HERE = WORK + '/r3_chx'
OUT = WORK + '/r3_verify/r3_chx_mix_ca'
for p in (HERE, WORK, WORK + '/ch', WORK + '/main', WORK + '/te'):    # same order as chx.py
    sys.path.insert(0, p)
import harness as H
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
import chx_lib
from chx_lib import ReplicaX, SEED0
from ch_common import connected_groups
from ch_feats import charlie_pool
from label_features import get_ctx

mode, k, threads = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
t00 = time.time()
tr, te, y = H.load()
n, m = len(tr), len(te)
splits = H.make_splits(tr, y, n_rep=5)
r, fit, val, elig = splits[k]
S = SEED0 + k
F, BL = charlie_pool(tr, te)
Fp = F[BL['PROJECT']]
groups = connected_groups(tr)
XA_dummy = pd.DataFrame(index=range(n + m))     # label_features only asserts len(XA) == n + m
print(f'[{mode} k={k}] n={n} m={m} fit={len(fit)} val={len(val)} elig={elig.sum()}  setup {time.time() - t00:.1f}s',
      flush=True)


def run(variant, fam, yy):
    t0 = time.time()
    alb = get_ctx(tr, te).key['alb'] if 'c' in variant else None
    R = ReplicaX(tr, te, yy, Fp, groups, variant, threads=threads,
                 XA=XA_dummy if 'a' in variant else None, CD=None, alb=alb)
    Xf, (Xv,), Xt = R.matrices(fit, [val], True, lf_seed=k)
    inner = R.inner_sets(fit, S)
    (rv,), rt, ip = R.run_family(fam, fit, Xf, [Xv], [val], Xt, inner, S)
    out = dict(raw_v=rv, ip=ip, v=R.correct(fam, fit, ip, rv, tr.iloc[val]), raw_t=rt,
               t=R.correct(fam, fit, ip, rt, te), ncol=Xf.shape[1])
    print(f'   run {variant}/{fam} ncol {Xf.shape[1]} {time.time() - t0:.1f}s', flush=True)
    return out


def cmp(tag, a, b, keys=('raw_v', 'v', 'ip', 'raw_t', 't')):
    for key in keys:
        d = np.max(np.abs(np.asarray(a[key], float) - np.asarray(b[key], float)))
        print(f'   {tag:28s} {key:6s} max|diff| = {d:.3e}', flush=True)


def auc_e(p):
    return roc_auc_score(y[val[elig]], np.asarray(p)[elig])


y_flip = y.copy()
y_flip[val] = 1 - y_flip[val]                       # every validation label flipped; fit labels untouched
assert (y_flip[fit] == y[fit]).all()
y_perm = y.copy()
y_perm[fit] = np.random.default_rng(12345).permutation(y[fit])   # fit labels permuted; val labels untouched

if mode == 'et':
    p = f'{HERE}/cache/et_a_k{k}.npz'
    p = p if os.path.exists(p) else f'{HERE}/cache/a_k{k}.npz'
    c = np.load(p)
    cache = dict(raw_v=c['et_raw_v'], v=c['et_v'], ip=c['et_ip'], raw_t=c['et_raw_t'], t=c['et_t'])
    print(f'   cache file {os.path.basename(p)}; cached val AUC(elig) {auc_e(cache["v"]):.5f}', flush=True)
    fl = run('a', 'extra', y_flip)
    cmp('val-flipped vs cache', fl, cache)
    print(f'   val-flipped AUC(elig, true labels) {auc_e(fl["v"]):.5f}', flush=True)
    pm = run('a', 'extra', y_perm)
    print(f'   FIT-PERMUTED AUC(elig): corrected {auc_e(pm["v"]):.5f}  raw {auc_e(pm["raw_v"]):.5f}', flush=True)
    np.savez(f'{OUT}/et_k{k}.npz', **{f'fl_{a}': b for a, b in fl.items()}, **{f'pm_{a}': b for a, b in pm.items()})
elif mode == 'catsm':
    rounds = int(sys.argv[4])
    chx_lib.CFG['cat']['rounds'] = rounds            # same code path, fewer boosting rounds (speed only)
    base = run('c', 'cat', y)
    fl = run('c', 'cat', y_flip)
    cmp(f'cat{rounds} val-flipped vs base', fl, base)
    print(f'   base AUC(elig) {auc_e(base["v"]):.5f}  val-flipped AUC(elig) {auc_e(fl["v"]):.5f}', flush=True)
    pm = run('c', 'cat', y_perm)
    print(f'   FIT-PERMUTED AUC(elig): corrected {auc_e(pm["v"]):.5f}  raw {auc_e(pm["raw_v"]):.5f}', flush=True)
    np.savez(f'{OUT}/catsm_k{k}.npz', **{f'base_{a}': b for a, b in base.items()},
             **{f'fl_{a}': b for a, b in fl.items()}, **{f'pm_{a}': b for a, b in pm.items()})
elif mode == 'catfull':
    c = np.load(f'{HERE}/cache/c_k{k}.npz')
    cache = dict(raw_v=c['cat_raw_v'], v=c['cat_v'], ip=c['cat_ip'], raw_t=c['cat_raw_t'], t=c['cat_t'])
    print(f'   cached cat val AUC(elig) {auc_e(cache["v"]):.5f}', flush=True)
    fl = run('c', 'cat', y_flip)
    cmp('cat2000 val-flipped vs cache', fl, cache)
    print(f'   val-flipped AUC(elig) {auc_e(fl["v"]):.5f}', flush=True)
    np.savez(f'{OUT}/catfull_k{k}.npz', **{f'fl_{a}': b for a, b in fl.items()})
print(f'[{mode} k={k}] done {time.time() - t00:.1f}s', flush=True)

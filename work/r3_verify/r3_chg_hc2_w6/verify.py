"""Auditor script for r3_chg_hc2_w6. usage: python -I -B verify.py {valflip|fitperm} K
valflip: flip VALIDATION labels (y and tr.Popularity_Type) of fold K, keep fit labels; check every model input is
         bit-identical to the true-label build, then run the full fold pipeline and compare with the cached fold.
fitperm: permute FIT labels of fold K (val labels true); run full fold pipeline; AUC on eligible val rows should be ~0.5."""
import sys
sys.dont_write_bytecode = True
mode, K = sys.argv[1], int(sys.argv[2]); PSEED = int(sys.argv[3]) if len(sys.argv) > 3 else 12345
sys.argv = [sys.argv[0], 'x']          # chg.py reads sys.argv at import -> THREADS = 1
import os
os.environ['OMP_NUM_THREADS'] = '1'
WORK = './work'
OUT = WORK + '/r3_verify/r3_chg_hc2_w6'
sys.path.insert(0, WORK + '/r3_chg')
import time
import numpy as np
from sklearn.metrics import roc_auc_score
import chg
import harness as H
from chg import build, fam_run, SEED0

t00 = time.time()
tr, te, y = H.load()
splits = H.make_splits(tr, y, n_rep=5)
r, fit, val, elig = splits[K]
S = SEED0 + K
lab_col = 'Popularity_Type'

def variant(yv):
    t2 = tr.copy()
    t2[lab_col] = np.where(yv == 1, 'High', 'Low')
    return t2, yv

def run_fold(R, trd):
    Xf, (Xv,), Xt = R.matrices(fit, [val], True, gseed=K)
    inner = R.inner_sets(fit, S, gseed=K)
    out = {}
    for fam, ft in (('cat', 'cat'), ('extra', 'et')):
        ta = time.time()
        (rv,), rt, ip = fam_run(R, fam, fit, Xf, [Xv], Xt, inner, S, outer=True)
        out[f'{ft}_raw_v'], out[f'{ft}_raw_t'], out[f'{ft}_ip'] = rv, rt, ip
        out[f'{ft}_v'] = R.correct(fam, fit, ip, rv, trd.iloc[val])
        out[f'{ft}_t'] = R.correct(fam, fit, ip, rt, te)
        print(f'  {fam} done {time.time() - ta:.1f}s', flush=True)
    return out, Xv

def aucs(out, ytrue):
    e = elig
    res = {}
    for key in ('cat_raw_v', 'et_raw_v', 'cat_v', 'et_v'):
        res[key] = roc_auc_score(ytrue[val][e], out[key][e])
    b = 0.4 * out['cat_v'] + 0.6 * out['et_v']
    res['w6_blend'] = roc_auc_score(ytrue[val][e], b[e])
    return res

cache = dict(np.load(f'{WORK}/r3_chg/cache/hc2_k{K}.npz'))
print(f'fold {K}: n_fit {len(fit)} n_val {len(val)} n_elig {elig.sum()}', flush=True)
print('cached fold AUCs (true labels):', {k: round(v, 5) for k, v in aucs(cache, y).items()}, flush=True)

if mode == 'valflip':
    y2 = y.copy()
    y2[val] = 1 - y2[val]
    tr2, y2 = variant(y2)
    R1 = build(tr, te, y, 'hc2', 1)
    R2 = build(tr2, te, y2, 'hc2', 1)
    assert np.array_equal(R1.Ptr, R2.Ptr, equal_nan=True) and np.array_equal(R1.Pte, R2.Pte, equal_nan=True)
    assert np.array_equal(R1.groups, R2.groups)
    A1 = R1.matrices(fit, [val], True, gseed=K)
    A2 = R2.matrices(fit, [val], True, gseed=K)
    for a, b, nm in ((A1[0], A2[0], 'Xf'), (A1[1][0], A2[1][0], 'Xv'), (A1[2], A2[2], 'Xt')):
        print(f'  input {nm} shape {a.shape} identical: {np.array_equal(a, b)}', flush=True)
    I1 = R1.inner_sets(fit, S, gseed=K)
    I2 = R2.inner_sets(fit, S, gseed=K)
    ok = all(np.array_equal(p[0], q[0]) and np.array_equal(p[1], q[1]) and np.array_equal(p[2], q[2])
             and np.array_equal(p[3], q[3]) for p, q in zip(I1, I2))
    print(f'  inner sets (it, iv, I, J) identical: {ok}', flush=True)
    print('  y[fit] identical:', np.array_equal(y[fit], y2[fit]), ' val labels differ:', (y[val] != y2[val]).mean(), flush=True)
    # graph column for val rows: confirm eligible val nodes are unseeded
    lab, _, _ = R1.G.node_seeds(y, fit)
    print('  eligible val nodes seeded (must be 0):', int(lab[R1.G.node[val[elig]]].sum()),
          ' non-eligible val nodes seeded:', int(lab[R1.G.node[val[~elig]]].sum()), 'of', int((~elig).sum()), flush=True)
    gcol = A1[1][0][:, -1]
    print('  standalone g_hc2 val AUC (true labels, eligible):', round(roc_auc_score(y[val][elig], gcol[elig]), 5), flush=True)
    del R1
    out, Xv = run_fold(R2, tr2)
    np.savez(f'{OUT}/valflip_k{K}.npz', **out)
    for key in ('cat_raw_v', 'et_raw_v', 'cat_ip', 'et_ip', 'cat_v', 'et_v', 'cat_raw_t', 'et_raw_t', 'cat_t', 'et_t'):
        d = np.abs(out[key] - cache[key])
        print(f'  {key:10s} max|diff| vs cached true-label fold: {d.max():.3e}  identical: {np.array_equal(out[key], cache[key])}', flush=True)
    print('valflip AUCs (true labels):', {k: round(v, 5) for k, v in aucs(out, y).items()}, flush=True)
elif mode == 'fitperm':
    rng = np.random.default_rng(PSEED)
    y3 = y.copy()
    y3[fit] = rng.permutation(y[fit])
    tr3, y3 = variant(y3)
    print('  frac fit labels changed:', (y3[fit] != y[fit]).mean(), ' val labels unchanged:', np.array_equal(y3[val], y[val]), flush=True)
    R3 = build(tr3, te, y3, 'hc2', 1)
    out, Xv = run_fold(R3, tr3)
    np.savez(f'{OUT}/fitperm_k{K}_s{PSEED}.npz', **out)
    print('  standalone g_hc2 val AUC under fit-perm:', round(roc_auc_score(y[val][elig], Xv[elig, -1]), 5),
          ' date-TE cols:', [round(roc_auc_score(y[val][elig], Xv[elig, j]), 4) for j in range(115, 118)], flush=True)
    print('fitperm AUCs (TRUE val labels):', {k: round(v, 5) for k, v in aucs(out, y).items()}, flush=True)
print(f'total {time.time() - t00:.1f}s', flush=True)

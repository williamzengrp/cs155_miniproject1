"""r3_chp: re-tune the residual correction (per family) and the Cat/ET weight for a given set of cached sources.
python -I -B tune_var.py CAT_SRCS ET_SRCS     (comma-separated chp_lib sources, e.g. var:lab:0:2000 var:lab:0)
Selection = mean AUC over the complete repeats; prints per-repeat deltas vs Charlie's correction settings."""
import sys
sys.dont_write_bytecode = True
import itertools
import numpy as np
sys.path.insert(0, './work/r3_chp')
import chp_lib as L
from corrlib import correction

cat_s, et_s = sys.argv[1].split(','), sys.argv[2].split(',')
RAW = {}
for tag, srcs in (('cat', cat_s), ('et', et_s)):
    for k in range(25):
        ps = [L.src_pred(s, tag, k) for s in srcs]
        RAW[(tag, k)] = None if any(p is None for p in ps) else np.mean([p[0] for p in ps], 0)
have_k = [k for k in range(25) if RAW[('cat', k)] is not None and RAW[('et', k)] is not None]
reps = sorted({k // 5 for k in have_k if all(j in have_k for j in range(5 * (k // 5), 5 * (k // 5) + 5))})
K = [k for k in have_k if k // 5 in reps]
print('repeats', reps)


def fam(tag, a, bw, sh):
    return {k: (RAW[(tag, k)] if a == 0 else
                np.clip(RAW[(tag, k)] + a * correction(L.G[k]['pv'], L.RR[(k, tag)], bw, sh), 0, 1)) for k in K}


def aucs(v):
    oof = np.full((5, L.n), np.nan)
    for k in K:
        _, fit, val, elig = L.splits[k]
        oof[k // 5, val[elig]] = v[k][elig]
    return np.array([L.roc_auc_score(L.y[~np.isnan(oof[r])], oof[r][~np.isnan(oof[r])]) for r in reps])


mix = lambda c, e, w: {k: (1 - w) * c[k] + w * e[k] for k in K}
c0, e0 = fam('cat', .35, .75, 10.), fam('et', .15, 3., 3.)
b0 = aucs(mix(c0, e0, .4))
print(f'charlie settings: {b0.mean():.5f}  per-rep {np.round(b0, 4)}')
cr, er = fam('cat', 0, 1, 1), fam('et', 0, 1, 1)
print(f'no correction: {aucs(mix(cr, er, .4)).mean():.5f}')
A = [0.0, 0.1, 0.2, 0.35, 0.5, 0.7]; BW = [0.75, 1.0, 1.5, 2.0, 3.0]; SH = [1., 3., 10.]
best = {}
for tag in ('cat', 'et'):
    rows = []
    for a, bw, sh in itertools.product(A, BW, SH):
        if a == 0 and (bw, sh) != (1.0, 1.):
            continue
        v = fam(tag, a, bw, sh)
        ba = aucs(mix(v, e0, .4) if tag == 'cat' else mix(c0, v, .4))
        rows.append((ba.mean(), a, bw, sh, int((ba - b0 > 0).sum()), np.round(ba - b0, 4)))
    rows.sort(key=lambda r: -r[0])
    print(f'--- {tag} top (blend AUC, alpha, bw, shrink, #reps>charlie-settings, per-rep d)')
    for r in rows[:6]:
        print('   ', round(r[0], 5), r[1:])
    best[tag] = rows[0][1:4]
cb, eb = fam('cat', *best['cat']), fam('et', *best['et'])
for w in (0.3, 0.35, 0.4, 0.45, 0.5):
    a = aucs(mix(cb, eb, w))
    print(f'best corr cat{best["cat"]} et{best["et"]} w_et {w}: {a.mean():.5f}  d vs charlie-settings {np.round(a - b0, 4)}')

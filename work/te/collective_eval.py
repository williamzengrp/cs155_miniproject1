"""Offline evaluation of collective inference from a collective_cv.py pickle (agent te).

AUC = pooled over eligible validation rows per repeat (same as harness.run_cv); paired diffs are vs the
plain predictions p0 of the SAME models, so they are free of seed noise.
usage: python collective_eval.py <tag>
"""
import pickle
import sys

sys.path.insert(0, './work/te')
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from common import LF, y, tr, te, OUT  # noqa: E402

tag = sys.argv[1]
D = pickle.load(open(f'{OUT}/coll_{tag}.pkl', 'rb'))
folds = D['folds']
ctx = LF.get_ctx(tr, te)


def aucs(get):
    oof = np.full((5, len(y)), np.nan)
    for f in folds:
        q = get(f)
        oof[f['r'], f['U'][:f['nval']]] = q[:f['nval']]
    return np.array([roc_auc_score(y[~np.isnan(o)], o[~np.isnan(o)]) for o in oof])


base = aucs(lambda f: f['p0'])
print(f'[{tag}] spec={D["spec"]} seeds={len(D["offs"])}  p0 AUC {base.mean():.5f} {np.round(base, 4)}')


def show(name, a):
    d = a - base
    print(f'  {name:40s} AUC {a.mean():.5f}  diff {d.mean():+.5f}  per-rep {np.array2string(d, precision=4, sign="+")}'
          f'  pos {int((d > 0).sum())}/5', flush=True)


for key in [k for k in folds[0] if k.startswith('pl_')]:
    show('pseudo-label ' + key, aucs(lambda f: f[key]))
grid = [dict(alpha=a) for a in (0.05, 0.1, 0.2, 0.5)]
grid += [dict(alpha=a, bw=1.5) for a in (0.2, 0.5, 1.0)]
grid += [dict(alpha=a, unl_w=0.0) for a in (0.1, 0.2)]          # labelled mates only
grid += [dict(alpha=a, lab_w=0.0) for a in (0.1, 0.2)]          # unlabelled mates only (pure smoothing)
grid += [dict(alpha=a, n_iter=5) for a in (0.1, 0.2)]
for g in grid:
    show('smooth ' + str(g), aucs(lambda f: LF.smooth_dates(ctx, f['p0'], f['U'], f['fit'], y, **g)))

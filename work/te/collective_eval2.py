"""Fixed-weight collective smoothing: q = (1-lam) p + lam * mean(mates), mates = same-date rows (labels for
labelled, predictions for unlabelled), optional audio kernel. Rows without mates keep p. (agent te)
usage: python collective_eval2.py <tag>"""
import pickle
import sys

sys.path.insert(0, './work/te')
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from common import LF, y, tr, te, OUT  # noqa: E402

tag = sys.argv[1]
D = pickle.load(open(f'{OUT}/coll_{tag}.pkl', 'rb'))
ctx = LF.get_ctx(tr, te)


def mix(f, lam, bw=None, lab_w=1.0, unl_w=1.0):
    U, L, p = f['U'], f['fit'], f['p0']
    # mates' mean via smooth_dates with alpha -> inf is (S/W); compute directly from one Jacobi step
    big = LF.smooth_dates(ctx, p, U, L, y, alpha=1e9, bw=bw, lab_w=lab_w, unl_w=unl_w)
    has = np.abs(big - p) > 0           # rows with at least one mate (else smooth_dates returns p)
    q = p.copy()
    q[has] = (1 - lam) * p[has] + lam * big[has]
    return q


def aucs(get):
    oof = np.full((5, len(y)), np.nan)
    for f in D['folds']:
        q = get(f)
        oof[f['r'], f['U'][:f['nval']]] = q[:f['nval']]
    return np.array([roc_auc_score(y[~np.isnan(o)], o[~np.isnan(o)]) for o in oof])


base = aucs(lambda f: f['p0'])
print(f'[{tag}] p0 AUC {base.mean():.5f}')
for g in [dict(lam=0.05), dict(lam=0.1), dict(lam=0.2), dict(lam=0.1, bw=1.5), dict(lam=0.2, bw=1.5),
          dict(lam=0.1, unl_w=0.0), dict(lam=0.1, lab_w=0.0)]:
    a = aucs(lambda f: mix(f, **g))
    d = a - base
    print(f'  fixed-mix {str(g):34s} AUC {a.mean():.5f} diff {d.mean():+.5f} per-rep {np.array2string(d, precision=4, sign="+")} pos {int((d > 0).sum())}/5', flush=True)

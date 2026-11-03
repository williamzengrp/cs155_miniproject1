"""Collective / transductive inference CV driver (agent te).

Per outer fold: L = fit rows (labelled), U = eligible validation rows + ALL real test rows (unlabelled),
exactly the situation at test time (there L = all train, U = test). The model is trained once on L
(label features out-of-fold), predicts p0 on U; then pseudo-label rounds recompute the label features
of U using sources L (labels, weight 1) + U (current predictions, weight beta) and re-predict with the
SAME model. Everything is saved so post-hoc smoothing can be evaluated offline on identical predictions.

usage: python collective_cv.py <tag> <seeds:int> [--base v2] [spec items...]
"""
import pickle
import sys
import time

sys.path.insert(0, './work/te')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import LF, y, tr, te, n, m, splits, TEST, OUT, SEED_OFFS, lgbm, matrices, base_frame  # noqa: E402

argv = sys.argv[1:]
base = '67'
if '--base' in argv:
    i = argv.index('--base'); base = argv[i + 1]; del argv[i:i + 2]
tag, nseeds, spec = argv[0], int(argv[1]), argv[2:]
XB = base_frame(base)
offs = SEED_OFFS[:nseeds]
ctx = LF.get_ctx(tr, te)
BETAS = (0.5, 1.0)
folds = []
t0 = time.time()
for k, (r, fit_idx, val_idx, elig) in enumerate(splits):
    U = np.concatenate([val_idx[elig], TEST])
    Xf, (Xu,) = matrices(fit_idx, [U], spec, seed=k, base=base)
    models = [lgbm(k + off).fit(Xf, y[fit_idx]) for off in offs]
    pred = lambda Xm: np.mean([mm.predict_proba(Xm)[:, 1] for mm in models], axis=0)  # noqa: E731
    rec = dict(r=r, fit=fit_idx, U=U, nval=int(elig.sum()), p0=pred(Xu))
    if spec:
        base_u = XB.iloc[U].reset_index(drop=True)
        for beta in BETAS:
            p = rec['p0']
            for rnd in (1, 2):
                vals = np.zeros(ctx.N); vals[:n] = y; vals[U] = p
                wts = np.zeros(ctx.N); wts[fit_idx] = 1.0; wts[U] = beta
                Fu = LF.encode(ctx, vals, wts, U, spec)
                p = pred(pd.concat([base_u, Fu], axis=1)[Xf.columns])
                rec[f'pl_b{beta}_r{rnd}'] = p
    folds.append(rec)
    if k % 5 == 4:
        print(f'fold {k + 1}/25  {time.time() - t0:.0f}s', flush=True)
pickle.dump(dict(spec=spec, offs=offs, base=base, folds=folds), open(f'{OUT}/coll_{tag}.pkl', 'wb'))
print('saved', f'{OUT}/coll_{tag}.pkl')

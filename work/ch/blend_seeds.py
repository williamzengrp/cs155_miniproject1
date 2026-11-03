"""ch agent: seed robustness of the charlie-view models' blend gain. For each seed set s (0/100/200) of the ADDED
model(s), gain = equal-weight rank blend(v3 5 + added) - blend(v3 5), per repeat; then averaged over seed sets."""
import sys
import numpy as np
WORK = './work'
sys.path.insert(0, WORK)
import harness as H
from blend import evaluate
tr, te, y = H.load()
V3 = [f'v3_{m}' for m in ('lgbm', 'xgb', 'cat', 'et', 'rf')]
base = evaluate(V3, y, nested=False, verbose=False)['equal']
nm = lambda b, s: b if s == 0 else f'{b}_off{s}'
for combo in [['ch_et_proj'], ['ch_lgbm_proj'], ['ch_et_proj', 'ch_lgbm_proj']]:
    D, singles = [], []
    for s in (0, 100, 200):
        names = [nm(b, s) for b in combo]
        D.append(evaluate(V3 + names, y, nested=False, verbose=False)['equal'] - base)
        singles.append([H.load_res(n)['aucs'].mean() for n in names])
    D = np.array(D)
    d = D.mean(0)
    print(f'| +{"+".join(combo)} eq | standalone per seed {np.round(np.array(singles).T, 4).tolist()} | '
          f'{base.mean() + d.mean():.5f} | {d.mean():+.5f} | {(d > 0).sum()}/5 | {" ".join(f"{x:+.4f}" for x in d)} | '
          f'per-seed-set {" ".join(f"{x:+.4f}" for x in D.mean(1))} | min single (seed,rep) {D.min():+.4f} |', flush=True)

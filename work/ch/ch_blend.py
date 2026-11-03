"""ch agent: marginal blend value of a candidate vs the 5 v3_* models (equal-weight rank blend, blend.evaluate).
usage: python -I ch_blend.py add NAME            -> 5 v3 + NAME at equal 1/6, and at fixed 0.1 / 0.2 / 0.3
       python -I ch_blend.py swap NAME V3MODEL   -> replace v3_<V3MODEL> by NAME (equal weights)
"""
import sys
import numpy as np
WORK = './work'
sys.path.insert(0, WORK)
import harness as H
from blend import evaluate

tr, te, y = H.load()
V3 = [f'v3_{m}' for m in ('lgbm', 'xgb', 'cat', 'et', 'rf')]
base = evaluate(V3, y, nested=False, verbose=False)['equal']
print(f'5 v3 equal rank blend: {base.mean():.5f} per-rep {np.round(base, 4)}')


def rep(tag, arr):
    d = arr - base
    print(f'| {tag} | {arr.mean():.5f} | {d.mean():+.5f} | {(d > 0).sum()}/5 | {" ".join(f"{x:+.4f}" for x in d)} |')


mode, name = sys.argv[1], sys.argv[2]
if mode == 'add':
    names = V3 + [name]
    rep(f'+{name} eq 1/6', evaluate(names, y, nested=False, verbose=False)['equal'])
    for w in (0.1, 0.2, 0.3):
        wt = [(1 - w) / 5] * 5 + [w]
        rep(f'+{name} w={w}', evaluate(names, y, weights=wt, nested=False, verbose=False)['fixed'])
else:
    names = [name if n == f'v3_{sys.argv[3]}' else n for n in V3]
    rep(f'swap v3_{sys.argv[3]} -> {name}', evaluate(names, y, nested=False, verbose=False)['equal'])

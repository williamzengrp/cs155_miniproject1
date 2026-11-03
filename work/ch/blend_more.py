"""ch agent: controls for the charlie-view blend gain (equal-weight rank blends, paired per repeat vs v3 5)."""
import sys
import numpy as np
WORK = './work'
sys.path.insert(0, WORK)
import harness as H
from blend import evaluate
tr, te, y = H.load()
V3 = [f'v3_{m}' for m in ('lgbm', 'xgb', 'cat', 'et', 'rf')]
base = evaluate(V3, y, nested=False, verbose=False)['equal']
def rep(tag, names, w=None):
    o = evaluate(names, y, weights=w, nested=False, verbose=False)
    a = o['fixed'] if w is not None else o['equal']
    d = a - base
    print(f'| {tag} | {a.mean():.5f} | {d.mean():+.5f} | {(d > 0).sum()}/5 | {" ".join(f"{x:+.4f}" for x in d)} |', flush=True)
for extra in sys.argv[1:]:
    names = V3 + extra.split(',')
    k = len(extra.split(','))
    rep(f'+{extra} (eq 1/{5+k})', names)
    if k > 1:
        rep(f'+{extra} (0.1 each)', names, [(1 - 0.1 * k) / 5] * 5 + [0.1] * k)

"""r3_chp: build + cache correction geometry for the 25 std folds (val + test queries) and verify that it
reproduces the cached corrected predictions of the replica exactly."""
import sys
sys.dont_write_bytecode = True
import os, pickle, time
import numpy as np
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/r3_chp')
import harness as H
from corrlib import Geo, correction
CFG = {'cat': (0.35, 0.75, 10.0), 'et': (0.15, 3.0, 3.0)}
tr, te, y = H.load()
splits = H.make_splits(tr, y, n_rep=5)
out = []
t0 = time.time()
maxdiff = 0
for k, (r, fit, val, elig) in enumerate(splits):
    c = dict(np.load(f'{WORK}/ch/cache/chfull_std_k{k}.npz'))
    g = Geo(tr.iloc[fit])
    pv, pt = g.pairs(tr.iloc[val]), g.pairs(te)
    for tag in ('cat', 'et'):
        a, bw, sh = CFG[tag]
        rref = g.ref_resid(y[fit] - c[f'{tag}_ip'])
        v = np.clip(c[f'{tag}_raw_v'] + a * correction(pv, rref, bw, sh), 0, 1)
        t = np.clip(c[f'{tag}_raw_t'] + a * correction(pt, rref, bw, sh), 0, 1)
        maxdiff = max(maxdiff, np.abs(v - c[f'{tag}_v']).max(), np.abs(t - c[f'{tag}_t']).max())
    out.append(dict(codes=g.codes, nref=g.nref, pv=pv, pt=pt))
print('max abs diff vs cached corrected preds:', maxdiff, f'{time.time() - t0:.1f}s')
pickle.dump(out, open(WORK + '/r3_chp/cache/geo_std.pkl', 'wb'))

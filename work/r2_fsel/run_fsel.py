"""fsel agent runner: tree model(s) on feature SUBSETS / random subspaces, any split set.
python -I run_fsel.py SPEC SEED0 NREP PREFIX [DO_FULL=1] [NSEED=1]
  -> results/<PREFIX>_<SPEC>.npz  (OOF on eligible rows, fold-avg test, test_full = full refit on all
     training rows averaged over 3 label-feature/model seeds)
SPEC is defined in specs.py: (model, list of column lists [one list = plain model; several = random-subspace
member whose prediction is the mean probability of the sub-models], params override or None).
Sub-model j in fold k uses model seed k + 100*j (+1000*s for extra seeds s < NSEED, averaged).
Label features: per-fold out-of-fold features from build_cache.py (identical to main/final_cv.py)."""
import sys, time
import numpy as np
D = './work/r2_fsel'
sys.path.insert(0, D)
import fsel_lib as L
from specs import SPECS
spec, SEED0, NREP, PREFIX = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
DO_FULL = (sys.argv[5] == '1') if len(sys.argv) > 5 else True
NSEED = int(sys.argv[6]) if len(sys.argv) > 6 else 1
assert SEED0 != 11000 or PREFIX.startswith('c11'), 'c11 reserved for verification'
import os
if not os.path.exists(L.C + f'lf_{SEED0}_{NREP}.pkl'):
    os.system(f'{sys.executable} -I {D}/build_cache.py {SEED0} {NREP}')
model, colsets, params = SPECS[spec]
folds = L.load_folds(SEED0, NREP)
t0 = time.time()
oof, tst = 0, 0
for s in range(NSEED):
    for j, cols in enumerate(colsets):
        o, t = L.run_cv(model, cols, folds, seed_off=100 * j + 1000 * s, params=params)
        oof, tst = oof + o / (len(colsets) * NSEED), tst + t / (len(colsets) * NSEED)
full = None
if DO_FULL:
    full = sum(L.run_full(model, cols, L.load_full(), params=params, seed_off=100 * j)
               for j, cols in enumerate(colsets)) / len(colsets)
a = L.save(f'{PREFIX}_{spec}', oof, tst, full)
print(f'[{PREFIX}_{spec}] AUC {a.mean():.5f} per-rep {np.round(a, 4)}  ({time.time()-t0:.0f}s)', flush=True)

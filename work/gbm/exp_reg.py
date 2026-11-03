"""Item 5: monotone constraints / extra regularisers on top of the recommended params (5x5 eval splits).

usage: python -I exp_reg.py <model> <scheme> [seed0]
Each variant is a single change to the recommended params; paired per-repeat diff vs the reference.
"""
import sys
sys.path.insert(0, './work/gbm')
import numpy as np
import lib as L

name, scheme = sys.argv[1], sys.argv[2]
seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
only = sys.argv[4].split(',') if len(sys.argv) > 4 else None
P = L.M.load_params()[name]['params']
cols = list(L.X.columns)
# High rate falls with release time: newer -> lower.  +1 = increasing in the feature.
MONO = {'release_year': -1, 't_frac_year': -1, 't_decade': -1, 'days_since_release': +1, 't_log_age': +1}
mono = [MONO.get(c, 0) for c in cols]

if name == 'lgbm':
    V = {
        'mono_time': dict(monotone_constraints=mono, monotone_constraints_method='advanced'),
        'extra_trees': dict(extra_trees=True),
        'path_smooth10': dict(path_smooth=10.0),
        'path_smooth50': dict(path_smooth=50.0),
        'max_bin63': dict(max_bin=63),
        'bynode0.5': dict(feature_fraction_bynode=0.5),
    }
elif name == 'xgb':
    V = {
        'mono_time': dict(monotone_constraints=tuple(mono)),
        'max_bin64': dict(max_bin=64),
        'bynode0.5': dict(colsample_bynode=0.5),
    }
elif name == 'cat':
    V = {
        'mono_time': dict(monotone_constraints=mono),
        'border64': dict(border_count=64),
    }
else:
    V = {}
if only:
    V = {k: v for k, v in V.items() if k in only}
splits = L.H.make_splits(L.tr, L.y, n_rep=5, seed0=seed0)
ref = L.run(name, P, splits, scheme=scheme, verbose=False)
L.log(f'\n### regularisers: {name} (recommended params, scheme {scheme}, feats {L.M.FEATS}, splits seed0={seed0}) '
      f'ref AUC {ref["auc"]:.5f} {np.round(ref["aucs"], 4)}')
for k, extra in V.items():
    r = L.run(name, {**P, **extra}, splits, scheme=scheme, verbose=False)
    L.log(f'- {name} {k:14s} AUC {r["auc"]:.5f} {np.round(r["aucs"], 4)} vs ref {L.pdiff(r, ref)}')

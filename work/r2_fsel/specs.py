"""Feature-subset specs for run_fsel.py: name -> (model, [column list, ...], params override or None)."""
import numpy as np
import fsel_lib as L
G = L.GROUPS
SPECS = {}


def without(*groups, extra=()):
    drop = set(sum((G[g] for g in groups), [])) | set(extra)
    return [c for c in L.ALL if c not in drop]


def rsub(frac, K, seed, keep=('label',)):
    """K random subspaces with a fraction `frac` of the non-forced features (forced groups always kept)."""
    forced = sum((G[g] for g in keep), [])
    pool = [c for c in L.ALL if c not in forced]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(K):
        pick = rng.choice(len(pool), int(round(frac * len(pool))), replace=False)
        out.append(forced + [pool[i] for i in sorted(pick)])
    return out


for mdl in ('lgbm', 'xgb', 'cat', 'et', 'rf'):
    SPECS[f'{mdl}_all'] = (mdl, [L.ALL], None)

# ---- random-subspace members (pre-registered: K=8 sub-models, 60% of the 58 label-free features each,
#      the 3 label features always kept; subspace draw seed 0)
for mdl in ('lgbm', 'et', 'xgb', 'rf', 'cat'):
    SPECS[f'rs_{mdl}_k8f60'] = (mdl, rsub(0.6, 8, 0), None)

"""Assemble gbm/best_params_newcv.json from the selection/confirmation outputs + decisions below.

usage: python -I write_params.py   (light; no model fitting)
"""
import json
import os
import sys
import numpy as np

GBM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, GBM)
WORK = os.path.dirname(GBM)
BASE = json.load(open(os.path.join(WORK, 'best_params_v3.json')))

# decision per model: 'teammate' = best_params_v3.json, 'tuned' = robust pick from gbm/sel_<model>_none.json
# acceptance rule (brief): mean paired gain >= ~0.001 AND positive in >= 4/5 repeats on the 5x5 splits.
DECISION = {'lgbm': 'teammate', 'xgb': 'tuned', 'cat': 'tuned', 'et': 'teammate'}
N_SEEDS = {'lgbm': 5, 'xgb': 5, 'cat': 3, 'et': 3, 'rf': 3}
WEIGHTING = 'none'


def ev(path, a, b):
    if not os.path.exists(path):
        return None
    d = np.load(path)
    diff = d[a] - d[b]
    return dict(tuned=round(float(d[a].mean()), 5), teammate=round(float(d[b].mean()), 5),
                diff=round(float(diff.mean()), 5), pos=f'{int((diff > 0).sum())}/{len(diff)}')


out = {}
for name, dec in DECISION.items():
    sel_path = os.path.join(GBM, f'sel_{name}_none.json')
    sel = json.load(open(sel_path)) if os.path.exists(sel_path) else None
    params = dict(sel['params']) if dec == 'tuned' else dict(BASE[name])
    if name == 'cat':
        params.pop('bagging_temperature', None)     # ignored by CatBoost's default MVS bootstrap
    if name == 'lgbm':
        params.setdefault('subsample_freq', 1)
    out[name] = dict(
        params=params,
        source=('Optuna re-tune under test-mimicking CV, pick ' + sel['picked']) if dec == 'tuned'
        else 'teammates best_params_v3.json (re-tune not robustly better)',
        weighting=WEIGHTING, n_seeds=N_SEEDS[name],
        final_fit=f'refit on ALL training rows (no early stopping), mean of {N_SEEDS[name]} seeds',
        tuned_candidate=sel['params'] if sel else None,
        cv_5x5_v1_67feats=ev(os.path.join(GBM, f'eval_{name}_none.npz'), 'tuned_aucs', 'base_aucs'),
        cv_5x5_v2_55feats=ev(os.path.join(GBM, f'eval_{name}_none_v2.npz'), 'tuned_aucs', 'base_aucs'),
    )
json.dump(out, open(os.path.join(GBM, 'best_params_newcv.json'), 'w'), indent=1)
print(json.dumps({k: (v['source'], v['cv_5x5_v1_67feats'], v['cv_5x5_v2_55feats']) for k, v in out.items()}, indent=1))

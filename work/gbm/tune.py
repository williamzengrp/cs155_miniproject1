"""Item 1: modest Optuna re-tuning under the test-mimicking CV.

usage: python -I tune.py <model> <n_trials> [scheme]
Tuning splits: H.make_splits(tr, y, n_rep=2, seed0=7000) (disjoint seeds from the 5x5 evaluation splits).
The teammates' params are enqueued as trial 0 so their tuning-split score is known.
Study stored in gbm/optuna.db (resumable). Fixed-size models only (no early stopping).
"""
import sys
import os
sys.path.insert(0, './work/gbm')
import numpy as np
import optuna
import lib as L

optuna.logging.set_verbosity(optuna.logging.WARNING)
name, n_trials = sys.argv[1], int(sys.argv[2])
scheme = sys.argv[3] if len(sys.argv) > 3 else 'none'
FOREST_TREES = 500          # forests: tree count fixed while tuning (more trees never hurt)


def suggest(name, t):
    if name == 'lgbm':
        return dict(
            n_estimators=t.suggest_int('n_estimators', 150, 2000, log=True),
            learning_rate=t.suggest_float('learning_rate', 0.004, 0.1, log=True),
            num_leaves=t.suggest_int('num_leaves', 4, 64, log=True),
            max_depth=t.suggest_int('max_depth', 2, 10),
            min_child_samples=t.suggest_int('min_child_samples', 5, 100, log=True),
            subsample=t.suggest_float('subsample', 0.5, 1.0),
            subsample_freq=1,
            colsample_bytree=t.suggest_float('colsample_bytree', 0.2, 1.0),
            reg_alpha=t.suggest_float('reg_alpha', 1e-8, 10, log=True),
            reg_lambda=t.suggest_float('reg_lambda', 1e-8, 30, log=True),
        )
    if name == 'xgb':
        return dict(
            n_estimators=t.suggest_int('n_estimators', 150, 2000, log=True),
            learning_rate=t.suggest_float('learning_rate', 0.004, 0.1, log=True),
            max_depth=t.suggest_int('max_depth', 2, 8),
            min_child_weight=t.suggest_float('min_child_weight', 0.5, 30, log=True),
            subsample=t.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=t.suggest_float('colsample_bytree', 0.2, 1.0),
            gamma=t.suggest_float('gamma', 1e-8, 5, log=True),
            reg_alpha=t.suggest_float('reg_alpha', 1e-8, 10, log=True),
            reg_lambda=t.suggest_float('reg_lambda', 1e-3, 30, log=True),
        )
    if name == 'cat':
        return dict(
            iterations=t.suggest_int('iterations', 300, 2000, log=True),
            learning_rate=t.suggest_float('learning_rate', 0.01, 0.1, log=True),
            depth=t.suggest_int('depth', 4, 8),
            l2_leaf_reg=t.suggest_float('l2_leaf_reg', 0.5, 30, log=True),
            random_strength=t.suggest_float('random_strength', 1e-3, 10, log=True),
            # default bootstrap is MVS, which ignores bagging_temperature (the teammates' tuned value
            # had no effect) -> tune the MVS subsample rate instead
            subsample=t.suggest_float('subsample', 0.5, 1.0),
            rsm=t.suggest_float('rsm', 0.2, 1.0),
        )
    if name in ('et', 'rf'):
        p = dict(
            n_estimators=FOREST_TREES,
            max_depth=t.suggest_int('max_depth', 4, 40, log=True),
            min_samples_leaf=t.suggest_int('min_samples_leaf', 1, 30, log=True),
            max_features=t.suggest_float('max_features', 0.05, 1.0, log=True),
            criterion=t.suggest_categorical('criterion', ['gini', 'entropy']),
        )
        if name == 'rf':
            p['max_samples'] = t.suggest_float('max_samples', 0.3, 1.0)
        return p
    raise ValueError(name)


splits = L.H.make_splits(L.tr, L.y, n_rep=2, seed0=7000)
storage = 'sqlite:///' + os.path.join(L.M.GBM, 'optuna.db')
study = optuna.create_study(direction='maximize', study_name=f'{name}_{scheme}', storage=storage,
                            load_if_exists=True,
                            sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=10, multivariate=True))
if len(study.trials) == 0:
    b = {k: v for k, v in L.BASE[name].items() if k not in ('subsample_freq',)}
    if name in ('et', 'rf'):
        b.pop('n_estimators', None)
        if name == 'rf':
            b['max_samples'] = 1.0
    if name == 'cat':
        b.pop('bagging_temperature', None)
        b['subsample'], b['rsm'] = 0.8, 1.0      # what the teammates' model effectively used
    study.enqueue_trial(b)


def objective(t):
    p = suggest(name, t)
    res = L.run(name, p, splits, scheme=scheme, verbose=False)
    t.set_user_attr('aucs', [float(a) for a in res['aucs']])
    print(f'trial {t.number:3d} AUC {res["auc"]:.5f} {np.round(res["aucs"], 4)}', flush=True)
    return res['auc']


n_left = n_trials - len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
if n_left > 0:
    study.optimize(objective, n_trials=n_left)
df = study.trials_dataframe().sort_values('value', ascending=False)
print(df.head(10).to_string())

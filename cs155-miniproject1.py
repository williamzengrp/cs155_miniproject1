"""CS155 Miniproject 1 -- tuned ensemble (XGBoost, LightGBM, CatBoost, ExtraTrees, RF, LogReg).

Usage:
    python cs155-miniproject1.py           # use tuned params from best_params.json (tunes if missing)
    python cs155-miniproject1.py --tune    # re-run the Optuna search for every model
    python cs155-miniproject1.py --tune --trials 80
"""
import argparse
import json
import os
import warnings

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.set_option('display.max_columns', None)

parser = argparse.ArgumentParser()
parser.add_argument('--tune', action='store_true', help='re-run hyperparameter search')
parser.add_argument('--trials', type=int, default=60, help='Optuna trials per model')
args = parser.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_FILE = os.path.join(HERE, 'best_params.json')
N_JOBS = -1

data_tr = pd.read_csv(os.path.join(HERE, 'train.csv'))
data_te = pd.read_csv(os.path.join(HERE, 'test.csv'))
TARGET = 'Popularity_Type'

# ---------- 1. drop rows with no target; keep song id for grouped CV ----------
# ~220 songs appear more than once in train (same uri) but never in test, so the uri is used as a
# CV group id: otherwise copies land on both sides of a fold and CV rewards memorisation.
data_tr = data_tr.dropna(subset=[TARGET]).reset_index(drop=True)
groups = pd.factorize(data_tr['uri'])[0]

# ---------- 2. drop useless columns ----------
drop_cols = ['track_href', 'uri', 'analysis_url', 'type']
data_tr = data_tr.drop(columns=drop_cols, errors='ignore')
data_te = data_te.drop(columns=drop_cols, errors='ignore')

# ---------- 3. release date -> numbers ----------
def parse_date(s):
    d = pd.to_datetime(s, format='%m/%d/%Y', errors='coerce')                 # 5/23/2024
    d = d.fillna(pd.to_datetime(s, format='mixed', errors='coerce'))          # anything else
    return d

for df in (data_tr, data_te):
    d = parse_date(df['track_album_release_date'])
    print('unparsed dates:', d.isna().sum())
    df['release_year'] = d.dt.year
    df['release_month'] = d.dt.month
    df['days_since_release'] = (pd.Timestamp('2025-01-01') - d).dt.days
    df['is_jan1'] = ((d.dt.month == 1) & (d.dt.day == 1)).astype(int)
    df.drop(columns=['track_album_release_date'], inplace=True)

# ---------- 4. target: High popularity = 1 (as in the assignment spec) ----------
y = (data_tr[TARGET] == 'High').astype(int).to_numpy()
print('positive (High) rate: %.3f' % y.mean())

# ---------- 5. features: same columns in train and test ----------
features = [c for c in data_tr.columns if c not in (TARGET, 'ID') and c in data_te.columns]
leftover = data_tr[features].select_dtypes(include='object').columns.tolist()
print('dropping non-numeric columns:', leftover)
features = [c for c in features if c not in leftover]

X = data_tr[features].reset_index(drop=True)
X_test = data_te[features].reset_index(drop=True)
# linear / sklearn-tree models cannot take NaN; boosters handle it natively
fill = X.median()
X_filled, X_test_filled = X.fillna(fill), X_test.fillna(fill)
print('features:', features)


# ---------- 6. model zoo + search spaces ----------
def make_model(name, p, seed=0):
    p = dict(p)
    if name == 'xgb':
        return XGBClassifier(**p, tree_method='hist', eval_metric='auc',
                             random_state=seed, n_jobs=N_JOBS)
    if name == 'lgbm':
        return LGBMClassifier(**p, random_state=seed, n_jobs=N_JOBS, verbose=-1)
    if name == 'cat':
        return CatBoostClassifier(**p, random_seed=seed, verbose=0,
                                  thread_count=N_JOBS, allow_writing_files=False)
    if name == 'et':
        return ExtraTreesClassifier(**p, random_state=seed, n_jobs=N_JOBS)
    if name == 'rf':
        return RandomForestClassifier(**p, random_state=seed, n_jobs=N_JOBS)
    if name == 'lr':
        return make_pipeline(QuantileTransformer(n_quantiles=200, output_distribution='normal'),
                             LogisticRegression(max_iter=5000, **p))
    raise ValueError(name)


def suggest(name, t):
    if name == 'xgb':
        return dict(
            n_estimators=t.suggest_int('n_estimators', 100, 3000, log=True),
            learning_rate=t.suggest_float('learning_rate', 0.002, 0.2, log=True),
            max_depth=t.suggest_int('max_depth', 2, 10),
            min_child_weight=t.suggest_float('min_child_weight', 0.5, 30, log=True),
            subsample=t.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=t.suggest_float('colsample_bytree', 0.3, 1.0),
            gamma=t.suggest_float('gamma', 1e-8, 5, log=True),
            reg_alpha=t.suggest_float('reg_alpha', 1e-8, 10, log=True),
            reg_lambda=t.suggest_float('reg_lambda', 1e-3, 30, log=True),
        )
    if name == 'lgbm':
        return dict(
            n_estimators=t.suggest_int('n_estimators', 100, 3000, log=True),
            learning_rate=t.suggest_float('learning_rate', 0.002, 0.2, log=True),
            num_leaves=t.suggest_int('num_leaves', 4, 64, log=True),
            max_depth=t.suggest_int('max_depth', 2, 10),
            min_child_samples=t.suggest_int('min_child_samples', 5, 100, log=True),
            subsample=t.suggest_float('subsample', 0.5, 1.0),
            subsample_freq=1,
            colsample_bytree=t.suggest_float('colsample_bytree', 0.3, 1.0),
            reg_alpha=t.suggest_float('reg_alpha', 1e-8, 10, log=True),
            reg_lambda=t.suggest_float('reg_lambda', 1e-8, 30, log=True),
        )
    if name == 'cat':
        return dict(
            iterations=t.suggest_int('iterations', 200, 2500, log=True),
            learning_rate=t.suggest_float('learning_rate', 0.01, 0.2, log=True),
            depth=t.suggest_int('depth', 3, 8),
            l2_leaf_reg=t.suggest_float('l2_leaf_reg', 0.1, 30, log=True),
            random_strength=t.suggest_float('random_strength', 1e-3, 10, log=True),
            bagging_temperature=t.suggest_float('bagging_temperature', 0, 2),
        )
    if name in ('et', 'rf'):
        return dict(
            n_estimators=t.suggest_int('n_estimators', 200, 1000, step=100),
            max_depth=t.suggest_int('max_depth', 4, 30, log=True),
            min_samples_leaf=t.suggest_int('min_samples_leaf', 1, 30, log=True),
            max_features=t.suggest_float('max_features', 0.2, 1.0),
            criterion=t.suggest_categorical('criterion', ['gini', 'entropy']),
        )
    if name == 'lr':
        return dict(C=t.suggest_float('C', 1e-3, 100, log=True))
    raise ValueError(name)


MODELS = ['xgb', 'lgbm', 'cat', 'et', 'rf', 'lr']
NEEDS_FILLED = {'et', 'rf', 'lr'}
# reduce trials for the slow / low-dimensional searches
TRIALS = {'xgb': args.trials, 'lgbm': args.trials, 'cat': max(20, args.trials // 2),
          'et': max(20, args.trials // 2), 'rf': max(20, args.trials // 2), 'lr': 20}


def data_for(name):
    return (X_filled, X_test_filled) if name in NEEDS_FILLED else (X, X_test)


def cv_auc(name, p, cv):
    Xn, _ = data_for(name)
    oof = np.zeros(len(y))
    for tr_idx, va_idx in cv.split(Xn, y, groups):
        m = make_model(name, p)
        m.fit(Xn.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = m.predict_proba(Xn.iloc[va_idx])[:, 1]
    return roc_auc_score(y, oof)


# ---------- 7. hyperparameter search (5-fold grouped CV AUC, seed 42) ----------
best_params = {}
if os.path.exists(PARAMS_FILE) and not args.tune:
    with open(PARAMS_FILE) as f:
        best_params = json.load(f)
    print('loaded tuned params from', PARAMS_FILE)

tune_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for name in MODELS:
    if name in best_params:
        continue
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(lambda t: cv_auc(name, suggest(name, t), tune_cv),
                   n_trials=TRIALS[name])
    best_params[name] = study.best_params
    if name == 'lgbm':
        best_params[name]['subsample_freq'] = 1
    print(f'[tune] {name:5s} best CV AUC {study.best_value:.5f}  {study.best_params}')
    with open(PARAMS_FILE, 'w') as f:
        json.dump(best_params, f, indent=2)


# ---------- 8. out-of-sample evaluation on FRESH folds (different seeds than tuning) ----------
# 3x repeated grouped 5-fold CV: every training row gets 3 out-of-fold predictions, averaged.
# The test set prediction is the average of all 15 fold models.
n_rep = 3
eval_splits = [s for r in range(n_rep) for s in StratifiedGroupKFold(
    n_splits=5, shuffle=True, random_state=2026 + r).split(X, y, groups)]
oof = {n: np.zeros(len(y)) for n in MODELS}
test_pred = {n: np.zeros(len(X_test)) for n in MODELS}
for name in MODELS:
    Xn, Xt = data_for(name)
    fold_aucs = []
    for k, (tr_idx, va_idx) in enumerate(eval_splits):
        m = make_model(name, best_params[name], seed=k)
        m.fit(Xn.iloc[tr_idx], y[tr_idx])
        p_va = m.predict_proba(Xn.iloc[va_idx])[:, 1]
        fold_aucs.append(roc_auc_score(y[va_idx], p_va))
        oof[name][va_idx] += p_va / n_rep
        test_pred[name] += m.predict_proba(Xt)[:, 1] / len(eval_splits)
    print(f'[oof]  {name:5s} AUC {roc_auc_score(y, oof[name]):.5f}  '
          f'(fold mean {np.mean(fold_aucs):.5f} +- {np.std(fold_aucs):.5f})')


# ---------- 9. ensembling ----------
def rank01(v):
    return rankdata(v) / len(v)

O = np.column_stack([rank01(oof[n]) for n in MODELS])


def fit_weights(Oa, ya):
    # non-negative weights summing to 1 that maximise AUC of the rank-blend (Nelder-Mead, multi-start)
    k = Oa.shape[1]

    def neg_auc(w):
        w = np.abs(w) / np.abs(w).sum()
        return -roc_auc_score(ya, Oa @ w)

    best = None
    for start in [np.ones(k) / k] + [np.eye(k)[i] * 0.5 + 0.5 / k for i in range(k)]:
        r = minimize(neg_auc, start, method='Nelder-Mead',
                     options=dict(maxiter=2000, xatol=1e-4, fatol=1e-6))
        if best is None or r.fun < best.fun:
            best = r
    w = np.abs(best.x)
    return w / w.sum()


results = {}
results['mean of all (rank avg)'] = roc_auc_score(y, O.mean(axis=1))
gbm_idx = [MODELS.index(n) for n in ('xgb', 'lgbm', 'cat')]
results['mean of GBMs (rank avg)'] = roc_auc_score(y, O[:, gbm_idx].mean(axis=1))

# honest estimate of the weighted blend: learn weights on 4/5 of the OOF preds, score on 1/5
nested = np.zeros(len(y))
for tr_idx, va_idx in StratifiedGroupKFold(5, shuffle=True, random_state=7).split(O, y, groups):
    w = fit_weights(O[tr_idx], y[tr_idx])
    nested[va_idx] = O[va_idx] @ w
results['weighted rank blend (nested)'] = roc_auc_score(y, nested)

w_final = fit_weights(O, y)
print('\nblend weights:', {n: round(float(w), 3) for n, w in zip(MODELS, w_final)})
print('\n=== out-of-sample AUC (3x5-fold grouped CV, folds unseen by tuning) ===')
for n in MODELS:
    print(f'  {n:30s} {roc_auc_score(y, oof[n]):.5f}')
for k, v in results.items():
    print(f'  {k:30s} {v:.5f}')

# pick the ensemble with the best honest score
choice = max(results, key=results.get)
if choice.startswith('weighted'):
    weights = w_final
elif choice.startswith('mean of GBMs'):
    weights = np.zeros(len(MODELS)); weights[gbm_idx] = 1 / len(gbm_idx)
else:
    weights = np.ones(len(MODELS)) / len(MODELS)
print(f'\nusing: {choice}  weights={ {n: round(float(w), 3) for n, w in zip(MODELS, weights)} }')

# ---------- 10. test predictions ----------
T = np.column_stack([rank01(test_pred[n]) for n in MODELS])
p_test = T @ weights          # rank-blended score in (0, 1]; AUC only depends on ordering

sub = pd.DataFrame({'ID': data_te['ID'], TARGET: p_test})
sub.to_csv(os.path.join(HERE, 'predictions.csv'), index=False)
print(sub.head())

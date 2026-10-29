"""CS155 Miniproject 1 -- refit the tuned ensemble on 100% of train.csv and predict test.csv.

cs155-miniproject1.py predicts test with the average of 15 fold models, each trained on ~80% of
the data. This script instead trains every model on all training rows (several seeds each) and
rank-blends them with the weights chosen by the CV run.

Needs best_params_v3.json and blend_weights.json (both written by cs155-miniproject1.py).

Usage:
    python refit_full.py              # -> predictions_full.csv
    python refit_full.py --seeds 10
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from xgboost import XGBClassifier

from time_features import FEATURE_GROUPS, build_features

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--seeds', type=int, default=5, help='random seeds averaged per model')
args = parser.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = 'Popularity_Type'
N_JOBS = -1

data_tr = pd.read_csv(os.path.join(HERE, 'train.csv')).dropna(subset=[TARGET]).reset_index(drop=True)
data_te = pd.read_csv(os.path.join(HERE, 'test.csv'))
y = (data_tr[TARGET] == 'High').astype(int).to_numpy()

X, X_test = build_features(data_tr, data_te, FEATURE_GROUPS)   # same features as cs155-miniproject1.py
fill = X.median()
X_filled, X_test_filled = X.fillna(fill), X_test.fillna(fill)

with open(os.path.join(HERE, 'best_params_v3.json')) as f:
    best_params = json.load(f)
with open(os.path.join(HERE, 'blend_weights.json')) as f:
    weights = json.load(f)
print('blend weights:', weights)


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


NEEDS_FILLED = {'et', 'rf', 'lr'}
blend = np.zeros(len(X_test))
for name, w in weights.items():
    Xn, Xt = (X_filled, X_test_filled) if name in NEEDS_FILLED else (X, X_test)
    p = np.zeros(len(Xt))
    for seed in range(args.seeds):
        m = make_model(name, best_params[name], seed=seed)
        m.fit(Xn, y)
        p += m.predict_proba(Xt)[:, 1] / args.seeds
    blend += w * rankdata(p) / len(p)
    print(f'[full] {name:5s} fitted on {len(y)} rows x {args.seeds} seeds', flush=True)

sub = pd.DataFrame({'ID': data_te['ID'], TARGET: blend})
sub.to_csv(os.path.join(HERE, 'predictions_full.csv'), index=False)
print(sub.head())

# sanity check: how closely does the full refit agree with the 15-fold-average submission?
cv_sub = pd.read_csv(os.path.join(HERE, 'predictions.csv'))
print('Spearman vs predictions.csv: %.4f' % sub[TARGET].corr(cv_sub[TARGET], method='spearman'))

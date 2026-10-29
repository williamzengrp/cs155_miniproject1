"""Ablation: which time-feature groups help? Grouped 2x5-fold CV AUC with tuned LGBM and CatBoost."""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from time_features import build_features

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = 'Popularity_Type'
tr = pd.read_csv(os.path.join(HERE, 'train.csv')).dropna(subset=[TARGET]).reset_index(drop=True)
te = pd.read_csv(os.path.join(HERE, 'test.csv'))
y = (tr[TARGET] == 'High').astype(int).to_numpy()
groups = pd.factorize(tr['uri'])[0]
P = json.load(open(os.path.join(HERE, 'best_params.json')))
splits = [s for r in range(2) for s in StratifiedGroupKFold(
    5, shuffle=True, random_state=2026 + r).split(tr, y, groups)]


def cv(X, model):
    oof = np.zeros(len(y))
    for k, (a, b) in enumerate(splits):
        if model == 'lgbm':
            m = LGBMClassifier(**P['lgbm'], random_state=k, n_jobs=-1, verbose=-1)
        else:
            m = CatBoostClassifier(**P['cat'], random_seed=k, verbose=0, thread_count=-1,
                                   allow_writing_files=False)
        m.fit(X.iloc[a], y[a])
        oof[b] += m.predict_proba(X.iloc[b])[:, 1] / 2
    return roc_auc_score(y, oof)


ALL = ['trend', 'seasonality', 'precision', 'density', 'detrended']
configs = {'baseline': ()}
configs.update({f'+{g}': (g,) for g in ALL})
configs['all'] = tuple(ALL)
configs.update({f'all-{g}': tuple(x for x in ALL if x != g) for g in ALL})
only = sys.argv[1:]
if only:
    configs = {c: tuple(c.split(',')) for c in only}
for name, g in configs.items():
    X, _ = build_features(tr, te, g)
    print(f'{name:40s} n_feat={X.shape[1]:3d}  lgbm {cv(X, "lgbm"):.5f}  cat {cv(X, "cat"):.5f}',
          flush=True)

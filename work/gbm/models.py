"""GBM agent helpers: model factory, sample-weight schemes, cached features.

    import sys; sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/gbm')
    import models as M
    tr, te, y, X, Xt = M.load_data()                 # 67 FEATURE_GROUPS features (cached)
    A, At = M.data_for('et', X, Xt)                   # median-filled copies for sklearn forests
    m = M.make_model('lgbm', params, seed=0, n_jobs=2)
    m.fit(A.iloc[idx], y[idx], sample_weight=M.sample_weight(tr['uri'].to_numpy()[idx], 'recommended'))
"""
import os
import sys
import json

import numpy as np
import pandas as pd

WORK = './work'
GBM = os.path.join(WORK, 'gbm')
if WORK not in sys.path:
    sys.path.insert(0, WORK)
import harness as H                                         # noqa: E402
from time_features import build_features, FEATURE_GROUPS   # noqa: E402

FILLED = ('et', 'rf')          # sklearn forests get median-filled features (as in the teammates' code)
MODELS = ('lgbm', 'xgb', 'cat', 'et', 'rf')


# feature set: 'v1' = teammates' 67 FEATURE_GROUPS features (used for tuning);
#              'v2' = fe agent's build_features_v2 (55 features; what the integrated results use)
FEATS = os.environ.get('GBM_FEATS', 'v1')


def load_data(feats=None):
    """tr, te, y, X, Xt for the chosen feature set (label-free, cached in gbm/feat_cache*.pkl)."""
    feats = feats or FEATS
    tr, te, y = H.load()
    cache = os.path.join(GBM, 'feat_cache.pkl' if feats == 'v1' else f'feat_cache_{feats}.pkl')
    if os.path.exists(cache):
        X, Xt = pd.read_pickle(cache)
    elif feats == 'v1':
        X, Xt = build_features(tr, te, FEATURE_GROUPS)
        pd.to_pickle((X, Xt), cache)
    elif feats == 'v2':
        saved = list(sys.path)
        sys.path.insert(0, os.path.join(WORK, 'fe'))
        try:
            from features_v2 import build_features_v2
            X, Xt = build_features_v2(tr, te)
        finally:
            sys.path[:] = saved
        pd.to_pickle((X, Xt), cache)
    else:
        raise ValueError(feats)
    assert len(X) == len(tr) and len(Xt) == len(te)
    return tr, te, y, X, Xt


def data_for(name, X, Xt):
    if name in FILLED:
        fill = X.median()
        return X.fillna(fill), Xt.fillna(fill)
    return X, Xt


def make_model(name, params, seed=0, n_jobs=2):
    """Fixed-size (no early stopping) model so a full-data refit is well defined."""
    p = dict(params)
    if name == 'lgbm':
        from lightgbm import LGBMClassifier
        p.setdefault('subsample_freq', 1)
        return LGBMClassifier(**p, random_state=seed, n_jobs=n_jobs, verbose=-1)
    if name == 'xgb':
        from xgboost import XGBClassifier
        return XGBClassifier(**p, tree_method='hist', random_state=seed, n_jobs=n_jobs)
    if name == 'cat':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**p, random_seed=seed, verbose=0, thread_count=n_jobs,
                                  allow_writing_files=False)
    if name == 'et':
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(**p, random_state=seed, n_jobs=n_jobs)
    if name == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**p, random_state=seed, n_jobs=n_jobs)
    raise ValueError(name)


def uri_counts(uri):
    """Number of copies of each row's song among the given rows."""
    s = pd.Series(uri)
    return s.map(s.value_counts()).to_numpy()


def sample_weight(uri, scheme='none'):
    """Per-row training weights for the rows being fitted (uri of the FIT rows only).

    'none'      every row weight 1 (teammates)
    'inv'       1/count per song: each song counts once
    'dup<a>'    rows of multi-copy songs get weight a (e.g. 'dup0.5'); singletons 1
    'invdup<a>' 1/count * a for multi-copy songs (each multi-copy song counts a)
    'recommended' -> see RECOMMENDED_WEIGHTING
    """
    if scheme == 'recommended':
        scheme = RECOMMENDED_WEIGHTING
    c = uri_counts(uri).astype(float)
    if scheme in ('none', None):
        return np.ones(len(c))
    if scheme == 'inv':
        return 1.0 / c
    if scheme.startswith('invdup'):
        a = float(scheme[6:])
        return np.where(c > 1, a / c, 1.0)
    if scheme.startswith('dup'):
        a = float(scheme[3:])
        return np.where(c > 1, a, 1.0)
    raise ValueError(scheme)


RECOMMENDED_WEIGHTING = 'none'    # updated at the end of the study (see log.md)


def load_params(path=os.path.join(GBM, 'best_params_newcv.json')):
    with open(path) as f:
        return json.load(f)

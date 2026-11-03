"""fsel agent: shared helpers. Uses cached label-free features + per-fold OOF label features
(build_cache.py; identical seeds to main/final_cv.py), so a model on a feature SUBSET can be run quickly.
Models/params/seeds/fill exactly as main/final_cv.py (params from main/final_params.json)."""
import os, sys, json, pickle, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
WORK = './work'
sys.path.insert(0, WORK)
import harness as H
warnings.filterwarnings('ignore')
C = WORK + '/r2_fsel/cache/'
P = json.load(open(WORK + '/main/final_params.json'))
LABEL = ['knnw_yr_40', 'teh_date_m10', 'cnt_date']

_B = pickle.load(open(C + 'base.pkl', 'rb'))
X, Xt, y = _B['X'], _B['Xt'], _B['y']
FILL = X.median()
ALL = list(X.columns) + LABEL
n, m = len(X), len(Xt)

GROUPS = {
    'audio': ['time_signature', 'speechiness', 'danceability', 'duration_ms', 'energy', 'mode', 'instrumentalness',
              'valence', 'key', 'tempo', 'loudness', 'acousticness', 'liveness'],
    'datebase': ['release_year', 'release_month', 'days_since_release', 'is_jan1'],
    'trend': ['t_frac_year', 't_log_age', 't_decade'],
    'season_raw': ['s_dow', 's_doy', 's_week', 's_quarter', 's_day'],
    'season_cyc': ['s_mon_sin', 's_mon_cos', 's_dow_sin', 's_dow_cos', 's_doy_sin', 's_doy_cos'],
    'season_flag': ['s_is_friday', 's_is_tuesday', 's_holiday', 's_summer'],
    'density': ['n_same_date', 'n_same_month', 'n_same_year', 'n_same_date_audio'],
    'albumnn2': ['a_n_close10', 'a_n_close20', 'a_n_close30', 'a_nn2_dist', 'a_global_nn_dist', 'a_global_nn5_dist'],
    'detrended': ['dt_danceability', 'dt_duration_ms', 'dt_energy', 'dt_valence', 'dt_tempo', 'dt_loudness',
                  'dt_acousticness', 'dt_speechiness', 'dt_instrumentalness', 'dt_liveness'],
    'dratio': ['dr_date_over_month', 'dr_date_over_year', 'dr_date_over_typical'],
    'label': LABEL,
}
assert sorted(sum(GROUPS.values(), [])) == sorted(ALL)


def load_folds(seed0, nrep):
    assert seed0 != 11000, 'c11 reserved'
    return pickle.load(open(C + f'lf_{seed0}_{nrep}.pkl', 'rb'))


def load_full():
    return pickle.load(open(C + 'full.pkl', 'rb'))


def mk(name, seed, nj=2, params=None):
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    p = dict(P[name] if params is None else params)
    if name == 'xgb': return XGBClassifier(**p, tree_method='hist', random_state=seed, n_jobs=nj)
    if name == 'lgbm': return LGBMClassifier(**p, random_state=seed, n_jobs=nj, verbose=-1)
    if name == 'cat': return CatBoostClassifier(**p, random_seed=seed, verbose=0, thread_count=nj, allow_writing_files=False)
    if name == 'et': return ExtraTreesClassifier(**p, random_state=seed, n_jobs=nj)
    if name == 'rf': return RandomForestClassifier(**p, random_state=seed, n_jobs=nj)
    raise ValueError(name)


def mat(name, cols, A, F):
    """Model matrix with the requested columns (label-free from A, label features from F), same fill as final_cv."""
    M = pd.concat([A.reset_index(drop=True), F.reset_index(drop=True)], axis=1)
    if name in ('et', 'rf'):
        M = M.fillna(pd.concat([FILL, F.median()]))
    return M[list(cols)]


def run_cv(name, cols, folds, seed_off=0, nj=2, params=None, verbose=False):
    """OOF (R x n, NaN where not eligible) and fold-averaged test preds. Model seed = fold no. k + seed_off."""
    R = max(f['r'] for f in folds) + 1
    oof = np.full((R, n), np.nan)
    tst = np.zeros(m)
    for k, f in enumerate(folds):
        mdl = mk(name, k + seed_off, nj, params).fit(mat(name, cols, X.iloc[f['fit_idx']], f['F_fit']), y[f['fit_idx']])
        pv = mdl.predict_proba(mat(name, cols, X.iloc[f['val_idx']], f['F_val']))[:, 1]
        oof[f['r'], f['val_idx'][f['elig']]] = pv[f['elig']]
        tst += mdl.predict_proba(mat(name, cols, Xt, f['F_te']))[:, 1] / len(folds)
        if verbose:
            print('fold', k, flush=True)
    return oof, tst


def run_full(name, cols, full, nj=2, params=None, seed_off=0):
    out = np.zeros(m)
    for s, (F_fit, F_te) in enumerate(full):
        mdl = mk(name, s + seed_off, nj, params).fit(mat(name, cols, X, F_fit), y)
        out += mdl.predict_proba(mat(name, cols, Xt, F_te))[:, 1] / len(full)
    return out


def aucs_of(oof):
    return np.array([roc_auc_score(y[~np.isnan(o)], o[~np.isnan(o)]) for o in oof])


def save(name, oof, tst, full=None):
    a = aucs_of(oof)
    H.save(name, dict(oof=oof, aucs=a, auc=float(a.mean()), test_foldavg=tst), full)
    return a

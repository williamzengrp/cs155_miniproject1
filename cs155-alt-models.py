"""CS155 Miniproject 1 -- alternative models (separate from cs155-miniproject1.py).

Adds stronger release-date features and tries model families not in the main ensemble:
  * CatBoost with the raw release date / year as native categorical features (ordered target stats)
  * RBF-kernel SVM, MLP, k-NN on quantile-normalised features + out-of-fold date target encoding
Evaluates each with grouped 5-fold CV and writes a rank blend to predictions_alt.csv.

Usage:
    python cs155-alt-models.py
"""
import os
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = 'Popularity_Type'

data_tr = pd.read_csv(os.path.join(HERE, 'train.csv')).dropna(subset=[TARGET]).reset_index(drop=True)
data_te = pd.read_csv(os.path.join(HERE, 'test.csv'))
y = (data_tr[TARGET] == 'High').astype(int).to_numpy()
groups = pd.factorize(data_tr['uri'])[0]   # duplicated songs stay in one fold


# ---------- 1. features (train + test together so frequency counts see both) ----------
def parse_date(s):
    d = pd.to_datetime(s, format='%m/%d/%Y', errors='coerce')
    return d.fillna(pd.to_datetime(s, format='mixed', errors='coerce'))


al = pd.concat([data_tr, data_te], ignore_index=True)
d = parse_date(al['track_album_release_date'])
al['date_str'] = d.dt.strftime('%Y-%m-%d').fillna('NA')
al['year_str'] = d.dt.year.astype('Int64').astype(str)
al['release_year'] = d.dt.year
al['release_month'] = d.dt.month
al['days_since_release'] = (pd.Timestamp('2025-01-01') - d).dt.days
al['is_jan1'] = ((d.dt.month == 1) & (d.dt.day == 1)).astype(int)
al['dow'] = d.dt.dayofweek
al['is_friday'] = (al['dow'] == 4).astype(int)              # major-label releases land on Fridays
al['date_cnt'] = al.groupby('date_str')['date_str'].transform('size')   # ~album size
al['year_cnt'] = al.groupby('release_year')['release_year'].transform('size')

AUDIO = ['time_signature', 'speechiness', 'danceability', 'duration_ms', 'energy', 'mode',
         'instrumentalness', 'valence', 'key', 'tempo', 'loudness', 'acousticness', 'liveness']
NUM = AUDIO + ['release_year', 'release_month', 'days_since_release', 'is_jan1', 'dow',
               'is_friday', 'date_cnt', 'year_cnt']
CAT = ['date_str', 'year_str']

n_tr = len(data_tr)
A = al.iloc[:n_tr].reset_index(drop=True)
A_test = al.iloc[n_tr:].reset_index(drop=True)
fill = A[NUM].median()


# ---------- 2. out-of-fold target encoding of the exact release date ----------
def target_encode(keys_fit, y_fit, keys_apply, prior, k=3):
    st = pd.DataFrame({'k': keys_fit, 'y': y_fit}).groupby('k')['y'].agg(['sum', 'count'])
    enc = (st['sum'] + prior * k) / (st['count'] + k)
    return pd.Series(keys_apply).map(enc).fillna(prior).to_numpy()


def add_date_te(fit_idx, apply_df):
    """Return (TE for the fit rows computed out-of-fold, TE for apply_df fit on all fit rows)."""
    keys, yf, gf = A['date_str'].iloc[fit_idx].to_numpy(), y[fit_idx], groups[fit_idx]
    prior = yf.mean()
    te_fit = np.zeros(len(fit_idx))
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=99).split(keys, yf, gf):
        te_fit[b] = target_encode(keys[a], yf[a], keys[b], prior)
    return te_fit, target_encode(keys, yf, apply_df['date_str'].to_numpy(), prior)


def numeric_matrix(df, te):
    M = df[NUM].fillna(fill).copy()
    M['date_te'] = te
    return M


# ---------- 3. models ----------
def fit_predict(name, fit_idx, apply_df, seed):
    """Fit on A.iloc[fit_idx]; return P(High) for apply_df."""
    yf = y[fit_idx]
    if name == 'cat_native':
        m = CatBoostClassifier(iterations=1500, learning_rate=0.02, depth=6, l2_leaf_reg=5,
                               random_seed=seed, verbose=0, allow_writing_files=False,
                               thread_count=-1, cat_features=CAT)
        m.fit(A.iloc[fit_idx][NUM + CAT], yf)
        return m.predict_proba(apply_df[NUM + CAT])[:, 1]

    te_fit, te_apply = add_date_te(fit_idx, apply_df)
    Xf, Xa = numeric_matrix(A.iloc[fit_idx], te_fit), numeric_matrix(apply_df, te_apply)
    if name == 'lgbm_te':
        m = LGBMClassifier(n_estimators=800, learning_rate=0.01, num_leaves=31, subsample=0.7,
                           subsample_freq=1, colsample_bytree=0.8, min_child_samples=15,
                           random_state=seed, verbose=-1)
    elif name == 'svm':
        m = make_pipeline(QuantileTransformer(n_quantiles=200, output_distribution='normal'),
                          SVC(C=1.0, gamma='scale', probability=True, random_state=seed))
    elif name == 'mlp':
        m = make_pipeline(QuantileTransformer(n_quantiles=200, output_distribution='normal'),
                          MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-2, max_iter=500,
                                        early_stopping=True, n_iter_no_change=20,
                                        random_state=seed))
    elif name == 'knn':
        m = make_pipeline(QuantileTransformer(n_quantiles=200, output_distribution='normal'),
                          KNeighborsClassifier(n_neighbors=75, weights='distance'))
    else:
        raise ValueError(name)
    m.fit(Xf, yf)
    return m.predict_proba(Xa)[:, 1]


MODELS = ['cat_native', 'lgbm_te', 'svm', 'mlp', 'knn']

# ---------- 4. grouped CV (2 repeats) -> OOF + averaged test predictions ----------
n_rep = 2
splits = [s for r in range(n_rep) for s in StratifiedGroupKFold(
    5, shuffle=True, random_state=2026 + r).split(A, y, groups)]
oof = {n: np.zeros(n_tr) for n in MODELS}
test_pred = {n: np.zeros(len(A_test)) for n in MODELS}
for name in MODELS:
    for k, (tr_idx, va_idx) in enumerate(splits):
        oof[name][va_idx] += fit_predict(name, tr_idx, A.iloc[va_idx], seed=k) / n_rep
        test_pred[name] += fit_predict(name, tr_idx, A_test, seed=k) / len(splits)
    print(f'[oof] {name:11s} AUC {roc_auc_score(y, oof[name]):.5f}', flush=True)


# ---------- 5. rank blend with non-negative weights (nested estimate) ----------
def rank01(v):
    return rankdata(v) / len(v)


def fit_weights(Oa, ya):
    k = Oa.shape[1]

    def neg_auc(w):
        w = np.abs(w) / np.abs(w).sum()
        return -roc_auc_score(ya, Oa @ w)

    best = min((minimize(neg_auc, s, method='Nelder-Mead', options=dict(maxiter=2000))
                for s in [np.ones(k) / k] + [np.eye(k)[i] * 0.5 + 0.5 / k for i in range(k)]),
               key=lambda r: r.fun)
    w = np.abs(best.x)
    return w / w.sum()


O = np.column_stack([rank01(oof[n]) for n in MODELS])
nested = np.zeros(n_tr)
for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=7).split(O, y, groups):
    nested[b] = O[b] @ fit_weights(O[a], y[a])
w = fit_weights(O, y)
print(f'\nmean rank blend           AUC {roc_auc_score(y, O.mean(axis=1)):.5f}')
print(f'weighted rank blend (nested) AUC {roc_auc_score(y, nested):.5f}')
print('weights:', {n: round(float(x), 3) for n, x in zip(MODELS, w)})

T = np.column_stack([rank01(test_pred[n]) for n in MODELS])
sub = pd.DataFrame({'ID': data_te['ID'], TARGET: T @ w})
sub.to_csv(os.path.join(HERE, 'predictions_alt.csv'), index=False)
print('wrote predictions_alt.csv')

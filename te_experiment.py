"""Experiment: out-of-fold target encodings of the release calendar on top of the time features.

Inside each outer CV fold, encodings for the fold's training rows come from an inner 5-fold split
(so a row never sees its own label) and encodings for the validation rows are fit on all of the
fold's training rows -- the same situation as train -> test.
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from time_features import build_features, parse_date

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = 'Popularity_Type'
tr = pd.read_csv(os.path.join(HERE, 'train.csv')).dropna(subset=[TARGET]).reset_index(drop=True)
te = pd.read_csv(os.path.join(HERE, 'test.csv'))
y = (tr[TARGET] == 'High').astype(int).to_numpy()
groups = pd.factorize(tr['uri'])[0]
P = json.load(open(os.path.join(HERE, 'best_params_time.json')))
X, _ = build_features(tr, te, ('trend', 'seasonality', 'density'))

d = parse_date(tr['track_album_release_date'].astype(str))
KEYS = {
    'date': tr['track_album_release_date'].astype(str).to_numpy(),
    'ym': d.dt.to_period('M').astype(str).to_numpy(),
    'week': d.dt.to_period('W').astype(str).to_numpy(),
}
NN_COLS = ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'speechiness',
           'instrumentalness', 'tempo', 'duration_ms']
Z = StandardScaler().fit_transform(X[NN_COLS].fillna(X[NN_COLS].median()))


def te_map(k_fit, y_fit, k_apply, prior, m):
    st = pd.DataFrame({'k': k_fit, 'y': y_fit}).groupby('k')['y'].agg(['sum', 'count'])
    enc = (st['sum'] + prior * m) / (st['count'] + m)
    cnt = pd.Series(k_apply).map(st['count']).fillna(0).to_numpy()
    return pd.Series(k_apply).map(enc).fillna(prior).to_numpy(), cnt


def knn_label(fit_idx, apply_idx, k, exclude_self):
    """Mean label of the k audio-nearest training songs released on the same date (else prior)."""
    out = np.full(len(apply_idx), np.nan)
    key = KEYS['date']
    fit_by_key = pd.Series(fit_idx).groupby(key[fit_idx]).apply(np.array).to_dict()
    for j, i in enumerate(apply_idx):
        cand = fit_by_key.get(key[i])
        if cand is None:
            continue
        if exclude_self:
            cand = cand[cand != i]
        if len(cand) == 0:
            continue
        dist = np.linalg.norm(Z[cand] - Z[i], axis=1)
        out[j] = y[cand[np.argsort(dist)[:k]]].mean()
    return out


def encode(fit_idx, apply_idx, variants, oof):
    """Encodings for apply_idx using labels of fit_idx; if oof, apply_idx == fit_idx via inner CV."""
    cols = {}
    if oof:
        parts = {}
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=99).split(
                fit_idx, y[fit_idx], groups[fit_idx]):
            for name, v in encode(fit_idx[a], fit_idx[b], variants, False).items():
                parts.setdefault(name, np.zeros(len(fit_idx)))[b] = v
        return parts
    prior = y[fit_idx].mean()
    for v in variants:
        if v.startswith('knn'):
            cols[v] = knn_label(fit_idx, apply_idx, int(v[3:]), False)
        else:
            enc, cnt = te_map(KEYS[v][fit_idx], y[fit_idx], KEYS[v][apply_idx], prior, m=3)
            cols[f'te_{v}'] = enc
            cols[f'tecnt_{v}'] = cnt
    return cols


splits = [s for r in range(2) for s in StratifiedGroupKFold(
    5, shuffle=True, random_state=2026 + r).split(X, y, groups)]


def run(variants):
    oof = np.zeros(len(y))
    for k, (a, b) in enumerate(splits):
        Xa, Xb = X.iloc[a].copy(), X.iloc[b].copy()
        if variants:
            for n, v in encode(a, a, variants, True).items():
                Xa[n] = v
            for n, v in encode(a, b, variants, False).items():
                Xb[n] = v
        m = LGBMClassifier(**P['lgbm'], random_state=k, n_jobs=-1, verbose=-1)
        m.fit(Xa, y[a])
        oof[b] += m.predict_proba(Xb)[:, 1] / 2
    return roc_auc_score(y, oof)


configs = sys.argv[1:] or ['', 'date', 'date,ym', 'date,ym,week', 'knn3', 'knn5', 'date,knn3']
for c in configs:
    v = [x for x in c.split(',') if x]
    print(f'{c or "no TE":20s} lgbm {run(v):.5f}', flush=True)

"""CS155 Miniproject 1 -- ensemble v2: time features + out-of-fold calendar target encoding,
more model families, and several blending strategies. Writes five submissions prediction_1..5.csv.

Target encodings (computed without ever seeing a row's own label):
  te_date / te_ym   smoothed High-rate of training songs released on the same date / month
  knn3              mean label of the 3 audio-nearest training songs released on the same date
For a CV fold, training rows get encodings from an inner 5-fold split and validation / test rows
get encodings fit on all of the fold's training rows -- the same situation as train -> test.

Submissions (CV AUC of each is printed; blend weights are scored with a nested split):
  prediction_1.csv  weighted rank blend, all models, TE features
  prediction_2.csv  logistic-regression stacker over the models' OOF logits, TE features
  prediction_3.csv  plain rank average of the five tree models, TE features
  prediction_4.csv  weighted rank blend without TE, tree models + SVM / MLP / kNN
  prediction_5.csv  the blend from 1 refit on 100% of the training data (3 seeds per model)

Usage:
    python ensemble_v2.py             # -> prediction_1..5.csv
    python ensemble_v2.py --start 6   # -> prediction_6..10.csv
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.optimize import minimize
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from time_features import FEATURE_GROUPS, build_features, parse_date

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=1, help='number of the first output file')
args = parser.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE
TARGET = 'Popularity_Type'
N_JOBS = -1

data_tr = pd.read_csv(os.path.join(HERE, 'train.csv')).dropna(subset=[TARGET]).reset_index(drop=True)
data_te = pd.read_csv(os.path.join(HERE, 'test.csv'))
y = (data_tr[TARGET] == 'High').astype(int).to_numpy()
groups = pd.factorize(data_tr['uri'])[0]
n, m = len(data_tr), len(data_te)
with open(os.path.join(HERE, 'best_params_v3.json')) as f:
    P = json.load(f)

# ---------- 1. features: rows 0..n-1 are train, n..n+m-1 are test ----------
X_tr, X_te = build_features(data_tr, data_te, FEATURE_GROUPS)
XA = pd.concat([X_tr, X_te], ignore_index=True)
raw = pd.concat([data_tr, data_te], ignore_index=True)['track_album_release_date'].astype(str)
d = parse_date(raw)
KEYS = {'date': raw.to_numpy(), 'ym': d.dt.to_period('M').astype(str).to_numpy()}
NN_COLS = ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'speechiness',
           'instrumentalness', 'tempo', 'duration_ms']
Z = StandardScaler().fit_transform(XA[NN_COLS].fillna(XA[NN_COLS].median()))
FILL = X_tr.median()


# ---------- 2. target encodings ----------
def te_map(fit_idx, apply_idx, key, prior, k=3):
    st = pd.DataFrame({'k': KEYS[key][fit_idx], 'y': y[fit_idx]}).groupby('k')['y'].agg(['sum', 'count'])
    enc = (st['sum'] + prior * k) / (st['count'] + k)
    ka = pd.Series(KEYS[key][apply_idx])
    return ka.map(enc).fillna(prior).to_numpy(), ka.map(st['count']).fillna(0).to_numpy()


def knn_label(fit_idx, apply_idx, prior, k=3):
    by_date = pd.Series(fit_idx).groupby(KEYS['date'][fit_idx]).apply(np.array).to_dict()
    out = np.full(len(apply_idx), prior)
    for j, i in enumerate(apply_idx):
        cand = by_date.get(KEYS['date'][i])
        if cand is not None and len(cand):
            near = cand[np.argsort(np.linalg.norm(Z[cand] - Z[i], axis=1))[:k]]
            out[j] = y[near].mean()
    return out


def encode(fit_idx, apply_idx):
    prior = y[fit_idx].mean()
    cols = {}
    for key in ('date', 'ym'):
        cols[f'te_{key}'], cols[f'tecnt_{key}'] = te_map(fit_idx, apply_idx, key, prior)
    cols['knn3'] = knn_label(fit_idx, apply_idx, prior)
    return pd.DataFrame(cols)


def encode_oof(fit_idx):
    out = None
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=99).split(
            fit_idx, y[fit_idx], groups[fit_idx]):
        e = encode(fit_idx[a], fit_idx[b])
        if out is None:
            out = pd.DataFrame(0.0, index=range(len(fit_idx)), columns=e.columns)
        out.iloc[b] = e.to_numpy()
    return out


def matrices(fit_idx, apply_sets, use_te):
    """Feature matrix for fit rows and for each array of apply rows."""
    Xf = XA.iloc[fit_idx].reset_index(drop=True)
    Xs = [XA.iloc[a].reset_index(drop=True) for a in apply_sets]
    if use_te:
        Xf = pd.concat([Xf, encode_oof(fit_idx)], axis=1)
        Xs = [pd.concat([x, encode(fit_idx, a)], axis=1) for x, a in zip(Xs, apply_sets)]
    return Xf, Xs


# ---------- 3. models ----------
TREES = ['xgb', 'lgbm', 'cat', 'et', 'rf']
ALT = ['svm', 'mlp', 'knn']


def qt():
    return QuantileTransformer(n_quantiles=200, output_distribution='normal')


def make_model(name, seed):
    if name == 'xgb':
        return XGBClassifier(**P['xgb'], tree_method='hist', random_state=seed, n_jobs=N_JOBS)
    if name == 'lgbm':
        return LGBMClassifier(**P['lgbm'], random_state=seed, n_jobs=N_JOBS, verbose=-1)
    if name == 'cat':
        return CatBoostClassifier(**P['cat'], random_seed=seed, verbose=0, thread_count=N_JOBS,
                                  allow_writing_files=False)
    if name == 'et':
        return ExtraTreesClassifier(**P['et'], random_state=seed, n_jobs=N_JOBS)
    if name == 'rf':
        return RandomForestClassifier(**P['rf'], random_state=seed, n_jobs=N_JOBS)
    if name == 'svm':
        return make_pipeline(qt(), SVC(C=1.0, gamma='scale', probability=True, random_state=seed))
    if name == 'mlp':
        return make_pipeline(qt(), MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-2,
                                                 max_iter=500, early_stopping=True,
                                                 n_iter_no_change=20, random_state=seed))
    if name == 'knn':
        return make_pipeline(qt(), KNeighborsClassifier(n_neighbors=75, weights='distance'))
    raise ValueError(name)


def fit_predict(name, Xf, yf, Xs, seed):
    if name not in ('xgb', 'lgbm', 'cat'):        # boosters take NaN natively
        Xf, Xs = Xf.fillna(FILL), [x.fillna(FILL) for x in Xs]
    mdl = make_model(name, seed)
    mdl.fit(Xf, yf)
    return [mdl.predict_proba(x)[:, 1] for x in Xs]


# ---------- 4. repeated grouped CV -> OOF + fold-averaged test predictions ----------
N_REP = 3
splits = [s for r in range(N_REP) for s in StratifiedGroupKFold(
    5, shuffle=True, random_state=2026 + r).split(X_tr, y, groups)]
test_idx = np.arange(n, n + m)
RUNS = [(mdl, True) for mdl in TREES + ALT] + [(mdl, False) for mdl in TREES + ALT]
oof = {r: np.zeros(n) for r in RUNS}
tst = {r: np.zeros(m) for r in RUNS}
for use_te in (True, False):
    for k, (a, b) in enumerate(splits):
        Xf, (Xb, Xt) = matrices(a, [b, test_idx], use_te)
        for mdl in TREES + ALT:
            pb, pt = fit_predict(mdl, Xf, y[a], [Xb, Xt], seed=k)
            oof[(mdl, use_te)][b] += pb / N_REP
            tst[(mdl, use_te)] += pt / len(splits)
        print(f'  te={use_te} fold {k + 1}/{len(splits)} done', flush=True)
for r in RUNS:
    print(f'[oof] {r[0]:5s} te={str(r[1]):5s} AUC {roc_auc_score(y, oof[r]):.5f}', flush=True)
np.savez(os.path.join(HERE, f'oof_v2_from{args.start}.npz'), **{f'oof_{a}_{b}': oof[(a, b)] for a, b in RUNS},
         **{f'tst_{a}_{b}': tst[(a, b)] for a, b in RUNS})


# ---------- 5. blends ----------
def rank01(v):
    return rankdata(v) / len(v)


def fit_weights(O, yy):
    kk = O.shape[1]

    def neg(w):
        w = np.abs(w) / np.abs(w).sum()
        return -roc_auc_score(yy, O @ w)

    best = min((minimize(neg, s, method='Nelder-Mead', options=dict(maxiter=3000))
                for s in [np.ones(kk) / kk] + [np.eye(kk)[i] * 0.5 + 0.5 / kk for i in range(kk)]),
               key=lambda r: r.fun)
    w = np.abs(best.x)
    return w / w.sum()


def stack_fit(O, yy):
    return LogisticRegression(C=1.0, max_iter=2000).fit(logit(np.clip(O, 1e-4, 1 - 1e-4)), yy)


def stack_pred(s, O):
    return s.predict_proba(logit(np.clip(O, 1e-4, 1 - 1e-4)))[:, 1]


outer = list(StratifiedGroupKFold(5, shuffle=True, random_state=7).split(X_tr, y, groups))


def nested(runs, how):
    """Honest CV AUC of a blend whose weights / stacker are learned on the OOF predictions."""
    O = np.column_stack([rank01(oof[r]) for r in runs])
    Op = np.column_stack([oof[r] for r in runs])
    res = np.zeros(n)
    for a, b in outer:
        if how == 'weights':
            res[b] = O[b] @ fit_weights(O[a], y[a])
        else:
            res[b] = stack_pred(stack_fit(Op[a], y[a]), Op[b])
    return roc_auc_score(y, res)


def write(fname, scores, cv):
    pd.DataFrame({'ID': data_te['ID'], TARGET: scores}).to_csv(os.path.join(OUT, fname), index=False)
    print(f'  {fname:24s} CV AUC {cv:.5f}', flush=True)


print('\n=== submissions ===')
te_runs = [(mdl, True) for mdl in TREES + ALT]
O_te = np.column_stack([rank01(oof[r]) for r in te_runs])
T_te = np.column_stack([rank01(tst[r]) for r in te_runs])
w1 = fit_weights(O_te, y)
write(f'prediction_{args.start + 0}.csv', T_te @ w1, nested(te_runs, 'weights'))
print('     weights', {r[0]: round(float(x), 3) for r, x in zip(te_runs, w1)})

stk = stack_fit(np.column_stack([oof[r] for r in te_runs]), y)
write(f'prediction_{args.start + 1}.csv', stack_pred(stk, np.column_stack([tst[r] for r in te_runs])),
      nested(te_runs, 'stack'))

tree_te = [(mdl, True) for mdl in TREES]
write(f'prediction_{args.start + 2}.csv', np.column_stack([rank01(tst[r]) for r in tree_te]).mean(axis=1),
      roc_auc_score(y, np.column_stack([rank01(oof[r]) for r in tree_te]).mean(axis=1)))

no_te = [(mdl, False) for mdl in TREES + ALT]
w4 = fit_weights(np.column_stack([rank01(oof[r]) for r in no_te]), y)
write(f'prediction_{args.start + 3}.csv', np.column_stack([rank01(tst[r]) for r in no_te]) @ w4,
      nested(no_te, 'weights'))
print('     weights', {r[0]: round(float(x), 3) for r, x in zip(no_te, w4)})

# ---------- 6. full refit of blend 1 on all training rows ----------
all_idx = np.arange(n)
Xf, (Xt,) = matrices(all_idx, [test_idx], use_te=True)
full = np.zeros(m)
for (mdl, _), w in zip(te_runs, w1):
    if w < 1e-3:
        continue
    p = np.mean([fit_predict(mdl, Xf, y, [Xt], seed=s)[0] for s in range(3)], axis=0)
    full += w * rank01(p)
write(f'prediction_{args.start + 4}.csv', full, nested(te_runs, 'weights'))
print(f'  ({args.start + 4} has the same CV estimate as {args.start}: same models and weights, trained on all rows)')

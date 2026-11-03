"""Shared evaluation harness for CS155 MP1 (song popularity, AUC).

Test-set construction (verified in EDA): the original data was split ~80/20 at the ROW level, then every
test row whose uri also appears in train was removed (test IDs 2..967 with ~104 gaps; 11 within-test
duplicates remain). So the test set is ~all "singleton" songs, while 11% of train rows are copies of
multi-playlist songs that are 77-96% High.

CV that mimics this: StratifiedKFold on rows; in each validation fold, score only rows whose uri does
NOT appear in the fit part ("eligible" rows). Pooled OOF AUC over eligible rows per repeat, averaged.
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

DATA = './data/'
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
os.makedirs(RESULTS, exist_ok=True)


def load():
    tr = pd.read_csv(DATA + 'train.csv')
    tr = tr.dropna(subset=['danceability']).reset_index(drop=True)   # 1 row with no features
    te = pd.read_csv(DATA + 'test.csv')
    y = (tr['Popularity_Type'] == 'High').astype(int).to_numpy()
    return tr, te, y


def make_splits(tr, y, n_rep=5, n_folds=5, seed0=1000):
    """List of (rep, fit_idx, val_idx, eligible_mask_over_val_idx)."""
    uri = tr['uri'].to_numpy()
    out = []
    for r in range(n_rep):
        skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed0 + r)
        for fit_idx, val_idx in skf.split(uri, y):
            fit_uris = set(uri[fit_idx])
            elig = np.array([u not in fit_uris for u in uri[val_idx]])
            out.append((r, fit_idx, val_idx, elig))
    return out


def make_group_splits(tr, y, n_rep=3, n_folds=5, seed0=2026):
    """Teammates' scheme (grouped by uri, all rows scored) -- for reference only."""
    groups = pd.factorize(tr['uri'])[0]
    out = []
    for r in range(n_rep):
        for fit_idx, val_idx in StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed0 + r).split(
                np.zeros(len(y)), y, groups):
            out.append((r, fit_idx, val_idx, np.ones(len(val_idx), bool)))
    return out


def run_cv(fit_predict, splits, y, n_test=None, verbose=True, tag=None):
    """fit_predict(fit_idx, val_idx, fold_no) -> (pred_val, pred_test_or_None).
    Returns dict with per-repeat OOF (NaN where not eligible) and per-repeat AUCs."""
    n_rep = max(s[0] for s in splits) + 1
    oof = np.full((n_rep, len(y)), np.nan)
    tst = None
    n_t = 0
    for k, (r, fit_idx, val_idx, elig) in enumerate(splits):
        pv, pt = fit_predict(fit_idx, val_idx, k)
        oof[r, val_idx[elig]] = np.asarray(pv)[elig]
        if pt is not None:
            tst = (0 if tst is None else tst) + np.asarray(pt)
            n_t += 1
    aucs = []
    for r in range(n_rep):
        m = ~np.isnan(oof[r])
        aucs.append(roc_auc_score(y[m], oof[r, m]))
    res = dict(oof=oof, aucs=np.array(aucs), auc=float(np.mean(aucs)),
               test_foldavg=None if tst is None else tst / n_t)
    if verbose:
        print(f'[{tag}] AUC {res["auc"]:.5f}  per-rep {np.round(aucs, 4)}', flush=True)
    return res


def save(name, res, test_full=None):
    np.savez(os.path.join(RESULTS, name + '.npz'), oof=res['oof'], aucs=res['aucs'],
             test_foldavg=np.array([]) if res['test_foldavg'] is None else res['test_foldavg'],
             test_full=np.array([]) if test_full is None else test_full)


def load_res(name):
    d = np.load(os.path.join(RESULTS, name + '.npz'))
    return {k: d[k] for k in d.files}


def paired(res_a, res_b):
    """Per-repeat AUC difference a-b (same splits) -> mean, std."""
    d = np.asarray(res_a['aucs']) - np.asarray(res_b['aucs'])
    return float(d.mean()), float(d.std(ddof=1)) if len(d) > 1 else 0.0

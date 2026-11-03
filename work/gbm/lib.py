"""Experiment helpers for the gbm agent (CV fit_predict factories on top of harness.py)."""
import os
import sys
import json
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models as M          # noqa: E402
H = M.H
warnings.filterwarnings('ignore')

NJ = 2
tr, te, y, X, Xt = M.load_data()
URI = tr['uri'].to_numpy()
BASE = json.load(open(os.path.join(M.WORK, 'best_params_v3.json')))
DATA = {n: M.data_for(n, X, Xt) for n in M.MODELS}


def dedup_index(idx, how='major'):
    """One row per song among rows idx; returns (rows, labels)."""
    df = pd.DataFrame({'i': idx, 'u': URI[idx], 'y': y[idx]})
    g = df.groupby('u', sort=False)
    first = g['i'].first().to_numpy()
    mean = g['y'].mean().to_numpy()
    lab = (mean >= 0.5).astype(int) if how == 'major' else y[first]
    return first, lab


def fit_rows(idx, scheme):
    """(rows, labels, weights) used to fit on rows idx under a training-set handling scheme."""
    if scheme in ('dedup', 'dedup_first'):
        rows, lab = dedup_index(idx, 'major' if scheme == 'dedup' else 'first')
        return rows, lab, np.ones(len(rows))
    return idx, y[idx], M.sample_weight(URI[idx], scheme)


def fit_one(name, params, rows, lab, w, seed, extra_fit=None):
    A, _ = DATA[name]
    m = M.make_model(name, params, seed=seed, n_jobs=NJ)
    m.fit(A.iloc[rows], lab, sample_weight=w, **(extra_fit or {}))
    return m


def fp_factory(name, params, scheme='none', seeds=(0,), test=False, seed_mult=1000):
    """Single refit on the fit part (seed = k + seed_mult*s for s in seeds), averaged over seeds."""
    A, At = DATA[name]

    def fp(fit_idx, val_idx, k):
        rows, lab, w = fit_rows(fit_idx, scheme)
        pv, pt = 0.0, 0.0
        for s in seeds:
            m = fit_one(name, params, rows, lab, w, seed=k + seed_mult * s)
            pv = pv + m.predict_proba(A.iloc[val_idx])[:, 1]
            if test:
                pt = pt + m.predict_proba(At)[:, 1]
        return pv / len(seeds), (pt / len(seeds) if test else None)
    return fp


def run(name, params, splits, scheme='none', seeds=(0,), test=False, tag=None, verbose=True):
    t = time.time()
    res = H.run_cv(fp_factory(name, params, scheme, seeds, test), splits, y, verbose=False)
    if verbose:
        print(f'[{tag or name}] AUC {res["auc"]:.5f} per-rep {np.round(res["aucs"], 4)} '
              f'({time.time() - t:.0f}s)', flush=True)
    return res


def pdiff(a, b):
    d = np.asarray(a['aucs']) - np.asarray(b['aucs'])
    return f'{d.mean():+.5f} (sd {d.std(ddof=1) if len(d) > 1 else 0:.5f}, pos {int((d > 0).sum())}/{len(d)})'


def log(msg, path=os.path.join(M.GBM, 'log.md')):
    with open(path, 'a') as f:
        f.write(msg.rstrip() + '\n')
    print(msg, flush=True)

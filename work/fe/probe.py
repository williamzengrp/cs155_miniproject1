"""Probe utilities for label-free feature engineering (fe agent)."""
import json, sys, time, warnings
import numpy as np, pandas as pd
WORK = './work'
sys.path.insert(0, WORK)
import harness as H
from time_features import build_features, FEATURE_GROUPS, AUDIO, parse_date
from lightgbm import LGBMClassifier
warnings.filterwarnings('ignore')

P = json.load(open(H.HERE + '/best_params_v3.json'))
tr, te, y = H.load()
splits = H.make_splits(tr, y, n_rep=5)
LOG = WORK + '/fe/log.md'


def lgbm(seed, nj=2, **over):
    p = dict(P['lgbm']); p.update(over)
    return LGBMClassifier(**p, random_state=seed, n_jobs=nj, verbose=-1)


def cv(X, Xt=None, tag='', seed_off=0, n_seeds=1, verbose=True, **over):
    X = X.reset_index(drop=True)
    def fp(fit_idx, val_idx, k):
        pv = 0; pt = None
        for s in range(n_seeds):
            m = lgbm(k + seed_off + 1000 * s, **over)
            m.fit(X.iloc[fit_idx], y[fit_idx])
            pv = pv + m.predict_proba(X.iloc[val_idx])[:, 1]
            if Xt is not None:
                pt = (0 if pt is None else pt) + m.predict_proba(Xt)[:, 1]
        return pv / n_seeds, (None if pt is None else pt / n_seeds)
    t = time.time()
    res = H.run_cv(fp, splits, y, verbose=False)
    res['time'] = time.time() - t
    if verbose:
        print(f'[{tag}] AUC {res["auc"]:.5f} per-rep {np.round(res["aucs"], 4)} ({res["time"]:.0f}s)', flush=True)
    return res


def diff(a, b):
    d = np.asarray(a['aucs']) - np.asarray(b['aucs'])
    return d.mean(), (d > 0).sum(), d


def log(line):
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def report(tag, res, base, nfeat=None, note=''):
    m, npos, d = diff(res, base)
    line = (f'| {tag} | {nfeat if nfeat is not None else ""} | {res["auc"]:.5f} | {m:+.5f} | {npos}/5 | '
            f'{" ".join(f"{x:+.4f}" for x in d)} | {note} |')
    print(line, flush=True)
    log(line)
    return m, npos


import os
CACHE = WORK + '/fe/cache'
os.makedirs(CACHE, exist_ok=True)


def cv_cached(key, X, Xt=None, seed_off=0, **kw):
    """Run cv (or load) -- cache keyed by a short name + seed offset."""
    fn = f'{CACHE}/{key}__s{seed_off}.npz'
    if os.path.exists(fn):
        d = np.load(fn)
        return dict(aucs=d['aucs'], auc=float(d['aucs'].mean()), oof=d['oof'])
    res = cv(X, Xt, tag=f'{key} s{seed_off}', seed_off=seed_off, **kw)
    np.savez(fn, aucs=res['aucs'], oof=res['oof'])
    return res


def groups_columns():
    """Map each time_features group -> its columns (base columns under 'base')."""
    all_groups = ('trend', 'seasonality', 'precision', 'density', 'density2', 'detrended',
                  'groupagg', 'albumnn', 'albumnn2')
    Xb, _ = build_features(tr, te, ())
    cols = {'base': list(Xb.columns)}
    for g in all_groups:
        Xg, _ = build_features(tr, te, (g,))
        cols[g] = [c for c in Xg.columns if c not in cols['base']]
    Xa, Xta = build_features(tr, te, all_groups)
    return cols, Xa, Xta

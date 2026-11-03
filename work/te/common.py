"""Shared setup for agent `te` experiments: data, 67 baseline features, probe models, CV runner."""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

W = './work'
TE_DIR = os.path.join(W, 'te')
OUT = os.path.join(TE_DIR, 'out')
sys.path.insert(0, W)
sys.path.insert(0, TE_DIR)
import harness as H  # noqa: E402
from time_features import build_features, FEATURE_GROUPS  # noqa: E402
import label_features as LF  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from catboost import CatBoostClassifier  # noqa: E402

warnings.filterwarnings('ignore')
os.makedirs(OUT, exist_ok=True)
tr, te, y = H.load()
n, m = len(tr), len(te)
X, Xt = build_features(tr, te, FEATURE_GROUPS)
XA = pd.concat([X, Xt], ignore_index=True)
_BASES = {'67': XA}


def base_frame(base='67'):
    """'67' = teammates' FEATURE_GROUPS; 'v2' = fe agent's build_features_v2 (55 label-free columns)."""
    if base not in _BASES:
        sys.path.insert(0, os.path.join(W, 'fe'))
        from features_v2 import build_features_v2
        X2, Xt2 = build_features_v2(tr, te)
        _BASES[base] = pd.concat([X2, Xt2], ignore_index=True)
    return _BASES[base]
P = json.load(open(os.path.join(W, 'best_params_v3.json')))
splits = H.make_splits(tr, y, n_rep=5)
TEST = np.arange(n, n + m)
SEED_OFFS = (0, 100, 200)


def lgbm(seed):
    return LGBMClassifier(**P['lgbm'], random_state=seed, n_jobs=2, verbose=-1)


def cat(seed, cat_features=None):
    return CatBoostClassifier(**P['cat'], random_seed=seed, verbose=0, thread_count=2,
                              allow_writing_files=False, cat_features=cat_features)


def matrices(fit_idx, apply_list, spec, seed, n_inner=5, base='67'):
    """Base label-free features (+ label features if spec) for fit rows and each apply array."""
    B = base_frame(base)
    Xf = B.iloc[fit_idx].reset_index(drop=True)
    Xs = [B.iloc[a].reset_index(drop=True) for a in apply_list]
    if spec:
        Ff, Fs = LF.label_features(XA, y, fit_idx, apply_list, tr, te, seed=seed, spec=list(spec),
                                   n_inner=n_inner)
        Xf = pd.concat([Xf, Ff], axis=1)
        Xs = [pd.concat([x, f], axis=1) for x, f in zip(Xs, Fs)]
    return Xf, Xs


def make_fp(spec=(), offs=(0,), n_inner=5, with_test=True, model=lgbm, base='67'):
    """fit_predict for H.run_cv: LightGBM on base features + label features, mean over seeds k+off."""
    def fp(fit_idx, val_idx, k):
        Xf, (Xv, Xt_) = matrices(fit_idx, [val_idx, TEST], spec, seed=k, n_inner=n_inner, base=base)
        pv, pt = 0, 0
        for off in offs:
            mdl = model(k + off).fit(Xf, y[fit_idx])
            pv = pv + mdl.predict_proba(Xv)[:, 1] / len(offs)
            if with_test:
                pt = pt + mdl.predict_proba(Xt_)[:, 1] / len(offs)
        return pv, (pt if with_test else None)
    return fp


def auc_of_oof(oof):
    return np.array([roc_auc_score(y[~np.isnan(o)], o[~np.isnan(o)]) for o in oof])


def baseline_aucs(offs):
    """Per-repeat AUC of the baseline averaged over the given seed offsets (saved by seednoise.py)."""
    oof = np.mean([np.load(os.path.join(OUT, f'base_seed{o}_oof.npy')) for o in offs], axis=0)
    return auc_of_oof(oof)


def report(tag, res, offs, t0=None, log=True, ref=None):
    b = baseline_aucs(offs) if ref is None else ref
    d = res['aucs'] - b
    line = (f'{tag:44s} seeds={len(offs)} AUC {res["auc"]:.5f}  base {b.mean():.5f}  diff {d.mean():+.5f}'
            f'  per-rep {np.array2string(d, precision=4, sign="+")}  pos {int((d > 0).sum())}/5'
            + (f'  ({time.time() - t0:.0f}s)' if t0 else ''))
    print(line, flush=True)
    if log:
        with open(os.path.join(OUT, 'results.txt'), 'a') as f:
            f.write(line + '\n')
    return d


def run(tag, spec=(), offs=(0,), n_inner=5, with_test=False, model=lgbm, save_oof=True, base='67',
        ref=None):
    t0 = time.time()
    res = H.run_cv(make_fp(spec, offs, n_inner, with_test, model, base), splits, y, verbose=False)
    report(tag, res, offs, t0, ref=ref)
    if save_oof:
        np.save(os.path.join(OUT, f'oof_{tag}.npy'), res['oof'])
    return res


def full_fit_test(spec=(), offs=SEED_OFFS, n_inner=5, model=lgbm, seed=0, base='67'):
    """Test predictions from models trained on ALL train rows (label features: inner OOF over train)."""
    all_idx = np.arange(n)
    Xf, (Xt_,) = matrices(all_idx, [TEST], spec, seed=seed, n_inner=n_inner, base=base)
    return np.mean([model(seed + off).fit(Xf, y).predict_proba(Xt_)[:, 1] for off in offs], axis=0)

"""CatBoost (teammates' params, thread_count=2) with release date / year as native categorical features.

Variants: base (67 feats), date (+ date string as cat), dateyr (+ date and year as cat),
date_dedup (+ date cat, training rows deduplicated by uri with weight = number of copies, so a copy of
the same song can never feed its label into another copy's ordered target statistic).
usage: python cat_native.py <variant> [<variant> ...] [--seeds N] [--test]
"""
import sys
import time

sys.path.insert(0, './work/te')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import H, XA, y, tr, te, n, m, splits, TEST, OUT, SEED_OFFS, cat, report  # noqa: E402
from time_features import parse_date  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith('--')]
nseeds = int(sys.argv[sys.argv.index('--seeds') + 1]) if '--seeds' in sys.argv else 1
args = [a for a in args if not a.isdigit()]
offs = SEED_OFFS[:nseeds]
al = pd.concat([tr, te], ignore_index=True)
raw = al['track_album_release_date'].astype(str)
XC = XA.copy()
XC['c_date'] = raw.to_numpy()
XC['c_year'] = parse_date(raw).dt.year.astype(int).astype(str).to_numpy()
uri = pd.factorize(al['uri'])[0]
CATS = {'base': [], 'date': ['c_date'], 'dateyr': ['c_date', 'c_year'], 'date_dedup': ['c_date'],
        'base_dedup': []}


def make_fp(variant, with_test):
    cats = CATS[variant]
    cols = list(XA.columns) + cats

    def fp(fit_idx, val_idx, k):
        fi, w = fit_idx, None
        if variant.endswith('dedup'):
            u = uri[fit_idx]
            _, first, cnt = np.unique(u, return_index=True, return_counts=True)
            fi, w = fit_idx[first], cnt.astype(float)
        pv, pt = 0, 0
        for off in offs:
            mdl = cat(k + off, cat_features=cats or None)
            mdl.fit(XC.iloc[fi][cols], y[fi], sample_weight=w)
            pv = pv + mdl.predict_proba(XC.iloc[val_idx][cols])[:, 1] / len(offs)
            if with_test:
                pt = pt + mdl.predict_proba(XC.iloc[TEST][cols])[:, 1] / len(offs)
        return pv, (pt if with_test else None)
    return fp


for v in args:
    t0 = time.time()
    res = H.run_cv(make_fp(v, '--test' in sys.argv), splits, y, verbose=False)
    print(f'[cat_{v}] seeds={len(offs)} AUC {res["auc"]:.5f} per-rep {np.round(res["aucs"], 4)} ({time.time() - t0:.0f}s)',
          flush=True)
    np.save(f'{OUT}/oof_cat_{v}_s{len(offs)}.npy', res['oof'])
    if res['test_foldavg'] is not None:
        np.save(f'{OUT}/testfa_cat_{v}_s{len(offs)}.npy', res['test_foldavg'])

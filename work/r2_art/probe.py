"""Paired LightGBM probe (fkd_lgbm config: 58 label-free + 3 OOF label features) with extra artifact blocks.
python -I probe.py SEED0 NREP OFFS(comma) CONFIGS(semicolon-separated, each '+'-joined blocks or 'base')
Seed per fold = k + off for both LightGBM and the label-feature inner folds (off=0 reproduces fkd_lgbm)."""
import json, os, sys, time, pickle, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
W = './work'
sys.path.insert(0, W); sys.path.insert(0, W + '/main'); sys.path.insert(0, W + '/te'); sys.path.insert(0, W + '/r2_art')
import harness as H
from features_final import build_features_final
from label_features import label_features, BEST_SPEC
from art_feats import art_features
warnings.filterwarnings('ignore')
SEED0, NREP = int(sys.argv[1]), int(sys.argv[2])
OFFS = [int(x) for x in sys.argv[3].split(',')]
CONFIGS = sys.argv[4].split(';')
P = json.load(open(W + '/best_params_v3.json'))['lgbm']
tr, te, y = H.load()
X, Xt = build_features_final(tr, te)
XA = pd.concat([X, Xt], ignore_index=True)
n, m = len(tr), len(te)
test_idx = np.arange(n, n + m)
blocks = sorted({b for c in CONFIGS if c != 'base' for b in c.split('+')})
EXTRA = {}
for b in blocks:
    EXTRA[b] = art_features(tr, te, blocks=(b,)) if b in ('date', 'ym', 'year', 'knn') else None
import art_feats as AF
for b in blocks:
    if EXTRA[b] is None:
        EXTRA[b] = AF.special_block(tr, te, b)
splits = H.make_splits(tr, y, n_rep=NREP, seed0=SEED0)
CACHE = W + f'/r2_art/cache_lf_{SEED0}_{NREP}.pkl'
cache = pickle.load(open(CACHE, 'rb')) if os.path.exists(CACHE) else {}
R = NREP
aucs = {c: np.zeros((len(OFFS), R)) for c in CONFIGS}
t0 = time.time()


def design(cfg, rows):
    parts = [XA.iloc[rows].reset_index(drop=True)]
    if cfg != 'base':
        for b in cfg.split('+'):
            E = EXTRA[b]
            if isinstance(E, tuple):                 # (drop_cols, replacement frame)
                drop, Rf = E
                parts[0] = parts[0].drop(columns=[c for c in drop if c in parts[0].columns])
                parts.append(Rf.iloc[rows].reset_index(drop=True))
            else:
                parts.append(E.iloc[rows].reset_index(drop=True))
    return pd.concat(parts, axis=1)


for oi, off in enumerate(OFFS):
    oof = {c: np.full((R, n), np.nan) for c in CONFIGS}
    for k, (r, fit_idx, val_idx, elig) in enumerate(splits):
        key = (k, off)
        if key not in cache:
            F_fit, (F_val,) = label_features(XA, y, fit_idx, [val_idx], tr, te, seed=k + off, spec=BEST_SPEC)
            cache[key] = (F_fit, F_val)
        F_fit, F_val = cache[key]
        for c in CONFIGS:
            A = pd.concat([design(c, fit_idx), F_fit.reset_index(drop=True)], axis=1)
            B = pd.concat([design(c, val_idx), F_val.reset_index(drop=True)], axis=1)
            mdl = LGBMClassifier(**P, random_state=k + off, n_jobs=2, verbose=-1).fit(A, y[fit_idx])
            oof[c][r, val_idx[elig]] = mdl.predict_proba(B)[:, 1][elig]
    for c in CONFIGS:
        for r in range(R):
            mk = ~np.isnan(oof[c][r])
            aucs[c][oi, r] = roc_auc_score(y[mk], oof[c][r][mk])
    print(f'off {off} done {time.time()-t0:.0f}s', flush=True)
    pickle.dump(cache, open(CACHE, 'wb'))
base = 'base' if 'base' in CONFIGS else CONFIGS[0]
mb = aucs[base].mean(0)
print(f'{base:30s} {mb.mean():.5f} per-rep {np.round(mb, 4)}  (per-off means {np.round(aucs[base].mean(1), 5)})')
for c in CONFIGS:
    if c == base: continue
    mc = aucs[c].mean(0)
    dd = mc - mb
    do = (aucs[c] - aucs[base]).mean(1)
    print(f'{c:30s} {mc.mean():.5f} diff {dd.mean():+.5f} pos {(dd>0).sum()}/{R} per-rep {np.round(dd, 4)} per-off {np.round(do, 5)}', flush=True)
json.dump({c: aucs[c].tolist() for c in CONFIGS}, open(W + f'/r2_art/probe_{SEED0}_{"_".join(map(str, OFFS))}_{abs(hash(sys.argv[4]))%10**6}.json', 'w'))

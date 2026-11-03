"""Paired LightGBM probe with FOLD-CONSISTENT artifact features (art_features_sim):
in each fold, fit rows = 'train file', eligible val rows = 'test file', ineligible val rows + real test rows left out.
python -I probe_sim.py SEED0 NREP OFFS CONFIGS   (CONFIGS ';'-separated, each 'base' or '+'-joined blocks among date,ym,knn;
  a block may be suffixed ':cols' with comma-free column filter tokens joined by '|', e.g. knn:dupfrac)"""
import json, os, sys, time, pickle, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
W = './work'
sys.path.insert(0, W); sys.path.insert(0, W + '/main'); sys.path.insert(0, W + '/te'); sys.path.insert(0, W + '/r2_art')
import harness as H
from features_final import build_features_final
from label_features import label_features, BEST_SPEC
from art_feats import art_features_sim
warnings.filterwarnings('ignore')
SEED0, NREP = int(sys.argv[1]), int(sys.argv[2])
OFFS = [int(x) for x in sys.argv[3].split(',')]
CONFIGS = sys.argv[4].split(';')
P = json.load(open(W + '/best_params_v3.json'))['lgbm']
tr, te, y = H.load()
X, Xt = build_features_final(tr, te)
XA = pd.concat([X, Xt], ignore_index=True)
n, m = len(tr), len(te)
N = n + m
uri = pd.factorize(pd.concat([tr, te], ignore_index=True)['uri'])[0]
splits = H.make_splits(tr, y, n_rep=NREP, seed0=SEED0)
CACHE = W + f'/r2_art/cache_lf_{SEED0}_{NREP}.pkl'
cache = pickle.load(open(CACHE, 'rb')) if os.path.exists(CACHE) else {}
blocks_all = sorted({b.split(':')[0] for c in CONFIGS if c != 'base' for b in c.split('+')})


def pick(Fk, cfg):
    cols = []
    for b in cfg.split('+'):
        name, _, filt = b.partition(':')
        cc = [c for c in Fk.columns if c.startswith(f'art_{name}')]
        if filt:
            toks = filt.split('|')
            cc = [c for c in cc if any(t in c for t in toks)]
        cols += cc
    return Fk[cols]


R = NREP
aucs = {c: np.zeros((len(OFFS), R)) for c in CONFIGS}
t0 = time.time()
simcache = {}
for oi, off in enumerate(OFFS):
    oof = {c: np.full((R, n), np.nan) for c in CONFIGS}
    for k, (r, fit_idx, val_idx, elig) in enumerate(splits):
        key = (k, off)
        if key not in cache:
            F_fit, (F_val,) = label_features(XA, y, fit_idx, [val_idx], tr, te, seed=k + off, spec=BEST_SPEC)
            cache[key] = (F_fit, F_val)
        F_fit, F_val = cache[key]
        if k not in simcache and blocks_all:
            trm = np.zeros(N, bool); trm[fit_idx] = True
            tem = np.zeros(N, bool); tem[val_idx[elig]] = True
            simcache[k] = art_features_sim(tr, te, trm, tem, blocks=tuple(blocks_all))
        for c in CONFIGS:
            A = [XA.iloc[fit_idx].reset_index(drop=True)]
            B = [XA.iloc[val_idx].reset_index(drop=True)]
            if c != 'base':
                Fk = pick(simcache[k], c)
                A.append(Fk.iloc[fit_idx].reset_index(drop=True)); B.append(Fk.iloc[val_idx].reset_index(drop=True))
            A = pd.concat(A + [F_fit.reset_index(drop=True)], axis=1)
            B = pd.concat(B + [F_val.reset_index(drop=True)], axis=1)
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
json.dump({c: aucs[c].tolist() for c in CONFIGS}, open(W + f'/r2_art/probesim_{SEED0}_{"_".join(map(str, OFFS))}_{abs(hash(sys.argv[4]))%10**6}.json', 'w'))

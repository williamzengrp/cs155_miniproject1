"""LightGBM screen (teammate params, n_jobs=2, 3 seeds k/k+100/k+200 averaged) of graph features on top of
58 label-free + BEST_SPEC label features.  python -I screen.py SEED0 NREP OUTTAG variant1 variant2 ...
Variants are defined in VARIANTS (graph feature names -> specs in graph_lib); 'base' = no graph features.
Results saved as results/graph_scr_<OUTTAG>_<variant>.npz (OOF only)."""
import json, sys, time, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/main'); sys.path.insert(0, WORK + '/te'); sys.path.insert(0, WORK + '/r2_graph')
import harness as H
from features_final import build_features_final
from label_features import label_features, BEST_SPEC
from graph_lib import SongGraph, graph_label_features, dup_mates_features
from graph_specs import GSPECS, VARIANTS
from spectral import spectral_embedding
warnings.filterwarnings('ignore')
SEED0, NREP, OUTTAG = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
VNAMES = sys.argv[4:]
P = json.load(open(WORK + '/best_params_v3.json'))['lgbm']
tr, te, y = H.load()
X, Xt = build_features_final(tr, te)
XA = pd.concat([X, Xt], ignore_index=True)
n, m = len(tr), len(te)
test_idx = np.arange(n, n + m)
G = SongGraph(tr, te)
need = sorted({g for v in VNAMES for g in VARIANTS[v]['add']})
# label-free spectral blocks: VARIANTS[v]['lf'] = list of (name, ndim, graph kwargs)
LF = {}
for v in VNAMES:
    for (nm, nd, gkw) in VARIANTS[v].get('lf', []):
        if nm not in LF:
            if nm.startswith('dup'):
                psi = dup_mates_features(G, **gkw)
            else:
                psi, _ = spectral_embedding(G, nd, **gkw)
            LF[nm] = pd.DataFrame(psi[G.node], columns=[f'{nm}_{j}' for j in range(psi.shape[1])])
splits = H.make_splits(tr, y, n_rep=NREP, seed0=SEED0)
oof = {v: np.full((NREP, n), np.nan) for v in VNAMES}
t0 = time.time()
for k, (r, fit_idx, val_idx, elig) in enumerate(splits):
    F_fit, (F_val,) = label_features(XA, y, fit_idx, [val_idx], tr, te, seed=k, spec=BEST_SPEC)
    if need:
        Gf, (Gv,) = graph_label_features(G, y, fit_idx, [val_idx], {g: GSPECS[g] for g in need}, seed=k)
    for v in VNAMES:
        drop = VARIANTS[v].get('drop', [])
        cols_g = VARIANTS[v]['add']
        lff = [LF[nm].iloc[fit_idx].reset_index(drop=True) for (nm, _, _) in VARIANTS[v].get('lf', [])]
        lfv = [LF[nm].iloc[val_idx].reset_index(drop=True) for (nm, _, _) in VARIANTS[v].get('lf', [])]
        Mf = pd.concat([X.iloc[fit_idx].reset_index(drop=True), F_fit.drop(columns=drop)] + ([Gf[cols_g]] if cols_g else []) + lff, axis=1)
        Mv = pd.concat([X.iloc[val_idx].reset_index(drop=True), F_val.drop(columns=drop)] + ([Gv[cols_g]] if cols_g else []) + lfv, axis=1)
        p = np.zeros(len(val_idx))
        for s in (0, 100, 200):
            mdl = LGBMClassifier(**P, random_state=k + s, n_jobs=2, verbose=-1).fit(Mf, y[fit_idx])
            p += mdl.predict_proba(Mv)[:, 1] / 3
        oof[v][r, val_idx[elig]] = p[elig]
    print(f'fold {k} {time.time()-t0:.0f}s', flush=True)
base = None
for v in VNAMES:
    aucs = np.array([roc_auc_score(y[~np.isnan(oof[v][r])], oof[v][r][~np.isnan(oof[v][r])]) for r in range(NREP)])
    res = dict(oof=oof[v], aucs=aucs, auc=float(aucs.mean()), test_foldavg=None)
    H.save(f'graph_scr_{OUTTAG}_{v}', res)
    if base is None: base = aucs
    d = aucs - base
    print(f'{v:28s} AUC {aucs.mean():.5f} per-rep {np.round(aucs, 4)}  vs first {d.mean():+.5f} pos {(d>0).sum()}/{NREP} {np.round(d,4)}', flush=True)

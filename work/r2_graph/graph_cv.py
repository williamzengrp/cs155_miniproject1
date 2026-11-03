"""final8 tree members + graph-propagated label features (agent r2 `graph`).
python -I graph_cv.py NJ PREFIX PARAMS_JSON SPLITSEED GSET model1 model2 ...
  GSET = name of a feature set in graph_specs.GSETS (list of graph feature names, see graph_specs.GSPECS)
  env NREP (default 5), DO_FULL (default 1: full refit on all training rows, 3 seeds -> test_full)
Same as work/main/final_cv.py (58 label-free feats + BEST_SPEC OOF label features, same params, same model
seeds = fold number) plus the graph features. Graph features: seeds = fit rows only; fit rows get inner-fold
(StratifiedGroupKFold by uri, same partition as the te label features) out-of-fold values; validation and test
rows are unlabelled nodes. Saves results/<PREFIX>_<model>.npz."""
import json, os, sys, time, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/main'); sys.path.insert(0, WORK + '/te'); sys.path.insert(0, WORK + '/r2_graph')
import harness as H
from features_final import build_features_final
from label_features import label_features, BEST_SPEC
from graph_lib import SongGraph, graph_label_features, dup_mates_features
from spectral import spectral_embedding
from graph_specs import GSPECS, GSETS
warnings.filterwarnings('ignore')

NJ, PREFIX, PJSON, SPLITSEED, GSET = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
MODELS = sys.argv[6:]
P = json.load(open(PJSON))
GDEF = GSETS[GSET] if isinstance(GSETS[GSET], dict) else dict(g=GSETS[GSET], lf=[])
GS = {g: GSPECS[g] for g in GDEF['g']}
tr, te, y = H.load()
X, Xt = build_features_final(tr, te)
XA = pd.concat([X, Xt], ignore_index=True)
n, m = len(tr), len(te)
test_idx = np.arange(n, n + m)
fill = X.median()
G = SongGraph(tr, te)
# label-free graph blocks (identical for fit / val / test rows; computed on the train+test pool)
LFB = []
for (nm, nd, gkw) in GDEF.get('lf', []):
    arr = dup_mates_features(G, **gkw) if nm.startswith('dup') else spectral_embedding(G, nd, **gkw)[0]
    LFB.append(pd.DataFrame(arr[G.node], columns=[f'{nm}_{j}' for j in range(arr.shape[1])]))
LFA = pd.concat(LFB, axis=1) if LFB else None
if LFA is not None:
    X = pd.concat([X, LFA.iloc[:n].reset_index(drop=True)], axis=1)
    Xt = pd.concat([Xt, LFA.iloc[n:].reset_index(drop=True)], axis=1)
    fill = X.median()


def mk(name, seed):
    p = dict(P[name])
    if name == 'xgb': return XGBClassifier(**p, tree_method='hist', random_state=seed, n_jobs=NJ)
    if name == 'lgbm': return LGBMClassifier(**p, random_state=seed, n_jobs=NJ, verbose=-1)
    if name == 'cat': return CatBoostClassifier(**p, random_seed=seed, verbose=0, thread_count=NJ, allow_writing_files=False)
    if name == 'et': return ExtraTreesClassifier(**p, random_state=seed, n_jobs=NJ)
    if name == 'rf': return RandomForestClassifier(**p, random_state=seed, n_jobs=NJ)


def mats(name, A, F):
    M = pd.concat([A.reset_index(drop=True), F.reset_index(drop=True)], axis=1)
    return M.fillna(pd.concat([fill, F.median()])) if name in ('et', 'rf') else M


def feats(fit_idx, apply_list, seed):
    F_fit, Fa = label_features(XA, y, fit_idx, apply_list, tr, te, seed=seed, spec=BEST_SPEC)
    if not GS:
        return F_fit, Fa
    G_fit, Ga = graph_label_features(G, y, fit_idx, apply_list, GS, seed=seed)
    return (pd.concat([F_fit, G_fit], axis=1),
            [pd.concat([a.reset_index(drop=True), b.reset_index(drop=True)], axis=1) for a, b in zip(Fa, Ga)])


splits = H.make_splits(tr, y, n_rep=int(os.environ.get('NREP', '5')), seed0=SPLITSEED)
DO_FULL = os.environ.get('DO_FULL', '1') == '1'
R = max(s[0] for s in splits) + 1
oof = {nm: np.full((R, n), np.nan) for nm in MODELS}
tst = {nm: np.zeros(m) for nm in MODELS}
t0 = time.time()
for k, (r, fit_idx, val_idx, elig) in enumerate(splits):
    F_fit, (F_val, F_te) = feats(fit_idx, [val_idx, test_idx], k)
    for nm in MODELS:
        mdl = mk(nm, k).fit(mats(nm, X.iloc[fit_idx], F_fit), y[fit_idx])
        pv = mdl.predict_proba(mats(nm, X.iloc[val_idx], F_val))[:, 1]
        oof[nm][r, val_idx[elig]] = pv[elig]
        tst[nm] += mdl.predict_proba(mats(nm, Xt, F_te))[:, 1] / len(splits)
    print(f'fold {k} done {time.time()-t0:.0f}s', flush=True)

full = {nm: np.zeros(m) for nm in MODELS}
for s in (range(3) if DO_FULL else []):
    F_fit, (F_te,) = feats(np.arange(n), [test_idx], s)
    for nm in MODELS:
        mdl = mk(nm, s).fit(mats(nm, X, F_fit), y)
        full[nm] += mdl.predict_proba(mats(nm, Xt, F_te))[:, 1] / 3
    print(f'full refit seed {s} done {time.time()-t0:.0f}s', flush=True)

for nm in MODELS:
    aucs = np.array([roc_auc_score(y[~np.isnan(oof[nm][r])], oof[nm][r][~np.isnan(oof[nm][r])]) for r in range(R)])
    res = dict(oof=oof[nm], aucs=aucs, auc=float(aucs.mean()), test_foldavg=tst[nm])
    H.save(f'{PREFIX}_{nm}', res, full[nm] if DO_FULL else None)
    c = np.corrcoef(tst[nm], full[nm])[0, 1] if DO_FULL else float('nan')
    print(f'[{PREFIX}_{nm}] AUC {aucs.mean():.5f} per-rep {np.round(aucs, 4)}  corr(foldavg,full)={c:.4f}', flush=True)

"""Charlie's label-free 115-col view members (ch_et_proj / ch_lgbm_proj, same configs as work/ch/chmodel.py and
work/main/conf_ch.py) + graph-propagated label features (agent r2 `graph`).
python -I graph_ch.py MODEL PREFIX SEED0 NREP GSET [--full]
  MODEL in {et, lgbm}; GSET in graph_specs.GSETS; --full: also refit on all training rows (3 seeds) -> test_full.
Graph features: seeds = fit rows only (inner StratifiedGroupKFold by uri for the fit rows' own values);
validation/test rows are unlabelled nodes. Saves results/<PREFIX>_ch_<MODEL>.npz"""
import json, os, sys, time, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/ch'); sys.path.insert(0, WORK + '/r2_graph')
import harness as H
from ch_feats import charlie_pool
from graph_lib import SongGraph, graph_label_features
from graph_specs import GSPECS, GSETS
warnings.filterwarnings('ignore')
MODEL, PREFIX, SEED0, NREP, GSET = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
GS = {g: GSPECS[g] for g in GSETS[GSET]}
tr, te, y = H.load()
n, m = len(tr), len(te)
test_idx = np.arange(n, n + m)
F, BL = charlie_pool(tr, te)
XP = F[BL['PROJECT']].astype(np.float32)
X, Xt = XP.iloc[:n].reset_index(drop=True), XP.iloc[n:].reset_index(drop=True)
P = json.load(open(WORK + '/best_params_v3.json'))
uri = tr['uri'].to_numpy()
G = SongGraph(tr, te)
splits = H.make_splits(tr, y, n_rep=NREP, seed0=SEED0)


def weights(idx):
    if MODEL != 'et':
        return None
    u = pd.Series(uri[idx])
    w = 1.0 / u.map(u.value_counts()).to_numpy(float)
    return w / w.mean()


def fit_predict_rows(fit_idx, apply_idx, seed, kfeat):
    Gf, Ga = graph_label_features(G, y, fit_idx, apply_idx, GS, seed=kfeat)
    A = pd.concat([X.iloc[fit_idx].reset_index(drop=True), Gf], axis=1)
    XA = pd.concat([X, Xt], ignore_index=True)
    Bs = [pd.concat([XA.iloc[ai].reset_index(drop=True), g], axis=1) for ai, g in zip(apply_idx, Ga)]
    if MODEL == 'et':
        med = A.median()
        mdl = ExtraTreesClassifier(n_estimators=900, max_depth=15, min_samples_leaf=2, max_features=0.1827077108658931,
                                   n_jobs=2, random_state=seed)
        mdl.fit(A.fillna(med), y[fit_idx], sample_weight=weights(fit_idx))
        return [mdl.predict_proba(B.fillna(med))[:, 1] for B in Bs]
    mdl = LGBMClassifier(**P['lgbm'], random_state=seed, n_jobs=2, verbose=-1).fit(A, y[fit_idx])
    return [mdl.predict_proba(B)[:, 1] for B in Bs]


t = time.time()
res = H.run_cv(lambda fi, vi, k: tuple(fit_predict_rows(fi, [vi, test_idx], k, k)), splits, y, tag=f'{PREFIX} ch {MODEL}')
print(f'   ({time.time() - t:.0f}s)', flush=True)
full = None
if '--full' in sys.argv:
    full = np.mean([fit_predict_rows(np.arange(n), [test_idx], s, s)[0] for s in (0, 1, 2)], axis=0)
    print('corr(test_full, test_foldavg) =', np.corrcoef(full, res['test_foldavg'])[0, 1].round(4))
H.save(f'{PREFIX}_ch_{MODEL}', res, full)
print('saved', f'{PREFIX}_ch_{MODEL}')

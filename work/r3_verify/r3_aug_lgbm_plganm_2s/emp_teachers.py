"""Val-label invariance of the teacher blocks used as pseudo-label sources (fold 0, rep 0):
zoo_cat_alb (offset-0 member) and graph_hc2_lgbm. Reproduce saved OOF, then flip ALL val labels and re-run."""
import os, sys, time, json, numpy as np, pandas as pd
WORK = './work'
for p in (WORK, WORK + '/main', WORK + '/te', WORK + '/r2_graph', WORK + '/r2_zoo'): sys.path.insert(0, p)
os.environ['NJ'] = '2'
import harness as H
from label_features import label_features, BEST_SPEC
K = 0
t0 = time.time()
tr, te, y = H.load(); n, m = len(tr), len(te); test_idx = np.arange(n, n + m)
r, fit_idx, val_idx, elig = H.make_splits(tr, y, n_rep=5, seed0=1000)[K]
yf = y.copy(); yf[val_idx] = 1 - yf[val_idx]
XA0 = np.zeros((n + m, 1))
# ---------------- zoo_cat_alb (single offset 0) ----------------
import zoo as Z
I = Z.Inputs(Z.get_cache(1000, 5)); f = I.D['folds'][K]
assert np.array_equal(f['fit_idx'], fit_idx) and np.array_equal(f['val_idx'], val_idx)
def zoo_run(yy):
    F_fit, (F_val, F_te) = label_features(XA0, yy, fit_idx, [val_idx, test_idx], tr, te, seed=K, spec=BEST_SPEC)
    Mf, Mv = I.mat(fit_idx, F_fit, 'cat_alb'), I.mat(val_idx, F_val, 'cat_alb')
    return Z.model('cat_alb', K + 0).fit(Mf, yy[fit_idx]).predict(Mv)
raw = np.load(WORK + '/r2_zoo/raw/1000_5_cat_alb_off0.npz')['oof'][r, val_idx[elig]]
p0 = zoo_run(y)
print('zoo_cat_alb fold0 reproduce raw off0 OOF maxabs', np.abs(p0[elig] - raw).max(), f'{time.time()-t0:.0f}s', flush=True)
p1 = zoo_run(yf)
print('zoo_cat_alb FLIP ALL VAL LABELS -> val preds identical:', bool(np.array_equal(p0, p1)), 'maxabs', np.abs(p0 - p1).max(), f'{time.time()-t0:.0f}s', flush=True)
# ---------------- graph_hc2_lgbm ----------------
from features_final import build_features_final
from graph_lib import SongGraph, graph_label_features
from graph_specs import GSPECS, GSETS
from lightgbm import LGBMClassifier
P = json.load(open(WORK + '/main/final_params.json'))
GS = {g: GSPECS[g] for g in (GSETS['hc2'] if not isinstance(GSETS['hc2'], dict) else GSETS['hc2']['g'])}
print('graph specs', GS, flush=True)
X, Xt = build_features_final(tr, te); XA = pd.concat([X, Xt], ignore_index=True)
G = SongGraph(tr, te)
print('features/graph built', f'{time.time()-t0:.0f}s', flush=True)
def g_run(yy):
    F_fit, (F_val, F_te) = label_features(XA, yy, fit_idx, [val_idx, test_idx], tr, te, seed=K, spec=BEST_SPEC)
    G_fit, (G_val, G_te) = graph_label_features(G, yy, fit_idx, [val_idx, test_idx], GS, seed=K)
    Ff = pd.concat([F_fit, G_fit], axis=1); Fv = pd.concat([F_val.reset_index(drop=True), G_val.reset_index(drop=True)], axis=1)
    Mf = pd.concat([X.iloc[fit_idx].reset_index(drop=True), Ff.reset_index(drop=True)], axis=1)
    Mv = pd.concat([X.iloc[val_idx].reset_index(drop=True), Fv.reset_index(drop=True)], axis=1)
    mdl = LGBMClassifier(**P['lgbm'], random_state=K, n_jobs=2, verbose=-1).fit(Mf, yy[fit_idx])
    return mdl.predict_proba(Mv)[:, 1], (Fv.to_numpy(), )
saved = H.load_res('graph_hc2_lgbm')['oof'][r, val_idx[elig]]
q0, A0 = g_run(y)
print('graph_hc2_lgbm fold0 reproduce saved OOF maxabs', np.abs(q0[elig] - saved).max(), 'corr', np.corrcoef(q0[elig], saved)[0, 1], f'{time.time()-t0:.0f}s', flush=True)
q1, A1 = g_run(yf)
print('graph_hc2_lgbm FLIP ALL VAL LABELS -> val label-feats identical:', bool(np.array_equal(A0[0], A1[0], equal_nan=True)),
      ' val preds identical:', bool(np.array_equal(q0, q1)), 'maxabs', np.abs(q0 - q1).max(), f'{time.time()-t0:.0f}s', flush=True)

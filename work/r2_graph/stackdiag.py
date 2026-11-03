"""Diagnostic: does a propagated graph score add information to the final8 blend? (nested LR on 2 inputs)"""
import sys, numpy as np
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/r2_graph')
import harness as H
import graph_lib as GL
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
tr, te, y = H.load(); n = len(tr)
MEM = ['fkd_lgbm', 'fkt_xgb', 'fkt_cat', 'fkd_et', 'fkd_rf', 'mdl_mlp_ple', 'ch_et_proj', 'ch_lgbm_proj']
res = [H.load_res(nm) for nm in MEM]
G = GL.SongGraph(tr, te)
splits = H.make_splits(tr, y, n_rep=5, seed0=1000)
from graph_specs import GSPECS
specs = {k: GSPECS[k] for k in ('g_hc2', 'g_hk2', 'g_hknn20', 'g_hcomb')}
oof = {k: np.full((5, n), np.nan) for k in specs}
for k, (r, fit_idx, val_idx, elig) in enumerate(splits):
    for nm, s in specs.items():
        f, _ = GL.propagate(G, y, fit_idx, s)
        oof[nm][r, val_idx[elig]] = f[G.node[val_idx[elig]]]
for nm in specs:
    g0, g1 = [], []
    for r in range(5):
        msk = ~np.isnan(res[0]['oof'][r])
        b = np.mean([rankdata(rr['oof'][r, msk]) / msk.sum() for rr in res], axis=0)
        g = oof[nm][r, msk]
        lg = np.log(np.clip(g, 1e-3, 1 - 1e-3) / (1 - np.clip(g, 1e-3, 1 - 1e-3)))
        yr = y[msk]
        Xb = np.column_stack([np.log(b / (1 - b + 1e-9) + 1e-9)])
        Xc = np.column_stack([Xb, lg])
        pb, pc = np.zeros(len(yr)), np.zeros(len(yr))
        for a, bb in StratifiedKFold(5, shuffle=True, random_state=r).split(Xb, yr):
            pb[bb] = LogisticRegression().fit(Xb[a], yr[a]).decision_function(Xb[bb])
            pc[bb] = LogisticRegression().fit(Xc[a], yr[a]).decision_function(Xc[bb])
        g0.append(roc_auc_score(yr, pb)); g1.append(roc_auc_score(yr, pc))
    d = np.array(g1) - np.array(g0)
    print(f'{nm:10s} blend {np.mean(g0):.5f} +graph {np.mean(g1):.5f}  diff {d.mean():+.5f} pos {(d>0).sum()}/5 {np.round(d,4)}')

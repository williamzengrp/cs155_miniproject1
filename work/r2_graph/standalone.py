"""Standalone AUC (outer val, eligible rows, pooled per repeat) of graph-propagated scores vs te label feats."""
import sys, time, numpy as np, pandas as pd
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/te'); sys.path.insert(0, WORK + '/r2_graph')
import harness as H
from label_features import label_features, BEST_SPEC, get_ctx, encode
from graph_lib import SongGraph, propagate
from sklearn.metrics import roc_auc_score
tr, te, y = H.load()
n, m = len(tr), len(te)
G = SongGraph(tr, te)
print('nodes', G.Nn)
splits = H.make_splits(tr, y, n_rep=5, seed0=1000)
SPECS = {
 'h_date_m3': dict(method='harm', graph=dict(w_date=1.0, k=0), m=3),
 'h_knn20_t1': dict(method='harm', graph=dict(w_date=0.0, k=20, tau=1.0), m=1),
 'h_knn20_t05': dict(method='harm', graph=dict(w_date=0.0, k=20, tau=0.5), m=1),
 'h_knn40_t1': dict(method='harm', graph=dict(w_date=0.0, k=40, tau=1.0), m=1),
 'h_comb_d1_k20': dict(method='harm', graph=dict(w_date=1.0, k=20, tau=1.0), m=1),
 'h_comb_d3_k20': dict(method='harm', graph=dict(w_date=3.0, k=20, tau=1.0), m=1),
 'h_comb_d1_k10': dict(method='harm', graph=dict(w_date=1.0, k=10, tau=1.0), m=1),
 'p_comb_d1_k20_a5': dict(method='ppr', graph=dict(w_date=1.0, k=20, tau=1.0), alpha=0.5, m=1),
 'p_comb_d1_k20_a8': dict(method='ppr', graph=dict(w_date=1.0, k=20, tau=1.0), alpha=0.8, m=1),
 'p_knn20_a5': dict(method='ppr', graph=dict(w_date=0.0, k=20, tau=1.0), alpha=0.5, m=1),
}
R = 5
oof = {k: np.full((R, n), np.nan) for k in list(SPECS) + list(BEST_SPEC)}
ctx = get_ctx(tr, te)
t0 = time.time()
for k, (r, fit_idx, val_idx, elig) in enumerate(splits):
    for name, s in SPECS.items():
        f, _ = propagate(G, y, fit_idx, s)
        oof[name][r, val_idx[elig]] = f[G.node[val_idx[elig]]]
    vals = np.zeros(ctx.N); vals[:n] = y
    wts = np.zeros(ctx.N); wts[fit_idx] = 1
    e = encode(ctx, vals, wts, val_idx[elig], BEST_SPEC)
    for c in BEST_SPEC:
        oof[c][r, val_idx[elig]] = e[c].to_numpy()
print('time', time.time() - t0)
for name, o in oof.items():
    a = [roc_auc_score(y[~np.isnan(o[r])], o[r][~np.isnan(o[r])]) for r in range(R)]
    print(f'{name:22s} {np.mean(a):.4f}  {np.round(a, 4)}')
np.savez(WORK + '/r2_graph/standalone_oof.npz', **oof)

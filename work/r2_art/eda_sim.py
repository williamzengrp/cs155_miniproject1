"""Univariate AUC on eligible val rows (std repeat 0) of full-pool vs fold-consistent artifact features,
and partial signal after the date TE / kNN label feature (rank-residual check). Labels used only for evaluation."""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
W = './work'
sys.path.insert(0, W); sys.path.insert(0, W + '/r2_art'); sys.path.insert(0, W + '/te')
import harness as H
from art_feats import art_features, art_features_sim
tr, te, y = H.load(); n, m = len(tr), len(te); N = n + m
splits = H.make_splits(tr, y, n_rep=5, seed0=1000)
full = art_features(tr, te, blocks=('date', 'knn'))
vals_full, vals_sim, ys = {}, {}, []
for k, (r, fit_idx, val_idx, elig) in enumerate(splits[:5]):
    trm = np.zeros(N, bool); trm[fit_idx] = True
    tem = np.zeros(N, bool); tem[val_idx[elig]] = True
    S = art_features_sim(tr, te, trm, tem, blocks=('date', 'knn'))
    ev = val_idx[elig]
    ys.append(y[ev])
    for c in S.columns:
        vals_sim.setdefault(c, []).append(S[c].to_numpy()[ev])
        vals_full.setdefault(c, []).append(full[c].to_numpy()[ev])
yy = np.concatenate(ys)
for c in vals_sim:
    a = np.concatenate(vals_full[c]); b = np.concatenate(vals_sim[c])
    oka, okb = ~np.isnan(a), ~np.isnan(b)
    print(f'{c:22s} full-pool AUC {roc_auc_score(yy[oka], a[oka]):.4f}   fold-consistent AUC {roc_auc_score(yy[okb], b[okb]):.4f}')

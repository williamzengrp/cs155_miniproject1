import sys, numpy as np
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/r2_graph')
import harness as H
import graph_lib as GL
from sklearn.metrics import roc_auc_score
tr, te, y = H.load(); n = len(tr)
G = GL.SongGraph(tr, te)
splits = H.make_splits(tr, y, n_rep=5, seed0=1000)
def run(fn):
    oof = np.full((5, n), np.nan)
    for k, (r, fit_idx, val_idx, elig) in enumerate(splits):
        f = fn(fit_idx)
        oof[r, val_idx[elig]] = f[G.node[val_idx[elig]]]
    a = [roc_auc_score(y[~np.isnan(oof[r])], oof[r][~np.isnan(oof[r])]) for r in range(5)]
    return np.mean(a)
def harm(fit_idx, W, m=0.3, prior='global', it=60):
    lab, val, _ = G.node_seeds(y, fit_idx)
    pr, g = G.year_prior(lab, val)
    if prior == 'global': pr = np.full(G.Nn, g)
    return GL.harmonic(W, lab, val, pr, m=m, n_iter=it)
for kw in [dict(w_date=0.0, k=20, tau=2.0, kernel='gauss'),
           dict(w_date=0.0, k=30, tau=2.0, kernel='gauss'),
           dict(w_date=0.0, k=20, tau=3.0, kernel='gauss'),
           dict(w_date=0.3, date_bw=1.5, k=20, tau=2.0, kernel='gauss'),
           dict(w_date=0.3, date_bw=1.5, k=20, tau=2.0, kernel='inv'),
           dict(w_date=1.0, date_bw=1.0, k=20, tau=2.0, kernel='gauss'),
           dict(w_date=0.3, date_bw=1.5, k=30, tau=2.0, kernel='gauss')]:
    Wk = G.build(**kw)
    for m in (0.1, 0.3, 1.0):
        print(kw, 'm', m, round(run(lambda f: harm(f, Wk, m=m)), 4), flush=True)

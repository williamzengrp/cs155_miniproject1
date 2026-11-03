import sys, numpy as np
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/r2_graph')
import harness as H
import graph_lib as GL
from sklearn.metrics import roc_auc_score
tr, te, y = H.load(); n = len(tr)
G = GL.SongGraph(tr, te)
splits = H.make_splits(tr, y, n_rep=5, seed0=1000)
def run(spec):
    oof = np.full((5, n), np.nan)
    for k, (r, fit_idx, val_idx, elig) in enumerate(splits):
        f, _ = GL.propagate(G, y, fit_idx, spec)
        oof[r, val_idx[elig]] = f[G.node[val_idx[elig]]]
    a = [roc_auc_score(y[~np.isnan(oof[r])], oof[r][~np.isnan(oof[r])]) for r in range(5)]
    return round(np.mean(a), 4)
g = dict(w_date=0.3, date_bw=1.5, k=20, tau=2.0, kernel='inv')
print('hc2 (global prior m0.3)', run(dict(method='harm', graph=g, m=0.3, prior='global')))
print('hc2 year prior        ', run(dict(method='harm', graph=g, m=0.3)))
for a in (0.3, 0.6, 0.9):
    print('ppr alpha', a, run(dict(method='ppr', graph=g, alpha=a, m=0.3, prior='global')))
for wd, bw in ((0.3, 1.0), (0.3, 2.0), (0.6, 1.5), (0.15, 1.5)):
    print('harm wd', wd, 'bw', bw, run(dict(method='harm', graph=dict(w_date=wd, date_bw=bw, k=20, tau=2.0, kernel='inv'), m=0.3, prior='global')))
for k, tau in ((15, 2.0), (20, 1.5), (20, 2.5)):
    print('harm k', k, 'tau', tau, run(dict(method='harm', graph=dict(w_date=0.3, date_bw=1.5, k=k, tau=tau, kernel='inv'), m=0.3, prior='global')))

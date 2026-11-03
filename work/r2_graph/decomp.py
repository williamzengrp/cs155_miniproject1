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
W20 = G.build(w_date=0.0, k=20, tau=1.0)
def harm(fit_idx, W=W20, m=1.0, it=60, prior='year'):
    lab, val, _ = G.node_seeds(y, fit_idx)
    pr, g = G.year_prior(lab, val)
    if prior == 'global': pr = np.full(G.Nn, g)
    return GL.harmonic(W, lab, val, pr, m=m, n_iter=it)
print('year prior only      ', round(run(lambda f: G.year_prior(*G.node_seeds(y, f)[:2])[0]), 4))
print('harm k20 m1 it60     ', round(run(lambda f: harm(f)), 4))
print('harm k20 m1 it1      ', round(run(lambda f: harm(f, it=1)), 4))
print('harm k20 m1 global   ', round(run(lambda f: harm(f, prior='global')), 4))
print('harm k20 m0.1        ', round(run(lambda f: harm(f, m=0.1)), 4))
print('harm k20 m3          ', round(run(lambda f: harm(f, m=3.0)), 4))
for k, tau in [(10, 1.0), (20, 2.0), (20, 0.7), (30, 1.0), (60, 1.0)]:
    Wk = G.build(w_date=0.0, k=k, tau=tau)
    print(f'harm k{k} tau{tau}      ', round(run(lambda f: harm(f, W=Wk)), 4))
Wg = G.build(w_date=0.0, k=20, tau=1.0, kernel='gauss')
print('harm k20 gauss       ', round(run(lambda f: harm(f, W=Wg)), 4))
for wd in (0.1, 0.3):
    Wc = G.build(w_date=wd, k=20, tau=1.0)
    print(f'harm comb wd{wd}     ', round(run(lambda f: harm(f, W=Wc)), 4))
Wb = G.build(w_date=0.3, date_bw=1.5, k=20, tau=1.0)
print('harm comb wd0.3 bw1.5', round(run(lambda f: harm(f, W=Wb)), 4))

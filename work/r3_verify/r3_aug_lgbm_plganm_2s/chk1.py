import sys, numpy as np
WORK = './work'
sys.path.insert(0, WORK)
import harness as H
from scipy.stats import spearmanr
tr, te, y = H.load()
R2 = H.load_res('r3_aug_lgbm_plganm_2s'); A = H.load_res('r3_aug_lgbm_plganm'); B = H.load_res('r3_aug_lgbm_plganm_s100')
for k in ('oof', 'test_foldavg', 'test_full'):
    a = np.nanmax(np.abs(R2[k] - (A[k] + B[k]) / 2)); print('2s vs mean', k, R2[k].shape, a)
print('aucs', R2['aucs'], A['aucs'], B['aucs'])
sp = H.make_splits(tr, y, n_rep=5, seed0=1000)
nonel = np.zeros((5, len(y)), bool)
for r, fi, vi, el in sp: nonel[r, vi[~el]] = True
for nm in ('chrep_bl', 'graph_hc2_lgbm', 'zoo_cat_alb'):
    T = H.load_res(nm)
    o = T['oof']
    print(nm, 'oof', o.shape, 'NaN==nonelig per rep', [bool((np.isnan(o[r]) == nonel[r]).all()) for r in range(o.shape[0])],
          'aucs', np.round(T['aucs'], 4), 'tf', T['test_full'].shape, 'tfa', T['test_foldavg'].shape,
          'spearman(tf,tfa)', round(spearmanr(T['test_full'], T['test_foldavg'])[0], 4) if T['test_foldavg'].size else None,
          'tf range', float(T['test_full'].min()), float(T['test_full'].max()), 'mean', float(T['test_full'].mean()))
# check alternative seeds: would another split seed produce the same NaN pattern?
sp2 = H.make_splits(tr, y, n_rep=1, seed0=2026)
non2 = np.zeros(len(y), bool)
for r, fi, vi, el in sp2: non2[vi[~el]] = True
print('control: seed2026 nonelig matches chrep rep0?', bool((np.isnan(H.load_res('chrep_bl')['oof'][0]) == non2).all()), 'n nonelig', nonel[0].sum(), non2.sum())

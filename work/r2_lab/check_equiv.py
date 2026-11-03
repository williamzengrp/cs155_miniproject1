"""Sanity: labx BEST_SPEC == te label_features BEST_SPEC on one fold; time Ctx + one LGBM fit."""
import sys, time, json
import numpy as np, pandas as pd
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/main'); sys.path.insert(0, WORK + '/te'); sys.path.insert(0, WORK + '/r2_lab')
import harness as H
from features_final import build_features_final
import label_features as LF
import labx as LX
from lightgbm import LGBMClassifier
tr, te, y = H.load()
t0 = time.time(); X, Xt = build_features_final(tr, te); print('feats', time.time() - t0, X.shape)
XA = pd.concat([X, Xt], ignore_index=True)
n, m = len(tr), len(te)
splits = H.make_splits(tr, y, n_rep=1)
r, fi, vi, el = splits[2]
t0 = time.time(); LX.get_ctx(tr, te); print('ctx labx', time.time() - t0)
t0 = time.time(); LF.get_ctx(tr, te); print('ctx te', time.time() - t0)
T = np.arange(n, n + m)
A = LF.label_features(XA, y, fi, [vi, T], tr, te, seed=2, spec=LF.BEST_SPEC)
t0 = time.time()
Bx = LX.label_features(XA, y, fi, [vi, T], tr, te, seed=2, spec=LX.BEST_SPEC); print('lf time', time.time() - t0)
for p, q in zip([A[0]] + A[1], [Bx[0]] + Bx[1]):
    print('max abs diff', np.abs(p.to_numpy() - q.to_numpy()).max())
spec = ['knnw_yr_10', 'knnw_yr_150', 'knnw_yr05_40', 'knnw_yr3_40', 'knnw_yrA_40', 'knnw_yrB_40', 'dmin_hi', 'dmin_lo',
        'tek_date_m10', 'chain_year-ym-date-alb_m10', 'teh_yc8_m10']
t0 = time.time()
F = LX.label_features(XA, y, fi, [vi], tr, te, seed=2, spec=spec, dedup=True); print('lf ext time', time.time() - t0)
print(F[0].describe().T[['mean', 'std', 'min', 'max']])
print(F[1][0].describe().T[['mean', 'std', 'min', 'max']])
P = json.load(open(WORK + '/main/final_params.json'))
Mf = pd.concat([X.iloc[fi].reset_index(drop=True), Bx[0]], axis=1)
t0 = time.time(); LGBMClassifier(**P['lgbm'], random_state=0, n_jobs=2, verbose=-1).fit(Mf, y[fi]); print('lgbm fit', time.time() - t0)

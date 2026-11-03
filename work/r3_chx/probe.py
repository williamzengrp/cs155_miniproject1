import sys, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, pickle
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/main'); sys.path.insert(0, WORK + '/te'); sys.path.insert(0, WORK + '/ch')
import harness as H
tr, te, y = H.load()
t0 = time.time()
from features_final import build_features_final
X, Xt = build_features_final(tr, te)
print('bff', time.time() - t0, X.shape, flush=True)
cols = ['a_global_nn_dist', 'a_nn2_dist', 'n_same_date_audio', 'a_n_close10', 'a_n_close20', 'a_n_close30']
print([c for c in cols if c in X.columns])
print(list(X.columns))
print(X[cols].describe().T)
from ch_feats import charlie_pool
t0 = time.time()
F, BL = charlie_pool(tr, te)
print('pool', time.time() - t0)
P = F[BL['PROJECT']]
n = len(tr)
XA = pd.concat([X, Xt], ignore_index=True)
for c in cols:
    v = XA[c].to_numpy(float)
    cc = []
    for pc in P.columns:
        w = P[pc].to_numpy(float)
        ok = np.isfinite(v) & np.isfinite(w)
        if ok.sum() > 100 and np.std(w[ok]) > 0:
            from scipy.stats import spearmanr
            cc.append((abs(spearmanr(v[ok], w[ok])[0]), pc))
    cc.sort(reverse=True)
    print(c, 'NaN frac', np.isnan(v).mean(), 'top corr in charlie view:', [(round(a, 3), b) for a, b in cc[:3]])
D = pickle.load(open(WORK + '/r2_zoo/cache/lf_1000_5.pkl', 'rb'))
spl = H.make_splits(tr, y, n_rep=5)
for k in range(25):
    f = D['folds'][k]
    assert (f['fit_idx'] == spl[k][1]).all() and (f['val_idx'] == spl[k][2]).all() and (f['elig'] == spl[k][3]).all()
print('zoo cache folds match std splits')
print(D['folds'][0]['F_fit'].head(), D['folds'][0]['F_fit'].isna().sum().to_dict())
from label_features import label_features, BEST_SPEC, get_ctx
t0 = time.time(); ctx = get_ctx(tr, te); print('ctx', time.time() - t0)
t0 = time.time()
f = D['folds'][0]
F_fit, (F_val, F_te) = label_features(XA, y, f['fit_idx'], [f['val_idx'], np.arange(n, n + len(te))], tr, te, seed=0, spec=BEST_SPEC)
print('lf', time.time() - t0, np.abs(F_fit.to_numpy() - f['F_fit'].to_numpy()).max(), np.abs(F_te.to_numpy() - f['F_te'].to_numpy()).max())
alb = ctx.key['alb']
print('alb n unique', len(np.unique(alb)), 'train', len(np.unique(alb[:n])), 'test', len(np.unique(alb[n:])), 'test albs seen in train', np.isin(np.unique(alb[n:]), alb[:n]).mean())
print((D['cats']['alb'] == alb).all())

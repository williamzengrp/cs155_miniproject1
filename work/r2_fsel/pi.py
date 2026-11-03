"""In-fold permutation importance (PI): for each OUTER fold, only the fit part is used. Inner 5-fold
StratifiedGroupKFold by uri (random_state = outer fold no., i.e. the same inner folds the OOF label features
were built with). LightGBM (final params) trained on inner-fit rows; AUC drop on inner-val rows (one row per uri)
when one feature / one group of features is permuted (2 permutation repeats).
python -I pi.py SEED0 NREP [model] -> cache/pi_<model>_<SEED0>_<NREP>.pkl  {'feat': (K, nfeat), 'group': (K, ngroup)}"""
import sys, time, pickle
import numpy as np, pandas as pd
sys.path.insert(0, './work/r2_fsel')
import fsel_lib as L
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import harness as H
SEED0, NREP = int(sys.argv[1]), int(sys.argv[2])
MODEL = sys.argv[3] if len(sys.argv) > 3 else 'lgbm'
folds = L.load_folds(SEED0, NREP)
tr, te, _ = H.load()
uri = pd.factorize(tr['uri'])[0]
y = L.y
GN = list(L.GROUPS)
PF = np.zeros((len(folds), len(L.ALL)))
PG = np.zeros((len(folds), len(GN)))
t0 = time.time()
for k, f in enumerate(folds):
    fit = f['fit_idx']
    M = L.mat(MODEL, L.ALL, L.X.iloc[fit], f['F_fit']).reset_index(drop=True)
    yf = y[fit]
    sg = StratifiedGroupKFold(5, shuffle=True, random_state=k)
    pf, pg, w = np.zeros(len(L.ALL)), np.zeros(len(GN)), 0
    for a, b in sg.split(fit, yf, uri[fit]):
        b = b[~pd.Series(uri[fit][b]).duplicated().to_numpy()]          # one row per song (test-like)
        mdl = L.mk(MODEL, k).fit(M.iloc[a], yf[a])
        Mb = M.iloc[b].reset_index(drop=True)
        base = roc_auc_score(yf[b], mdl.predict_proba(Mb)[:, 1])
        rng = np.random.default_rng(1000 * k + w)
        for rep in range(2):
            perm = rng.permutation(len(b))
            for j, c in enumerate(L.ALL):
                Mp = Mb.copy(); Mp[c] = Mb[c].to_numpy()[perm]
                pf[j] += base - roc_auc_score(yf[b], mdl.predict_proba(Mp)[:, 1])
            for g, gn in enumerate(GN):
                Mp = Mb.copy()
                for c in L.GROUPS[gn]:
                    Mp[c] = Mb[c].to_numpy()[perm]
                pg[g] += base - roc_auc_score(yf[b], mdl.predict_proba(Mp)[:, 1])
        w += 1
    PF[k], PG[k] = pf / (2 * w), pg / (2 * w)
    print(f'outer fold {k} {time.time()-t0:.0f}s', flush=True)
pickle.dump(dict(feat=PF, group=PG, feat_names=L.ALL, group_names=GN), open(L.C + f'pi_{MODEL}_{SEED0}_{NREP}.pkl', 'wb'))
print('mean group PI:', dict(zip(GN, np.round(PG.mean(0), 5))))
o = np.argsort(PF.mean(0))
for j in o:
    print(f'{L.ALL[j]:24s} {PF[:, j].mean():+.5f}  frac>0 {(PF[:, j] > 0).mean():.2f}')

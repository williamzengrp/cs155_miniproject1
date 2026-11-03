"""r2_eda: the mdl agent's MLP-PLE member (mdl_mlp_ple recipe: old 67 teammate features, PLE16, 256-128, drop .4,
wd 1e-3, 40 epochs, 5 seeds) PLUS the te agent's out-of-fold label features (BEST_SPEC: knnw_yr_40, teh_date_m10,
cnt_date). Motivation (r2_eda/eda5.out): the label-free MLP member is ~0.024 AUC below the GBMs on rows with >= 4
labelled date-mates. Runs in the torch venv (work/mdl/venv).

python -I mlplab.py SEED0 NREP SAVE_NAME        env DO_FULL=1 -> test_full from a full refit (3 label-feature seeds x
                                                3 MLP seeds); NJ threads (default 2)
Leakage: label features exactly as main/final_cv.py (fit rows out-of-fold via inner StratifiedGroupKFold grouped by
uri, seed = fold number; val/test rows from fit_idx labels only). PLE / scaling fit on fit rows only.
"""
import os, sys, time, warnings
WORK = './work'
sys.path.insert(0, WORK + '/mdl'); sys.path.insert(0, WORK + '/te'); sys.path.insert(0, WORK + '/r2_eda')
import numpy as np
import pandas as pd
import torch
import torch_mlp as TM          # sets torch threads from NJ, loads the 67 teammate features
import common as C
from label_features import BEST_SPEC
from eda_label import label_features_ext
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
H = C.H
seed0, nrep, save = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
DO_FULL = os.environ.get('DO_FULL', '0') == '1'
P = {"feats": "all", "h": [256, 128], "drop": 0.4, "wd": 0.001, "seeds": 5, "ple": 16, "epochs": 40}
tr, te, y, X, Xt = C.data()
n, m = len(tr), len(te)
XA = pd.concat([X, Xt], ignore_index=True)
test_idx = np.arange(n, n + m)
splits = H.make_splits(tr, y, n_rep=nrep, seed0=seed0)
EP = P['epochs']


def withlab(A, F):
    return pd.concat([A.reset_index(drop=True), F.reset_index(drop=True)], axis=1)


def fit_apply(Xf, yf, others, seeds):
    A, Os = TM.prep(P, Xf, others)
    acc = [0] * len(Os)
    for s in seeds:
        o = TM.train_predict(P, A, yf, Os, seed=s, snaps=(EP,))[EP]
        acc = [a + v / len(seeds) for a, v in zip(acc, o)]
    return acc


R = max(s[0] for s in splits) + 1
oof = np.full((R, n), np.nan)
tst = 0
t0 = time.time()
for k, (r, fi, vi, el) in enumerate(splits):
    F_fit, (F_val, F_te) = label_features_ext(XA, y, fi, [vi, test_idx], tr, te, seed=k, spec=BEST_SPEC)
    pv, pt = fit_apply(withlab(X.iloc[fi], F_fit), y[fi], [withlab(X.iloc[vi], F_val), withlab(Xt, F_te)],
                       [1000 * k + s for s in range(P['seeds'])])
    oof[r, vi[el]] = 1 / (1 + np.exp(-pv[el]))
    tst = tst + 1 / (1 + np.exp(-pt)) / len(splits)
    if k % 5 == 4:
        print(f'fold {k} {time.time()-t0:.0f}s', flush=True)
aucs = np.array([roc_auc_score(y[~np.isnan(oof[r])], oof[r][~np.isnan(oof[r])]) for r in range(R)])
res = dict(oof=oof, aucs=aucs, auc=float(aucs.mean()), test_foldavg=tst)
full = None
if DO_FULL:
    tf = 0
    for s in range(3):
        F_fit, (F_te,) = label_features_ext(XA, y, np.arange(n), [test_idx], tr, te, seed=s, spec=BEST_SPEC)
        tf = tf + fit_apply(withlab(X, F_fit), y, [withlab(Xt, F_te)], [77 + 10 * s + j for j in range(3)])[0] / 3
    full = 1 / (1 + np.exp(-tf))
    print('corr(full, foldavg)', np.corrcoef(full, tst)[0, 1].round(4))
H.save(save, res, full)
print(f'[{save}] AUC {aucs.mean():.5f} per-rep {np.round(aucs, 4)} ({time.time()-t0:.0f}s)', flush=True)

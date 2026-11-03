import sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
W = './work'
sys.path.insert(0, W); sys.path.insert(0, W + '/r2_art')
import harness as H
from art_feats import art_features
tr, te, y = H.load()
n = len(tr)
A = art_features(tr, te, blocks=('date', 'ym', 'year', 'knn'))
uri = pd.factorize(pd.concat([tr, te])['uri'])[0]
sing = np.bincount(uri)[uri][:n] == 1
print('train singletons', sing.sum())
for c in A.columns:
    v = A[c].to_numpy()[:n][sing]; ok = ~np.isnan(v)
    a = roc_auc_score(y[sing][ok], v[ok]) if ok.sum() > 10 else np.nan
    print(f'{c:22s} AUC(singletons) {a:.4f}  cover {ok.mean():.2f}  train mean {np.nanmean(A[c][:n]):.3f}  test mean {np.nanmean(A[c][n:]):.3f}')

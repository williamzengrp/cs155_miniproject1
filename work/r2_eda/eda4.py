"""r2_eda: diagnostic -- do labels of NEAR-date (not same-date) audio-similar songs explain the final8 residual?
LOO over all train labels (diagnostic only; never used as a feature in this form)."""
import sys
import numpy as np, pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score
WORK = './work'
sys.path.insert(0, WORK)
import harness as H
from time_features import parse_date

tr, te, y = H.load()
n = len(tr)
D = pd.read_pickle(WORK + '/r2_eda/blend_std.pkl')
SIM = ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'speechiness', 'instrumentalness', 'tempo', 'duration_ms']
al = pd.concat([tr, te], ignore_index=True)
Z = QuantileTransformer(n_quantiles=200, output_distribution='normal').fit_transform(al[SIM])
Z = (Z / Z.std(0))[:n]
d = parse_date(tr['track_album_release_date'].astype(str))
day = (d - d.min()).dt.days.to_numpy()
raw = tr['track_album_release_date'].astype(str).to_numpy()
uri = tr['uri'].to_numpy()
prec = tr['track_album_release_date'].astype(str).str.len().to_numpy()
for W, T in ((14, 1.5), (60, 1.5), (60, 99), (365, 1.0)):
    rate = np.full(n, np.nan); cnt = np.zeros(n)
    for i in range(n):
        if prec[i] != 10:
            continue
        c = np.nonzero((np.abs(day - day[i]) <= W) & (raw != raw[i]) & (uri != uri[i]) & (prec == 10))[0]
        if len(c) == 0:
            continue
        dd = np.linalg.norm(Z[c] - Z[i], axis=1)
        c = c[dd < T]
        if len(c):
            rate[i], cnt[i] = y[c].mean(), len(c)
    E = D.assign(nr=rate, nc=cnt)
    E = E[(E.n_elig > 0) & E.nr.notna()]
    rows = []
    for k, G in E.groupby(pd.cut(E.nr, [-0.01, 0.0, 0.2, 0.4, 0.6, 0.99, 1.0]), observed=True):
        z = (G.y - G.p).sum() / np.sqrt(max((G.p * (1 - G.p)).sum(), 1e-9))
        rows.append((str(k), len(G), round(G.y.mean(), 3), round(G.p.mean(), 3), round(z, 2)))
    # partial signal: AUC of nr for predicting residual sign among rows with neighbours
    print(f'window +-{W}d, audio dist<{T}: rows {len(E)}  corr(nr, y-p) = {np.corrcoef(E.nr, E.y - E.p)[0,1]:+.3f}')
    for r in rows:
        print('   ', r)

"""r2_eda: diagnostics -- (a) label agreement with nearest other-uri neighbour by audio distance,
(b) LOO signal of mate-duplicate features after controlling for the LOO date TE (logistic regression)."""
import sys
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score
WORK = './work'
sys.path.insert(0, WORK)
import harness as H

tr, te, y = H.load()
n = len(tr)
al = pd.concat([tr, te], ignore_index=True)
SIM = ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'speechiness', 'instrumentalness', 'tempo', 'duration_ms']
Z = QuantileTransformer(n_quantiles=200, output_distribution='normal').fit_transform(al[SIM])
Z = Z / Z.std(0)
uri = pd.factorize(al['uri'])[0]
raw = al['track_album_release_date'].astype(str).to_numpy()
Zt = Z[:n]
nn = NearestNeighbors(n_neighbors=8).fit(Zt)
dist, ind = nn.kneighbors(Zt)
d1 = np.full(n, np.nan); l1 = np.full(n, np.nan); same_date = np.zeros(n, bool)
for i in range(n):
    for d, j in zip(dist[i], ind[i]):
        if uri[j] != uri[i]:
            d1[i], l1[i], same_date[i] = d, y[j], raw[j] == raw[i]
            break
B = pd.DataFrame({'y': y, 'd1': d1, 'l1': l1, 'sd': same_date})
B['bin'] = pd.cut(B.d1, [0, 0.05, 0.15, 0.3, 0.5, 0.8, 1.2, 10])
print(B.groupby(['bin', 'sd'], observed=True).apply(lambda g: pd.Series({'n': len(g), 'agree': (g.y == g.l1).mean(), 'yrate': g.y.mean(), 'l1rate': g.l1.mean()})).round(3))
# test rows: distance to nearest train row
nn2 = NearestNeighbors(n_neighbors=1).fit(Zt)
dt, _ = nn2.kneighbors(Z[n:])
print('test nearest-train distance quantiles', np.round(np.quantile(dt[:, 0], [0.01, 0.05, 0.1, 0.25, 0.5]), 3))
print('train nearest other-uri train distance quantiles', np.round(np.nanquantile(d1, [0.01, 0.05, 0.1, 0.25, 0.5]), 3))

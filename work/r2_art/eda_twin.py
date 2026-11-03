import sys
import numpy as np, pandas as pd
W = './work'
sys.path.insert(0, W)
import harness as H
from time_features import AUDIO, parse_date
tr, te, y = H.load()
n, m = len(tr), len(te)
al = pd.concat([tr, te], ignore_index=True)
is_te = np.r_[np.zeros(n, bool), np.ones(m, bool)]
uri = pd.factorize(al['uri'])[0]
first = pd.Series(np.arange(n + m)).groupby(uri).first().to_numpy()
u = al.iloc[first].reset_index(drop=True)
ute = is_te[first]
for keycols, nm in ((AUDIO, 'audio13'), (['danceability', 'energy', 'loudness', 'duration_ms', 'tempo'], 'audio5'),
                    (['duration_ms'], 'duration'), (['duration_ms', 'tempo'], 'dur+tempo')):
    key = u[keycols].round(6).astype(str).agg('|'.join, axis=1)
    g = pd.DataFrame({'k': key, 'te': ute}).groupby('k')['te'].agg(['size', 'sum'])
    g = g[g['size'] >= 2]
    # count pairs by type
    pairs_tt = (g['sum'] * (g['sum'] - 1) / 2).sum()
    pairs_rr = ((g['size'] - g['sum']) * (g['size'] - g['sum'] - 1) / 2).sum()
    pairs_rt = (g['sum'] * (g['size'] - g['sum'])).sum()
    print(f'{nm:10s} groups {len(g):4d}  pairs train-train {pairs_rr:.0f}  train-test {pairs_rt:.0f}  test-test {pairs_tt:.0f}  '
          f'(random expect cross = {pairs_rr*0.5:.0f}, tt = {pairs_rr*0.0625:.0f})')
# same date + identical duration (likely same track, different uri)
key = u['track_album_release_date'].astype(str) + '|' + u['duration_ms'].astype(str)
g = pd.DataFrame({'k': key, 'te': ute}).groupby('k')['te'].agg(['size', 'sum']); g = g[g['size'] >= 2]
print('date+dur', len(g), 'rr', ((g['size']-g['sum'])*(g['size']-g['sum']-1)/2).sum(), 'rt', (g['sum']*(g['size']-g['sum'])).sum(), 'tt', (g['sum']*(g['sum']-1)/2).sum())

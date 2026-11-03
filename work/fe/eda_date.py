import sys; sys.path.insert(0, './work')
import numpy as np, pandas as pd
import harness as H
from time_features import AUDIO, parse_date
from sklearn.metrics import roc_auc_score
tr, te, y = H.load()
al = pd.concat([tr.assign(src='tr', y=y), te.assign(src='te')], ignore_index=True)
n = len(tr)
raw = al['track_album_release_date'].astype(str)
d = parse_date(raw)
al['ucnt'] = al['uri'].map(al['uri'].value_counts())
u = al.drop_duplicates('uri').copy()
ru = u['track_album_release_date'].astype(str)
gs = ru.map(ru.value_counts())
print('distinct songs:', len(u), 'distinct dates:', ru.nunique())
print('song-level date group size distribution:', gs.value_counts().sort_index().to_dict())
# label rate of singletons by date group size (songs)
al['g_songs'] = raw.map(ru.value_counts())
s1 = al[:n][al[:n].ucnt == 1]
print(s1.groupby(np.minimum(s1.g_songs, 12))['y'].agg(['mean', 'size']).T.round(2).to_string())
# by year bucket and group size
yr = d.dt.year
s1 = s1.assign(yr=yr[:n][al[:n].ucnt == 1])
s1['era'] = pd.cut(s1.yr, [0, 1989, 2009, 2014, 2019, 2021, 2023, 2030])
print(s1.pivot_table(index='era', columns=np.minimum(s1.g_songs, 6), values='y', aggfunc='mean').round(2).to_string())
print(s1.pivot_table(index='era', columns=np.minimum(s1.g_songs, 6), values='y', aggfunc='size').to_string())
# day of week: Friday share by era
print(s1.groupby(['era', d.dt.dayofweek[:n][al[:n].ucnt == 1] == 4])['y'].agg(['mean', 'size']).unstack().round(2).to_string())
# within-date group: is energy/loudness similar? within-group std vs global std
for c in ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'speechiness', 'tempo', 'duration_ms', 'key', 'mode', 'instrumentalness', 'liveness']:
    gstd = u.groupby(ru)[c].std().dropna()
    print(f'{c:18s} global std {u[c].std():10.4f}  mean within-date std {gstd.mean():10.4f}  ratio {gstd.mean()/u[c].std():.2f}')

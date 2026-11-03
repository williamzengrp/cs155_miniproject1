import sys; sys.path.insert(0, './work')
import numpy as np, pandas as pd
import harness as H
from time_features import AUDIO, parse_date
from sklearn.metrics import roc_auc_score
tr, te, y = H.load()
al = pd.concat([tr.assign(src='tr', y=y), te.assign(src='te')], ignore_index=True)
n = len(tr)
raw = al['track_album_release_date'].astype(str)
akey = al[AUDIO].round(6).astype(str).agg('|'.join, axis=1)
al['akey'] = akey
al['ucnt'] = al['uri'].map(al['uri'].value_counts())
# twins: same audio, different uri
nu = al.groupby('akey')['uri'].transform('nunique')
al['n_twin_uri'] = nu - 1
print('rows with an audio twin of a different uri:', (al.n_twin_uri > 0).sum(), ' train:', (al.n_twin_uri[:n] > 0).sum(), ' test:', (al.n_twin_uri[n:] > 0).sum())
t = al[al.n_twin_uri > 0].sort_values('akey')
print(t[['akey', 'uri', 'src', 'track_album_release_date', 'y']].head(20).to_string())
# same audio+date key vs uri
adkey = akey + '|' + raw
al['n_ad'] = adkey.map(adkey.value_counts())
print('audio+date count vs uri count agreement:', (al.n_ad == al.ucnt).mean())
print(pd.crosstab(al.n_ad, al.ucnt))
# label rate by ucnt in train
print(al[:n].groupby('ucnt')['y'].agg(['mean', 'size']))
print('test ucnt:', al[n:].ucnt.value_counts().to_dict())
# date group: other songs (diff uri) on the same date that are duplicated
uniq = al.drop_duplicates('uri')
g = uniq.groupby(uniq['track_album_release_date'].astype(str))
dup_songs_on_date = g['ucnt'].apply(lambda s: (s > 1).sum())
songs_on_date = g['uri'].size()
al['date_nsongs'] = raw.map(songs_on_date)
al['date_ndupsongs'] = raw.map(dup_songs_on_date)
al['other_dupsongs'] = al['date_ndupsongs'] - (al['ucnt'] > 1).astype(int)
al['other_songs'] = al['date_nsongs'] - 1
trn = al[:n]
# eval only on singletons (ucnt==1) since those are what matter in test
s1 = trn[trn.ucnt == 1]
print('\nsingletons in train:', len(s1), 'High rate', s1.y.mean().round(3))
print(s1.groupby(np.minimum(s1.other_dupsongs, 3))['y'].agg(['mean', 'size']))
print('among singletons with >=1 other song on date:')
m = s1.other_songs > 0
print(s1[m].groupby(np.minimum(s1[m].other_dupsongs, 3))['y'].agg(['mean', 'size']))
print('frac of other songs that are dup:', roc_auc_score(s1[m].y, s1[m].other_dupsongs / s1[m].other_songs))
# year-only dates
p = raw.str.len()
print('\nprecision label rate:', trn.groupby(p[:n])['y'].agg(['mean', 'size']).to_dict())
print('test precision counts', p[n:].value_counts().to_dict())
d = parse_date(raw)
print('year-only years:', sorted(d[p == 4].dt.year.unique())[:50])

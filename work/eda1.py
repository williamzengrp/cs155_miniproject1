import pandas as pd, numpy as np
D='./data/'
tr=pd.read_csv(D+'train.csv'); te=pd.read_csv(D+'test.csv')
print(tr.shape, te.shape)
print('empty rows train:', tr.isna().all(axis=1).sum())
print(tr.isna().sum()[tr.isna().sum()>0])
print(te.isna().sum()[te.isna().sum()>0])
tr=tr.dropna(subset=['Popularity_Type']).reset_index(drop=True)
print(tr['Popularity_Type'].value_counts())
vc=tr['uri'].value_counts()
print('train unique uris', tr['uri'].nunique(), 'rows', len(tr), 'extra copies', len(tr)-tr['uri'].nunique())
print('copy count distribution:', vc.value_counts().sort_index().to_dict())
print('test unique uris', te['uri'].nunique(), 'rows', len(te))
print('test uris in train:', te['uri'].isin(tr['uri']).sum())
# label by dup count
tr['cnt']=tr['uri'].map(vc)
tr['y']=(tr['Popularity_Type']=='High').astype(int)
print(tr.groupby('cnt')['y'].agg(['mean','size']))
# conflicting labels
g=tr.groupby('uri')['y'].agg(['min','max','size'])
print('songs with conflicting labels', ((g['min']!=g['max'])).sum())
# are duplicate rows identical in audio features?
AUDIO=['time_signature','speechiness','danceability','duration_ms','energy','mode','instrumentalness','valence','key','tempo','loudness','acousticness','liveness']
d=tr[tr.cnt>1].groupby('uri')[AUDIO+['track_album_release_date']].nunique()
print('dup groups with differing features:', (d>1).sum().to_dict())
# test ID structure
ids=te['ID'].values
print('test ID min/max', ids.min(), ids.max(), 'sorted?', (np.diff(ids)>0).all(), 'n', len(ids))

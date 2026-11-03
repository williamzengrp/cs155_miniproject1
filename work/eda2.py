import pandas as pd, numpy as np
D='./data/'
tr=pd.read_csv(D+'train.csv'); te=pd.read_csv(D+'test.csv')
print(tr[tr['danceability'].isna()].T)
ids=te['ID'].values
missing=sorted(set(range(0,ids.max()+1))-set(ids))
print('n missing IDs in [0,max]:', len(missing), missing[:30])
AUDIO=['time_signature','speechiness','danceability','duration_ms','energy','mode','instrumentalness','valence','key','tempo','loudness','acousticness','liveness']
tr=tr.dropna(subset=['danceability']).reset_index(drop=True)
tr['y']=(tr['Popularity_Type']=='High').astype(int)
al=pd.concat([tr.assign(src='tr'),te.assign(src='te')],ignore_index=True)
key=al[AUDIO].round(6).astype(str).agg('|'.join,axis=1)
al['akey']=key
# twins: same audio, different uri
g=al.groupby('akey')['uri'].nunique()
twin_keys=g[g>1].index
print('audio keys shared by >1 distinct uri:', len(twin_keys), 'rows involved:', al['akey'].isin(twin_keys).sum())
tw=al[al['akey'].isin(twin_keys)].sort_values('akey')
print(tw[['akey','uri','src','track_album_release_date','y']].head(30).to_string())
# test rows with an audio twin in train
trkeys=set(al.loc[al.src=='tr','akey'])
te_tw=al[(al.src=='te')&al['akey'].isin(trkeys)]
print('test rows with audio twin in train:', len(te_tw))
# within-test dup labels unknown; within-train twin label agreement
t2=al[(al.src=='tr')&al['akey'].isin(twin_keys)]
agg=t2.groupby('akey').agg(nuri=('uri','nunique'),ymean=('y','mean'),n=('y','size'))
print(agg[agg.nuri>1].describe())
# same-date group sizes
print('date string formats:', al['track_album_release_date'].astype(str).str.len().value_counts().to_dict())
vc=al['track_album_release_date'].value_counts()
print('top dates:', vc.head(10).to_dict())

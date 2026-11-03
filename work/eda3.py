import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
D='./data/'
tr=pd.read_csv(D+'train.csv').dropna(subset=['danceability']).reset_index(drop=True); te=pd.read_csv(D+'test.csv')
tr['y']=(tr['Popularity_Type']=='High').astype(int)
dates=tr['track_album_release_date']
# LOO date-mean excluding same uri
s=[];idx=[]
grp=tr.groupby('track_album_release_date')
for d,g in grp:
    if g['uri'].nunique()<2: continue
    for i,r in g.iterrows():
        m=g[g.uri!=r.uri]
        s.append(m['y'].mean()); idx.append(i)
s=np.array(s); idx=np.array(idx)
print('rows with >=1 date-mate (diff uri) in train:', len(idx), 'of', len(tr))
print('AUC of LOO date-mate mean on those rows:', roc_auc_score(tr.y.values[idx], s))
# test rows with date-mates in train
te_has=te['track_album_release_date'].isin(set(dates))
print('test rows with a train date-mate:', te_has.sum(), 'of', len(te))
# by group size bucket
cnt=tr.groupby('track_album_release_date')['uri'].transform('nunique')
for lo,hi in [(2,2),(3,5),(6,10),(11,100)]:
    m=(cnt.values[idx]>=lo)&(cnt.values[idx]<=hi)
    if m.sum()>20: print(f'  date group size {lo}-{hi}: n={m.sum()} AUC={roc_auc_score(tr.y.values[idx][m], s[m]):.3f} pos={tr.y.values[idx][m].mean():.2f}')
# year effect
d=pd.to_datetime(dates,errors='coerce',format='mixed')
tr['year']=d.dt.year
print(tr.groupby(pd.cut(tr.year,[0,1979,1989,1999,2009,2014,2019,2021,2022,2023,2024,2030]))['y'].agg(['mean','size']))
print('dow:', tr.groupby(d.dt.dayofweek)['y'].agg(['mean','size']).T)
# Year-only (yyyy) dates
print('year-only dates label mean:', tr[dates.str.len()==4]['y'].agg(['mean','size']).to_dict())
# Single feature AUCs
AUDIO=['time_signature','speechiness','danceability','duration_ms','energy','mode','instrumentalness','valence','key','tempo','loudness','acousticness','liveness']
for c in AUDIO+['year']:
    v=tr[c].fillna(tr[c].median())
    a=roc_auc_score(tr.y,v); print(f'{c:18s} {max(a,1-a):.3f} {"+" if a>0.5 else "-"}')

import sys, numpy as np, pandas as pd
sys.path.insert(0, './work')
import harness as H
tr, te, y = H.load()
g = tr.groupby('uri')['Popularity_Type'].nunique()
print('uris with inconsistent labels:', (g > 1).sum(), 'of', (tr.groupby('uri').size() > 1).sum(), 'dup uris')
al = pd.concat([tr, te], ignore_index=True)
raw = al['track_album_release_date'].astype(str)
uri = pd.factorize(al['uri'])[0]
# unique songs per date
first = pd.Series(np.arange(len(al))).groupby(uri).first().to_numpy()
ds = raw.iloc[first].value_counts()
print('dates:', len(ds), 'size dist (unique songs):', ds.value_counts().sort_index().head(15).to_dict())
print('songs in dates of size>=2:', ds[ds >= 2].sum(), 'of', ds.sum())
# label homogeneity within date (train unique songs)
trf = tr.drop_duplicates('uri')
yy = (trf['Popularity_Type'] == 'High').astype(int)
gd = trf.groupby('track_album_release_date')
m = gd['Popularity_Type'].transform('size') >= 2
s = gd['uri'].transform('size')
# leave-one-out date mean for train singles
S = yy.groupby(trf['track_album_release_date']).transform('sum'); C = s
loo = (S - yy) / (C - 1)
from sklearn.metrics import roc_auc_score
mm = C >= 2
print('LOO date-mean AUC (uri-level):', roc_auc_score(yy[mm], loo[mm]), 'n', mm.sum())
# fraction of pairs same label within date

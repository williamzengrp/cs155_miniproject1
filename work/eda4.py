import pandas as pd, numpy as np
D='./'
te=pd.read_csv(D+'data/test.csv')
p6=pd.read_csv(D+'cs155_miniproject1/predictions/prediction_6.csv')
assert (p6.ID.values==te.ID.values).all()
r=p6['Popularity_Type'].rank(pct=True)
dup=te['uri'].map(te['uri'].value_counts())>1
print('test dup rows', dup.sum(), 'pct-rank of their preds:', np.round(np.sort(r[dup].values),2))
print('mean pct-rank of dup rows', r[dup].mean().round(3))

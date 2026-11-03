"""EDA for test-construction artifacts (label-free features; labels only used here to measure signal on train)."""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
W = './work'
sys.path.insert(0, W)
import harness as H
from time_features import parse_date
tr, te, y = H.load()
n, m = len(tr), len(te)
al = pd.concat([tr, te], ignore_index=True)
raw = al['track_album_release_date'].astype(str)
d = parse_date(raw)
yr = d.dt.year.to_numpy()
is_te = np.r_[np.zeros(n, bool), np.ones(m, bool)]
uri = pd.factorize(al['uri'])[0]
ucnt = np.bincount(uri)[uri]          # pool copies of own song
# global non-randomness check: test share of distinct songs by year bucket
first = pd.Series(np.arange(n + m)).groupby(uri).first().to_numpy()
df = pd.DataFrame({'yr': yr[first], 'te': is_te[first], 'multi': ucnt[first] >= 2})
df['yb'] = pd.cut(df.yr, [0, 1989, 1999, 2009, 2014, 2017, 2019, 2021, 2022, 2023, 2024, 2030])
print(df.groupby('yb', observed=True).agg(n=('te', 'size'), test_share=('te', 'mean'), multi=('multi', 'mean')))
# High rate by year bucket for train singletons
sing = (ucnt[:n] == 1)
print('train singletons', sing.sum(), 'High rate', y[sing].mean())

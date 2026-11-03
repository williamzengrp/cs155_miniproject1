# EDA only: does within-date clustering give label-purer "albums" than the raw date? (labels used only to
# judge the clustering threshold, not as features)
import sys; sys.path.insert(0, './work/fe')
sys.path.insert(0, './work')
import numpy as np, pandas as pd
import harness as H
from fe_lib import *
from sklearn.metrics import roc_auc_score
tr, te, y = H.load(); n = len(tr)
al, raw, d, akey = make_pool(tr, te)
yy = np.r_[y, np.full(len(te), np.nan)]
ucnt = al['uri'].map(al['uri'].value_counts()).to_numpy()
def loo_auc(gid):
    # LOO mean label over other train SONGS (dedup by uri) in the same group, for train singletons
    df = pd.DataFrame({'g': gid, 'y': yy, 'uri': al['uri']}).iloc[:n].drop_duplicates('uri')
    s = df.groupby('g')['y'].transform('sum'); c = df.groupby('g')['y'].transform('count')
    loo = (s - df['y']) / (c - 1)
    m = (c > 1).to_numpy() & (ucnt[df.index] == 1)
    return roc_auc_score(df['y'][m], loo[m]), m.sum()
print('date groups:', loo_auc(pd.factorize(raw)[0]))
for cols_name, cols in [('ALB9', None), ('6', ALB6)]:
    for t in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        alb_row, alb, u, ru = album_assign(al, raw, akey, t, cols)
        nal = pd.Series(alb).nunique()
        a, m = loo_auc(alb_row)
        print(f'{cols_name} t={t}: n_albums={nal}  LOO-AUC={a:.4f} on {m} rows')
print('--- same-row comparison')
def loo(gid):
    df = pd.DataFrame({'g': gid, 'y': yy, 'uri': al['uri']}).iloc[:n].drop_duplicates('uri')
    s = df.groupby('g')['y'].transform('sum'); c = df.groupby('g')['y'].transform('count')
    return df, (s - df['y']) / (c - 1), c
dfd, lood, cd = loo(pd.factorize(raw)[0])
for t in [2.0, 2.5, 3.0]:
    alb_row, alb, u, ru = album_assign(al, raw, akey, t)
    dfa, looa, ca = loo(alb_row)
    m = ((ca > 1) & (cd > 1)).to_numpy() & (ucnt[dfa.index] == 1)
    # rows with album-mates AND extra date-mates outside the album (where the split matters)
    m2 = m & (cd > ca).to_numpy()
    print(f't={t}: rows {m.sum()} date-LOO {roc_auc_score(dfa.y[m], lood[m]):.4f} album-LOO {roc_auc_score(dfa.y[m], looa[m]):.4f} | '
          f'rows w/ other albums on date {m2.sum()}: date {roc_auc_score(dfa.y[m2], lood[m2]):.4f} album {roc_auc_score(dfa.y[m2], looa[m2]):.4f}')

"""r2_eda: second pass -- candidate label-free signals vs the final8 blend residual, and the worst misranks."""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
WORK = './work'
sys.path.insert(0, WORK)
import harness as H

pd.set_option('display.width', 250)
tr, te, y = H.load()
n = len(tr)
D = pd.read_pickle(WORK + '/r2_eda/blend_std.pkl')
al = pd.concat([tr, te], ignore_index=True)
raw = al['track_album_release_date'].astype(str)
uri = al['uri']
ucnt = uri.map(uri.value_counts())
rows_on_date = raw.map(raw.value_counts())
first = ~uri.duplicated()
songs_on_date = raw.map(raw[first].value_counts())
multi_on_date = raw.map(raw[first & (ucnt >= 2)].value_counts()).fillna(0)
# date-mates' extra copies (exclude own song)
mate_extra = (rows_on_date - ucnt) - (songs_on_date - 1)
mate_multi = multi_on_date - (ucnt >= 2).astype(int)
D['mate_extra'] = mate_extra[:n].values
D['mate_multi'] = mate_multi[:n].values
D['songs_on_date'] = songs_on_date[:n].values
D['mate_multi_frac'] = D.mate_multi / np.maximum(D.songs_on_date - 1, 1)
# test-side: how many test rows have mates with multiple copies
tm = pd.DataFrame({'mate_multi': mate_multi[n:].values, 'songs': songs_on_date[n:].values})
print('test rows with mate_multi>0:', (tm.mate_multi > 0).sum(), 'of', len(tm))
# LOO date TE (diagnostic only, label-based, own uri excluded)
yy = pd.Series(np.r_[y, np.full(len(te), np.nan)])
lab = yy.notna()
grp_sum = yy.groupby(raw).sum()
grp_cnt = lab.groupby(raw).sum()
own_sum = yy.groupby(uri).sum()
own_cnt = lab.groupby(uri).sum()
s_ = raw.map(grp_sum) - uri.map(own_sum)
c_ = raw.map(grp_cnt) - uri.map(own_cnt)
D['loo_cnt'] = c_[:n].values
D['loo_te'] = (s_ / c_.replace(0, np.nan))[:n].values

E = D[D.n_elig > 0].copy()


def seg(name, key):
    rows = []
    for k, G in E.groupby(key, observed=True):
        if len(G) < 8:
            continue
        z = (G.y - G.p).sum() / np.sqrt(max((G.p * (1 - G.p)).sum(), 1e-9))
        auc = roc_auc_score(G.y, G.s) if 0 < G.y.mean() < 1 else np.nan
        rows.append((k, len(G), G.y.mean(), G.p.mean(), z, auc))
    print(f'\n== {name}')
    print(pd.DataFrame(rows, columns=[name, 'n', 'rate', 'pmean', 'z', 'auc_in']).round(3).to_string(index=False))


seg('mate_multi', E.mate_multi.clip(upper=4))
seg('mate_extra', E.mate_extra.clip(upper=5))
seg('mate_multi_frac', pd.cut(E.mate_multi_frac, [-0.01, 0, 0.1, 0.25, 0.5, 1.01]))
seg('loo_cnt', E.loo_cnt.clip(upper=12))
seg('loo_te', pd.cut(E.loo_te, [-0.01, 0, 0.2, 0.4, 0.6, 0.8, 0.99, 1.01]))
# among rows with mates: residual vs mate_multi at fixed loo_te
E2 = E[E.loo_cnt > 0]
seg('loo_te x mate_multi>0 (rows with labelled mates)', pd.cut(E2.loo_te, [-0.01, 0.2, 0.5, 0.8, 1.01]).astype(str) + ' mm>0=' + (E2.mate_multi > 0).astype(str))
# worst misranks
pos, neg = E[E.y == 1], E[E.y == 0]
print('\nlowest-scored HIGH songs:')
print(pos.nsmallest(25, 's')[['s', 'raw', 'uri_cnt', 'songs_on_date', 'loo_cnt', 'loo_te', 'duration_ms', 'instrumentalness', 'speechiness', 'loudness', 'acousticness', 'energy', 'tempo', 'liveness']].round(3).to_string())
print('\nhighest-scored LOW songs:')
print(neg.nlargest(25, 's')[['s', 'raw', 'uri_cnt', 'songs_on_date', 'loo_cnt', 'loo_te', 'duration_ms', 'instrumentalness', 'speechiness', 'loudness', 'acousticness', 'energy', 'tempo', 'liveness']].round(3).to_string())

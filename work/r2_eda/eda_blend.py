"""r2_eda: error analysis of the final8 OOF rank blend on the std splits (label-free segments).
Builds per-repeat equal-weight rank average of the 8 members over eligible rows, calibrates it per repeat with
isotonic regression (diagnostic only), and reports per-segment: n, High rate, mean calibrated p, residual z-score,
within-segment AUC. Saves the per-row blend table to r2_eda/blend_std.pkl.
"""
import sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
WORK = './work'
sys.path.insert(0, WORK)
import harness as H
from time_features import parse_date

MEM = ['fkd_lgbm', 'fkt_xgb', 'fkt_cat', 'fkd_et', 'fkd_rf', 'mdl_mlp_ple', 'ch_et_proj', 'ch_lgbm_proj']
tr, te, y = H.load()
n = len(tr)
res = [H.load_res(m) for m in MEM]
R = res[0]['oof'].shape[0]
S = np.full((R, n), np.nan)    # blend rank score
P = np.full((R, n), np.nan)    # isotonic calibrated
aucs = []
for r in range(R):
    m = ~np.isnan(res[0]['oof'][r])
    O = np.column_stack([rankdata(x['oof'][r, m]) / m.sum() for x in res])
    s = O.mean(1)
    S[r, m] = s
    P[r, m] = IsotonicRegression(out_of_bounds='clip').fit(s, y[m]).predict(s)
    aucs.append(roc_auc_score(y[m], s))
print('final8 per-rep', np.round(aucs, 4), 'mean', np.mean(aucs))

al = pd.concat([tr, te], ignore_index=True)
raw = al['track_album_release_date'].astype(str)
d = parse_date(raw)
D = pd.DataFrame({'y': y})
D['s'] = np.nanmean(S, 0)
D['p'] = np.nanmean(P, 0)
D['n_elig'] = (~np.isnan(S)).sum(0)
D['raw'] = raw[:n].values
D['year'] = d.dt.year[:n].values
D['month'] = d.dt.month[:n].values
D['dow'] = d.dt.dayofweek[:n].values
D['day'] = d.dt.day[:n].values
D['prec'] = raw[:n].str.len().values
D['jan1'] = ((D.month == 1) & (D.day == 1)).astype(int)
cnt_all = raw.map(raw.value_counts())
uri_cnt = al['uri'].map(al['uri'].value_counts())
D['uri_cnt'] = uri_cnt[:n].values
# distinct-uri date group size in pool
first = ~al['uri'].duplicated()
nd_u = raw[first].value_counts()
D['date_n_songs'] = raw[:n].map(nd_u).values
te_raw = te['track_album_release_date'].astype(str)
D['date_n_test'] = raw[:n].map(te_raw.value_counts()).fillna(0).values
for c in ['time_signature', 'mode', 'key', 'duration_ms', 'speechiness', 'liveness', 'instrumentalness', 'tempo',
          'loudness', 'acousticness', 'energy', 'danceability', 'valence']:
    D[c] = tr[c].values
D.to_pickle(WORK + '/r2_eda/blend_std.pkl')
E = D[D.n_elig > 0].copy()
print('rows ever eligible', len(E), 'of', n, ' High rate', E.y.mean().round(3))


def seg(name, key):
    g = E.groupby(key, observed=True)
    rows = []
    for k, G in g:
        if len(G) < 8:
            continue
        resid = (G.y - G.p).sum()
        var = (G.p * (1 - G.p)).sum()
        z = resid / np.sqrt(max(var, 1e-9))
        auc = roc_auc_score(G.y, G.s) if 0 < G.y.mean() < 1 else np.nan
        rows.append((k, len(G), G.y.mean(), G.p.mean(), z, auc))
    T = pd.DataFrame(rows, columns=[name, 'n', 'rate', 'pmean', 'z', 'auc_in'])
    print(f'\n== {name}')
    print(T.round(3).to_string(index=False))


seg('year_bin', pd.cut(E.year, [0, 1980, 1990, 2000, 2005, 2010, 2013, 2015, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2030]))
seg('dow', E.dow)
seg('month', E.month)
seg('prec', E.prec)
seg('jan1', E.jan1)
seg('date_n_songs', pd.cut(E.date_n_songs, [0, 1, 2, 3, 5, 8, 15, 30, 1000]))
seg('date_n_test', pd.cut(E.date_n_test, [-1, 0, 1, 2, 4, 100]))
seg('uri_cnt', E.uri_cnt)
seg('time_signature', E.time_signature)
seg('mode', E['mode'])
seg('key', E.key)
seg('dur', pd.qcut(E.duration_ms, 10))
seg('speech', pd.cut(E.speechiness, [0, 0.04, 0.1, 0.2, 0.33, 0.66, 1]))
seg('live', pd.cut(E.liveness, [0, 0.1, 0.2, 0.4, 0.8, 1]))
seg('instr', pd.cut(E.instrumentalness, [-1, 0, 1e-4, 0.01, 0.1, 0.5, 0.9, 1]))
seg('tempo', pd.cut(E.tempo, [-1, 1, 70, 90, 110, 130, 150, 250]))
seg('loud', pd.qcut(E.loudness, 10))
seg('acous', pd.qcut(E.acousticness, 10))
seg('energy', pd.qcut(E.energy, 10))
seg('dance', pd.qcut(E.danceability, 10))
seg('valence', pd.qcut(E.valence, 10))
seg('pred_decile', pd.qcut(E.s, 10))

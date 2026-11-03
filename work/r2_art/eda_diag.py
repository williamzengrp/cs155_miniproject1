"""Why does the real-test-file neighbourhood share carry more univariate signal than the fold-simulated one?
Compare (a) real test file as 'test', (b) 5 random pseudo-test draws from the TRAIN file with uri-removal
(same process as the real construction), evaluated on train singletons; plus era-controlled AUC."""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
W = './work'
sys.path.insert(0, W); sys.path.insert(0, W + '/r2_art'); sys.path.insert(0, W + '/te')
import harness as H
from art_feats import art_features_sim
from time_features import parse_date
tr, te, y = H.load(); n, m = len(tr), len(te); N = n + m
al = pd.concat([tr, te], ignore_index=True)
uri = pd.factorize(al['uri'])[0]
c = np.bincount(uri)[uri]
sing = np.nonzero(c[:n] == 1)[0]
yr = parse_date(al.track_album_release_date.astype(str)).dt.year.to_numpy()
eb = np.digitize(yr, [1990, 2010, 2015, 2020, 2022, 2023, 2024])


def within_era_auc(v, rows):
    # AUC computed within era buckets, averaged weighted by pairs (removes era confounding)
    num = den = 0.0
    for b in np.unique(eb[rows]):
        r = rows[eb[rows] == b]
        yy = y[r]; vv = v[r]
        if yy.min() == yy.max(): continue
        p = yy.sum() * (len(yy) - yy.sum())
        num += roc_auc_score(yy, vv) * p; den += p
    return num / den

trm = np.r_[np.ones(n, bool), np.zeros(m, bool)]
F = art_features_sim(tr, te, trm, ~trm, blocks=('date', 'knn'))
print('REAL test file: ', {k: (round(roc_auc_score(y[sing], np.nan_to_num(F[k].to_numpy()[sing], nan=0.2)), 4),
                              round(within_era_auc(np.nan_to_num(F[k].to_numpy(), nan=0.2), sing), 4)) for k in ['art_date_share', 'art_knn50_tefrac', 'art_knn50_dupfrac']})
rng = np.random.default_rng(0)
for s in range(5):
    # pseudo split of the train file: 20% pseudo-test, remove pseudo-test rows whose uri is in pseudo-train
    pt = rng.random(n) < 0.2
    trm = np.zeros(N, bool); trm[:n] = ~pt
    tri = set(uri[:n][~pt])
    tem = np.zeros(N, bool); tem[:n] = pt & np.array([u not in tri for u in uri[:n]])
    G = art_features_sim(tr, te, trm, tem, blocks=('date', 'knn'))
    ev = np.intersect1d(sing, np.nonzero(trm[:n])[0])      # pseudo-train singletons
    print('pseudo draw', s, {k: (round(roc_auc_score(y[ev], np.nan_to_num(G[k].to_numpy()[ev], nan=0.2)), 4),
                                 round(within_era_auc(np.nan_to_num(G[k].to_numpy(), nan=0.2), ev), 4)) for k in ['art_date_share', 'art_knn50_tefrac', 'art_knn50_dupfrac']})

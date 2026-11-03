"""Label-free features from traces of how the test set was built (agent r2_art).

Test construction: random ~80/20 ROW split of the original data, then every test row whose uri also appears
in train was removed. Multi-playlist songs (mostly High) therefore almost never survive in test, and
"orphan" train singletons are songs whose other copy was removed from test.

Exchangeability (why these features are valid for both train and test rows, and in CV):
every statistic for row i is computed over OTHER rows only (uri != uri_i; same uri implies same date and
features). For a true singleton song i, the assignment of the other rows to train/test and the removal of
other test rows (which depends only on whether THEIR uri is in train) are independent of whether i itself
went to train or test. So the joint law of (feature_i, y_i) is the same for a train singleton and a test
row; the CV (eligible validation rows ~ singletons) therefore scores these features exactly as the test set
would. No labels are used anywhere in this file.
"""
import sys
import numpy as np
import pandas as pd
W = './work'
sys.path.insert(0, W); sys.path.insert(0, W + '/te')
from time_features import parse_date


def _others(key, uri, mask_rows, w=None):
    """For every pool row: sum over rows sharing `key`, restricted to mask_rows, minus own-uri rows."""
    w = np.ones(len(key)) if w is None else w
    v = mask_rows.astype(float) * w
    K = np.bincount(key, weights=v, minlength=key.max() + 1)
    U = np.bincount(uri, weights=v, minlength=uri.max() + 1)
    return K[key] - U[uri]


def art_features(tr, te, blocks=('date',), knn_k=(20, 50)):
    n, m = len(tr), len(te)
    al = pd.concat([tr, te], ignore_index=True)
    N = n + m
    raw = al['track_album_release_date'].astype(str)
    d = parse_date(raw)
    uri = pd.factorize(al['uri'])[0]
    is_te = np.r_[np.zeros(n, bool), np.ones(m, bool)]
    ucnt = np.bincount(uri)[uri]                       # copies of the row's song in the pool
    first = np.zeros(N, bool)
    first[pd.Series(np.arange(N)).groupby(uri).first().to_numpy()] = True   # one row per song
    multi = ucnt >= 2
    keys = {'date': pd.factorize(raw)[0], 'ym': pd.factorize(d.dt.to_period('M').astype(str))[0],
            'year': pd.factorize(d.dt.year)[0]}
    F = {}
    for b in blocks:
        if b in keys:
            k = keys[b]
            R = _others(k, uri, ~is_te)                          # train-file rows of other songs
            T = _others(k, uri, is_te)                           # test-file rows of other songs
            RS = _others(k, uri, ~is_te & first)                 # distinct other train songs
            TS = _others(k, uri, is_te & first)                  # distinct other test songs
            DS = _others(k, uri, multi & first)                  # distinct other multi-copy songs (pool)
            RS1 = _others(k, uri, ~is_te & first & ~multi)       # other train songs that look single
            p = f'art_{b}_'
            F[p + 'R'] = R
            F[p + 'T'] = T
            F[p + 'dupS'] = DS
            F[p + 'dupfrac'] = np.where(RS + TS > 0, DS / np.maximum(RS + TS, 1), np.nan)
            F[p + 'share'] = np.where(RS + TS > 0, TS / np.maximum(RS + TS, 1), np.nan)
            F[p + 'exc'] = RS1 - 4.0 * TS                        # orphan excess estimate (E ~ #orphans)
        elif b == 'knn':
            from label_features import get_ctx
            ctx = get_ctx(tr, te)
            ind, dist = ctx.nbr['yr']
            for kk in knn_k:
                I = ind[:, :kk]
                ok = np.isfinite(dist[:, :kk])
                F[f'art_knn{kk}_tefrac'] = (is_te[I] * ok).sum(1) / np.maximum(ok.sum(1), 1)
                F[f'art_knn{kk}_dupfrac'] = (multi[I] * ok).sum(1) / np.maximum(ok.sum(1), 1)
        elif b == 'selfx':
            # self-excluded versions of the pool density/look-alike features (own copies removed)
            pass
        else:
            raise ValueError(b)
    return pd.DataFrame(F)


def special_block(tr, te, b):
    """'selfx': own-copy-excluded versions of the pool density / look-alike features (test rows never have
    copies of their song in the pool, train duplicates do) -> returns (columns to drop, replacement frame).
    'dupcnt': copies of the row's own song in the pool (lets the model explain duplicates away)."""
    from sklearn.neighbors import NearestNeighbors
    n, m = len(tr), len(te)
    al = pd.concat([tr, te], ignore_index=True)
    N = n + m
    raw = al['track_album_release_date'].astype(str)
    uri = pd.factorize(al['uri'])[0]
    ucnt = np.bincount(uri)[uri].astype(float)
    if b == 'dupcnt':
        return pd.DataFrame({'own_copies': ucnt})
    if b != 'selfx':
        raise ValueError(b)
    d = parse_date(raw)
    ym = d.dt.to_period('M').astype(str)
    year = d.dt.year
    F = pd.DataFrame(index=al.index)
    F['n_same_date'] = raw.map(raw.value_counts()) - (ucnt - 1)
    F['n_same_month'] = ym.map(ym.value_counts()) - (ucnt - 1)
    F['n_same_year'] = year.map(year.value_counts()) - (ucnt - 1)
    F['n_same_date_audio'] = al.groupby([raw, al['tempo'].round(0)])['tempo'].transform('size') - (ucnt - 1)
    cols = ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'speechiness']
    Zs = ((al[cols] - al[cols].mean()) / al[cols].std()).fillna(0).to_numpy()
    out = {k: np.zeros(N) for k in ('a_n_close10', 'a_n_close20', 'a_n_close30')}
    out['a_nn2_dist'] = np.full(N, np.nan)
    for _, idx in al.groupby(raw).indices.items():
        if len(idx) < 2:
            continue
        D = np.linalg.norm(Zs[idx, None, :] - Zs[None, idx, :], axis=2)
        D[uri[idx][:, None] == uri[idx][None, :]] = np.inf          # own row and own copies
        for r in (1.0, 2.0, 3.0):
            out[f'a_n_close{int(r * 10)}'][idx] = (D < r).sum(axis=1)
        Ds = np.sort(D, axis=1)
        ok = np.isfinite(Ds[:, 1])
        out['a_nn2_dist'][idx[ok]] = Ds[ok, 1]
    nbr = NearestNeighbors(n_neighbors=12).fit(Zs)
    dist, ind = nbr.kneighbors(Zs)
    g1, g5 = np.full(N, np.nan), np.full(N, np.nan)
    for i in range(N):
        keep = dist[i][uri[ind[i]] != uri[i]]
        g1[i], g5[i] = keep[0], keep[4]
    out['a_global_nn_dist'] = g1
    out['a_global_nn5_dist'] = g5
    for k, v in out.items():
        F[k] = v
    return list(F.columns), F.reset_index(drop=True)


def art_features_sim(tr, te, trainmask, testmask, blocks=('date', 'knn'), knn_k=(20, 50)):
    """Fold-consistent version: only rows in trainmask | testmask form the pool (others are 'removed').
    CV fold:  trainmask = fit rows, testmask = eligible validation rows (val rows whose uri is not in fit),
              real test rows and ineligible val rows are left out -> the val fold plays the test file and the
              removal of val copies of fit songs mirrors the real removal of test copies of train songs.
    Final fit: trainmask = all train rows, testmask = all test rows.
    Features are returned for ALL pool rows (rows outside the pool get values computed against the pool;
    in CV those are ineligible val rows / real test rows whose predictions are not scored)."""
    n, m = len(tr), len(te)
    al = pd.concat([tr, te], ignore_index=True)
    N = n + m
    raw = al['track_album_release_date'].astype(str)
    d = parse_date(raw)
    uri = pd.factorize(al['uri'])[0]
    inpool = trainmask | testmask
    ucnt = np.bincount(uri, weights=inpool.astype(float), minlength=uri.max() + 1)[uri]
    firstpool = np.zeros(N, bool)
    pidx = np.nonzero(inpool)[0]
    firstpool[pd.Series(pidx).groupby(uri[pidx]).first().to_numpy()] = True
    multi = ucnt >= 2
    keys = {'date': pd.factorize(raw)[0], 'ym': pd.factorize(d.dt.to_period('M').astype(str))[0]}
    F = {}
    for b in blocks:
        if b in keys:
            k = keys[b]
            RS = _others(k, uri, trainmask & firstpool)
            TS = _others(k, uri, testmask & firstpool)
            DS = _others(k, uri, multi & firstpool)
            p = f'art_{b}_'
            F[p + 'dupS'] = DS
            F[p + 'dupfrac'] = np.where(RS + TS > 0, DS / np.maximum(RS + TS, 1), np.nan)
            F[p + 'share'] = np.where(RS + TS > 0, TS / np.maximum(RS + TS, 1), np.nan)
        elif b == 'knn':
            from label_features import get_ctx
            ctx = get_ctx(tr, te)
            ind, dist = ctx.nbr['yr']
            ok = np.isfinite(dist) & inpool[ind]
            rank = np.cumsum(ok, axis=1)
            for kk in knn_k:
                sel = ok & (rank <= kk)
                cnt = np.maximum(sel.sum(1), 1)
                F[f'art_knn{kk}_tefrac'] = (testmask[ind] & sel).sum(1) / cnt
                F[f'art_knn{kk}_dupfrac'] = (multi[ind] & sel).sum(1) / cnt
        else:
            raise ValueError(b)
    return pd.DataFrame(F)

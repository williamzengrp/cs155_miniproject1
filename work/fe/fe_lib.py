"""Candidate label-free feature groups (fe agent). Every function takes the train+test pool and returns a
DataFrame aligned with the pool rows. Nothing here looks at labels."""
import numpy as np
import pandas as pd
from time_features import AUDIO, parse_date

ALB6 = ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'speechiness']


def make_pool(tr, te):
    al = pd.concat([tr, te], ignore_index=True)
    raw = al['track_album_release_date'].astype(str)
    d = parse_date(raw)
    akey = al[AUDIO].round(6).astype(str).agg('|'.join, axis=1)
    return al, raw, d, akey


def dup_feats(al, raw, d, akey):
    """Explicit duplicate / twin signals on the train+test pool."""
    F = pd.DataFrame(index=al.index)
    F['dup_uri_cnt'] = al['uri'].map(al['uri'].value_counts()).astype(float)
    adk = akey + '|' + raw
    F['dup_ad_cnt'] = adk.map(adk.value_counts()).astype(float)
    # twins: identical audio, different uri (re-release / single+album versions)
    nu = al.groupby(akey)['uri'].transform('nunique')
    F['twin_n_uri'] = (nu - 1).astype(float)
    nd = al.groupby(akey)['track_album_release_date'].transform('nunique')
    F['twin_n_dates'] = (nd - 1).astype(float)
    return F


def dedup_index(al, akey, raw, by='uri'):
    """Row index of one representative per song; and map pool row -> representative position."""
    key = al['uri'] if by == 'uri' else (akey + '|' + raw)
    codes, uniq = pd.factorize(key)
    first = pd.Series(np.arange(len(al))).groupby(codes).first().to_numpy()
    return first, codes


def dedup_group_feats(al, raw, d, akey, by='uri'):
    """Teammates' density / albumnn / groupagg features recomputed on the de-duplicated pool (one row per
    song), so that they describe album-mates rather than copies of the same song."""
    first, codes = dedup_index(al, akey, raw, by)
    u = al.iloc[first].reset_index(drop=True)
    ru = raw.iloc[first].reset_index(drop=True)
    G = pd.DataFrame(index=u.index)
    G['u_n_same_date'] = ru.map(ru.value_counts()).astype(float)
    g = u.groupby(ru)
    for c in ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'speechiness',
              'instrumentalness', 'tempo', 'duration_ms']:
        mean = g[c].transform('mean')
        G[f'u_g_mean_{c}'] = mean
        G[f'u_g_dev_{c}'] = u[c] - mean
    G['u_g_std_energy'] = g['energy'].transform('std')
    G['u_g_std_loudness'] = g['loudness'].transform('std')
    Zs = ((u[ALB6] - u[ALB6].mean()) / u[ALB6].std()).fillna(0).to_numpy()
    out = {k: np.zeros(len(u)) for k in ('u_a_n_close', 'u_a_n_close10', 'u_a_n_close20', 'u_a_n_close30')}
    out['u_a_nn_dist'] = np.full(len(u), np.nan)
    out['u_a_nn2_dist'] = np.full(len(u), np.nan)
    for _, idx in u.groupby(ru).indices.items():
        if len(idx) < 2:
            continue
        D = np.linalg.norm(Zs[idx, None, :] - Zs[None, idx, :], axis=2)
        np.fill_diagonal(D, np.inf)
        out['u_a_n_close'][idx] = (D < 1.5).sum(axis=1)
        for r in (1.0, 2.0, 3.0):
            out[f'u_a_n_close{int(r * 10)}'][idx] = (D < r).sum(axis=1)
        out['u_a_nn_dist'][idx] = D.min(axis=1)
        if len(idx) >= 3:
            out['u_a_nn2_dist'][idx] = np.sort(D, axis=1)[:, 1]
    from sklearn.neighbors import NearestNeighbors
    dist, _ = NearestNeighbors(n_neighbors=6).fit(Zs).kneighbors(Zs)
    out['u_a_global_nn_dist'] = dist[:, 1]
    out['u_a_global_nn5_dist'] = dist[:, 5]
    for k, v in out.items():
        G[k] = v
    return G.iloc[codes].reset_index(drop=True)


def cluster_feats(al, k=8, seed=0, cols=None, prefix='km'):
    """Genre-like proxies: KMeans on standardized audio (fit on the de-duplicated pool)."""
    from sklearn.cluster import KMeans
    cols = cols or ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'speechiness',
                    'instrumentalness', 'tempo', 'duration_ms', 'liveness']
    X = al[cols].copy()
    X['duration_ms'] = np.log(X['duration_ms'])
    X['instrumentalness'] = np.log10(X['instrumentalness'] + 1e-4)
    X['speechiness'] = np.log(X['speechiness'])
    mu, sd = X.mean(), X.std()
    Z = ((X - mu) / sd).to_numpy()
    _, first = np.unique(al['uri'].to_numpy(), return_index=True)
    km = KMeans(k, n_init=10, random_state=seed).fit(Z[first])
    D = km.transform(Z)
    F = pd.DataFrame(D, columns=[f'{prefix}{k}_d{j}' for j in range(k)])
    F[f'{prefix}{k}_lab'] = D.argmin(1)
    F[f'{prefix}{k}_dmin'] = D.min(1)
    return F


def _zaudio(al, cols=None):
    cols = cols or ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'speechiness',
                    'instrumentalness', 'tempo', 'duration_ms', 'liveness']
    X = al[cols].copy().astype(float)
    if 'duration_ms' in X: X['duration_ms'] = np.log(X['duration_ms'])
    if 'instrumentalness' in X: X['instrumentalness'] = np.log10(X['instrumentalness'] + 1e-4)
    if 'speechiness' in X: X['speechiness'] = np.log(X['speechiness'])
    if 'liveness' in X: X['liveness'] = np.log(X['liveness'])
    return ((X - X.mean()) / X.std()).to_numpy()


def pca_feats(al, k=5):
    from sklearn.decomposition import PCA
    Z = _zaudio(al)
    _, first = np.unique(al['uri'].to_numpy(), return_index=True)
    p = PCA(k, random_state=0).fit(Z[first])
    return pd.DataFrame(p.transform(Z), columns=[f'pca{j}' for j in range(k)])


def gmm_feats(al, k=8, seed=0):
    from sklearn.mixture import GaussianMixture
    Z = _zaudio(al)
    _, first = np.unique(al['uri'].to_numpy(), return_index=True)
    g = GaussianMixture(k, covariance_type='full', n_init=3, random_state=seed).fit(Z[first])
    P = g.predict_proba(Z)
    F = pd.DataFrame(P, columns=[f'gmm{k}_p{j}' for j in range(k)])
    F[f'gmm{k}_ll'] = g.score_samples(Z)            # typicality (log density)
    return F


def resid_feats(al, d):
    """Mastering / style residuals: loudness given energy (+year), acousticness given energy."""
    F = pd.DataFrame(index=al.index)
    e = al['energy'].to_numpy()
    yr = d.dt.year.to_numpy().astype(float)
    def poly_resid(target, cols):
        A = np.column_stack([np.ones(len(al))] + cols)
        coef, *_ = np.linalg.lstsq(A, target, rcond=None)
        return target - A @ coef
    F['r_loud_energy'] = poly_resid(al['loudness'].to_numpy(), [e, e ** 2, e ** 3])
    F['r_loud_energy_year'] = poly_resid(al['loudness'].to_numpy(), [e, e ** 2, e ** 3, yr, yr ** 2])
    F['r_acou_energy'] = poly_resid(al['acousticness'].to_numpy(), [e, e ** 2, e ** 3])
    F['r_dance_valence'] = al['danceability'] * al['valence']
    F['r_energy_x_loud'] = al['energy'] * al['loudness']
    return F


def release_day_feats(d, raw):
    F = pd.DataFrame(index=d.index)
    dow = d.dt.day_of_week
    full = raw.str.len() == 10
    grd = pd.Timestamp('2015-07-10')
    F['rd_std_day'] = (full & (((d >= grd) & (dow == 4)) | ((d < grd) & dow.isin([0, 1])))).astype(int)
    F['rd_fri_post'] = (full & (d >= grd) & (dow == 4)).astype(int)
    F['rd_year_only'] = (raw.str.len() == 4).astype(int)
    return F


ALBCOLS = ['loudness', 'duration_ms', 'instrumentalness', 'speechiness', 'acousticness', 'energy',
           'danceability', 'liveness', 'valence']


def album_assign(al, raw, akey, t=2.0, cols=None, method='average'):
    """Split each release-date group (de-duplicated pool) into likely albums by agglomerative clustering
    on standardized audio. Returns (album id per pool row, album id per unique song, unique-song frame)."""
    from scipy.cluster.hierarchy import linkage, fcluster
    first, codes = dedup_index(al, akey, raw, 'uri')
    u = al.iloc[first].reset_index(drop=True)
    ru = raw.iloc[first].reset_index(drop=True)
    Z = _zaudio(u, cols or ALBCOLS)
    alb = np.zeros(len(u), int)
    gid = 0
    for _, idx in u.groupby(ru).indices.items():
        if len(idx) == 1:
            alb[idx] = gid; gid += 1; continue
        lab = fcluster(linkage(Z[idx], method), t, 'distance')
        alb[idx] = gid + lab - 1
        gid += lab.max()
    return alb[codes], alb, u, ru


def album_feats(al, raw, akey, t=2.0, cols=None, method='average', means=True):
    alb_row, alb, u, ru = album_assign(al, raw, akey, t, cols, method)
    s = pd.Series(alb)
    F = pd.DataFrame(index=u.index)
    F['alb_size'] = s.map(s.value_counts()).astype(float)
    F['alb_n_on_date'] = pd.Series(alb).groupby(ru).transform('nunique').astype(float)
    F['alb_share'] = F['alb_size'] / ru.map(ru.value_counts())
    if means:
        for c in ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'speechiness',
                  'instrumentalness', 'duration_ms']:
            m = u[c].groupby(alb).transform('mean')
            F[f'alb_mean_{c}'] = m
    first, codes = dedup_index(al, akey, raw, 'uri')
    out = F.iloc[codes].reset_index(drop=True)
    return out, alb_row


def typicality_feats(al, raw, d, akey, ks=(5, 20), era=2):
    """How 'mainstream' a song sounds: mean distance to its k nearest other songs (de-duplicated pool,
    10 standardized audio dims), globally and among songs released within +-era years."""
    from sklearn.neighbors import NearestNeighbors
    first, codes = dedup_index(al, akey, raw, 'uri')
    u = al.iloc[first].reset_index(drop=True)
    yr = d.iloc[first].dt.year.to_numpy()
    Z = _zaudio(u)
    F = pd.DataFrame(index=u.index)
    kmax = max(ks)
    dist, _ = NearestNeighbors(n_neighbors=kmax + 1).fit(Z).kneighbors(Z)
    for k in ks:
        F[f'ty_glob_k{k}'] = dist[:, 1:k + 1].mean(1)
    out = {k: np.full(len(u), np.nan) for k in ks}
    for yv in np.unique(yr):
        q = np.where(yr == yv)[0]
        c = np.where(np.abs(yr - yv) <= era)[0]
        kk = min(kmax + 1, len(c))
        dd, _ = NearestNeighbors(n_neighbors=kk).fit(Z[c]).kneighbors(Z[q])
        for k in ks:
            out[k][q] = dd[:, 1:k + 1].mean(1) if kk > 1 else np.nan
    for k in ks:
        F[f'ty_era_k{k}'] = out[k]
        F[f'ty_ratio_k{k}'] = F[f'ty_era_k{k}'] / F[f'ty_glob_k{k}']
    return F.iloc[codes].reset_index(drop=True)


def density_ratio_feats(al, raw, d, akey):
    """Release-date group size relative to its month / year (song level, de-duplicated)."""
    first, codes = dedup_index(al, akey, raw, 'uri')
    ru = raw.iloc[first].reset_index(drop=True)
    du = d.iloc[first].reset_index(drop=True)
    ym = du.dt.to_period('M').astype(str)
    yr = du.dt.year
    F = pd.DataFrame(index=ru.index)
    nd = ru.map(ru.value_counts()).astype(float)
    F['dr_date_over_month'] = nd / ym.map(ym.value_counts())
    F['dr_date_over_year'] = nd / yr.map(yr.value_counts())
    # same-weekday dates of the same year: how big is this date compared with a typical one
    dow = du.dt.day_of_week
    key = yr.astype(str) + '_' + dow.astype(str)
    per_date = pd.DataFrame({'k': key, 'r': ru}).drop_duplicates('r')
    n_dates = per_date.groupby('k').size()
    F['dr_date_over_typical'] = nd / (key.map(key.value_counts()) / key.map(n_dates))
    return F.iloc[codes].reset_index(drop=True)


ERA_COLS = ['danceability', 'duration_ms', 'energy', 'valence', 'tempo', 'loudness', 'acousticness',
            'speechiness', 'instrumentalness', 'liveness']


def era_relative_feats(al, raw, d, akey, w=2.0, cols=None, kinds=('rank', 'z')):
    """Audio features relative to songs released within +-w years (sliding window on the continuous date,
    de-duplicated pool): percentile rank and z-score."""
    cols = cols or ERA_COLS
    first, codes = dedup_index(al, akey, raw, 'uri')
    u = al.iloc[first].reset_index(drop=True)
    du = d.iloc[first].reset_index(drop=True)
    t = (du.dt.year + (du.dt.dayofyear - 1) / 365.25).to_numpy()
    M = np.abs(t[:, None] - t[None, :]) <= w                      # reference windows (n x n bool)
    cnt = M.sum(1)
    F = pd.DataFrame(index=u.index)
    for c in cols:
        v = u[c].to_numpy().astype(float)
        if 'rank' in kinds:
            less = (M & (v[None, :] < v[:, None])).sum(1)
            eq = (M & (v[None, :] == v[:, None])).sum(1)
            F[f'er_rank_{c}'] = (less + 0.5 * eq) / cnt
        if 'z' in kinds:
            s1 = M @ v
            s2 = M @ (v * v)
            mu = s1 / cnt
            sd = np.sqrt(np.maximum(s2 / cnt - mu ** 2, 1e-12))
            F[f'er_z_{c}'] = (v - mu) / sd
    F['er_window_n'] = cnt.astype(float)
    return F.iloc[codes].reset_index(drop=True)

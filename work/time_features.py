"""Time-series style features built from track_album_release_date (shared by the model scripts).

Every feature is label-free and computed on train+test together, so counts / era statistics see the
full release calendar. Groups:
  trend        continuous time, age, decade
  seasonality  month / day-of-week / day-of-year / week, plus sin-cos encodings and release-day flags
  precision    whether the date was given as a full date, year-month, or year only
  density      how many songs share the release date / month / year (album size, era volume)
  detrended    audio features relative to their era (rolling median over nearby release years)
  groupagg     release-date group (~album) mean of each audio feature and the song's deviation
  albumnn      same-date songs that sound alike: count within a radius, nearest distance
  albumnn2     more radii, 2nd-neighbour distance, nearest look-alikes across all dates
"""
import numpy as np
import pandas as pd

AUDIO = ['time_signature', 'speechiness', 'danceability', 'duration_ms', 'energy', 'mode',
         'instrumentalness', 'valence', 'key', 'tempo', 'loudness', 'acousticness', 'liveness']
DETREND = ['danceability', 'duration_ms', 'energy', 'valence', 'tempo', 'loudness',
           'acousticness', 'speechiness', 'instrumentalness', 'liveness']
REF_DATE = pd.Timestamp('2025-01-01')
# feature groups used by the model scripts (chosen with feature_ablation.py)
FEATURE_GROUPS = ('trend', 'seasonality', 'density', 'albumnn', 'albumnn2', 'groupagg')


def parse_date(s):
    d = pd.to_datetime(s, format='%Y-%m-%d', errors='coerce')
    d = d.fillna(pd.to_datetime(s, format='%Y-%m', errors='coerce'))
    d = d.fillna(pd.to_datetime(s, format='%Y', errors='coerce'))
    return d.fillna(pd.to_datetime(s, format='mixed', errors='coerce'))


def build_features(tr, te, groups=('trend', 'seasonality', 'precision', 'density', 'detrended')):
    """Return (X_train, X_test) DataFrames: audio features + the requested time feature groups."""
    al = pd.concat([tr, te], ignore_index=True)
    raw = al['track_album_release_date'].astype(str)
    d = parse_date(raw)
    F = al[AUDIO].copy()
    year = d.dt.year

    # baseline date features (what the original pipeline had)
    F['release_year'] = year
    F['release_month'] = d.dt.month
    F['days_since_release'] = (REF_DATE - d).dt.days
    F['is_jan1'] = ((d.dt.month == 1) & (d.dt.day == 1)).astype(int)

    if 'trend' in groups:
        F['t_frac_year'] = year + (d.dt.dayofyear - 1) / 365.25          # continuous time axis
        F['t_log_age'] = np.log1p(F['days_since_release'].clip(lower=0))
        F['t_decade'] = (year // 10) * 10

    if 'seasonality' in groups:
        dow, doy, mon = d.dt.dayofweek, d.dt.dayofyear, d.dt.month
        F['s_dow'] = dow
        F['s_doy'] = doy
        F['s_week'] = d.dt.isocalendar().week.astype(float)
        F['s_quarter'] = d.dt.quarter
        F['s_day'] = d.dt.day
        for name, v, period in (('mon', mon - 1, 12), ('dow', dow, 7), ('doy', doy - 1, 365.25)):
            F[f's_{name}_sin'] = np.sin(2 * np.pi * v / period)
            F[f's_{name}_cos'] = np.cos(2 * np.pi * v / period)
        F['s_is_friday'] = (dow == 4).astype(int)        # global release day since 2015
        F['s_is_tuesday'] = (dow == 1).astype(int)       # US release day before 2015
        F['s_holiday'] = mon.isin([11, 12]).astype(int)
        F['s_summer'] = mon.isin([6, 7, 8]).astype(int)

    if 'precision' in groups:
        F['p_precision'] = raw.str.len().map({10: 2, 7: 1, 4: 0}).fillna(2)

    if 'density' in groups:
        ym = d.dt.to_period('M').astype(str)
        F['n_same_date'] = raw.map(raw.value_counts())
        F['n_same_month'] = ym.map(ym.value_counts())
        F['n_same_year'] = year.map(year.value_counts())
        F['n_same_date_audio'] = al.groupby([raw, al['tempo'].round(0)])['tempo'].transform('size')

    if 'density2' in groups:
        # release volume in a window around the date (smoothed calendar density)
        day = (d - d.min()).dt.days
        cnt = day.value_counts().sort_index()
        full = cnt.reindex(np.arange(day.min(), day.max() + 1), fill_value=0)
        for w in (7, 30, 365):
            roll = full.rolling(w, center=True, min_periods=1).sum()
            F[f'n_win{w}'] = day.map(roll)
        wk = d.dt.to_period('W').astype(str)
        F['n_same_week'] = wk.map(wk.value_counts())
        F['date_rank_in_year'] = d.groupby(year).rank(pct=True)

    if 'groupagg' in groups:
        # release-date group ~ album / release batch: its average sound and the song's deviation
        g = al.groupby(raw)
        for c in ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'speechiness',
                  'instrumentalness', 'tempo', 'duration_ms']:
            mean = g[c].transform('mean')
            F[f'g_mean_{c}'] = mean
            F[f'g_dev_{c}'] = al[c] - mean
        F['g_std_energy'] = g['energy'].transform('std')
        F['g_std_loudness'] = g['loudness'].transform('std')

    if 'albumnn' in groups:
        # songs on the same date that sound alike (same artist / album): count within a radius
        cols = ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'speechiness']
        Zs = ((al[cols] - al[cols].mean()) / al[cols].std()).fillna(0).to_numpy()
        n_close, nn_dist = np.zeros(len(al)), np.full(len(al), np.nan)
        for _, idx in al.groupby(raw).indices.items():
            if len(idx) < 2:
                continue
            D = np.linalg.norm(Zs[idx, None, :] - Zs[None, idx, :], axis=2)
            np.fill_diagonal(D, np.inf)
            n_close[idx] = (D < 1.5).sum(axis=1)
            nn_dist[idx] = D.min(axis=1)
        F['a_n_close'] = n_close
        F['a_nn_dist'] = nn_dist

    if 'albumnn2' in groups:
        # finer version: several radii, 2nd-neighbour distance, and look-alikes across ALL dates
        cols = ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'speechiness']
        Zs = ((al[cols] - al[cols].mean()) / al[cols].std()).fillna(0).to_numpy()
        out = {k: np.zeros(len(al)) for k in ('a_n_close10', 'a_n_close20', 'a_n_close30')}
        out['a_nn2_dist'] = np.full(len(al), np.nan)
        for _, idx in al.groupby(raw).indices.items():
            if len(idx) < 2:
                continue
            D = np.linalg.norm(Zs[idx, None, :] - Zs[None, idx, :], axis=2)
            np.fill_diagonal(D, np.inf)
            for r in (1.0, 2.0, 3.0):
                out[f'a_n_close{int(r * 10)}'][idx] = (D < r).sum(axis=1)
            if len(idx) >= 3:
                out['a_nn2_dist'][idx] = np.sort(D, axis=1)[:, 1]
        from sklearn.neighbors import NearestNeighbors
        nbr = NearestNeighbors(n_neighbors=6).fit(Zs)
        dist, _ = nbr.kneighbors(Zs)
        out['a_global_nn_dist'] = dist[:, 1]
        out['a_global_nn5_dist'] = dist[:, 5]
        for k, v in out.items():
            F[k] = v

    if 'detrended' in groups:
        # era baseline: median of each audio feature over releases within +-2 years
        yrs = np.arange(year.min(), year.max() + 1)
        for c in DETREND:
            med = al.groupby(year)[c].median().reindex(yrs).interpolate(limit_direction='both')
            base = med.rolling(5, center=True, min_periods=1).median()
            F[f'dt_{c}'] = al[c] - year.map(base)

    n = len(tr)
    return F.iloc[:n].reset_index(drop=True), F.iloc[n:].reset_index(drop=True)

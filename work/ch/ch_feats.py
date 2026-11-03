"""ch agent: own re-implementation of the charlie branch's label-free feature builders (experiments/pdf_idea/
date_pool.py + experiments/similarity_study/similarity.py). Computed on the train+test pool, no labels; uri is
used only to exclude a song's own copies in the cross-date look-alike block (allowed: duplicate identification).

charlie_pool(tr, te) -> DataFrame over pool rows (train rows first, then test) with every block, and BLOCKS,
a dict block name -> column list. PROJECT = charlie's final model view (basic + density + secondary).
"""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances

FEAT = ['time_signature', 'speechiness', 'danceability', 'duration_ms', 'energy', 'mode', 'instrumentalness',
        'valence', 'key', 'tempo', 'loudness', 'acousticness', 'liveness']


def _std(M):
    M = np.asarray(M, float)
    med = np.nanmedian(M, axis=0)
    M = np.where(np.isfinite(M), M, med)
    sd = M.std(0)
    sd[sd == 0] = 1
    return (M - M.mean(0)) / sd


def charlie_pool(tr, te):
    df = pd.concat([tr, te], ignore_index=True)
    n = len(df)
    raw = df['track_album_release_date'].fillna('').astype(str).str.strip()
    date = pd.to_datetime(raw, format='mixed', errors='coerce')
    audio = df[FEAT].apply(pd.to_numeric, errors='coerce')
    known_day = raw.str.fullmatch(r'\d{4}-\d{2}-\d{2}').to_numpy()
    known_month = raw.str.fullmatch(r'\d{4}-\d{2}(-\d{2})?').to_numpy()
    year = date.dt.year.to_numpy(float)
    month = np.where(known_month, date.dt.month.to_numpy(float), np.nan)
    day = np.where(known_day, date.dt.day.to_numpy(float), np.nan)
    wd = np.where(known_day, date.dt.dayofweek.to_numpy(float), np.nan)
    doy = np.where(known_day, date.dt.dayofyear.to_numpy(float), np.nan)
    B = {c: audio[c].to_numpy(float) for c in FEAT}
    B.update(release_year=year, release_month=month, release_day=day, weekday=wd, day_of_year=doy,
             release_quarter=np.ceil(month / 3), fractional_year=year + (doy - 1) / 365.25,
             decade=np.floor(year / 10) * 10, is_jan1=raw.str.endswith('-01-01').to_numpy(float),
             is_friday=(wd == 4).astype(float), is_tuesday=(wd == 1).astype(float),
             is_summer=np.isin(month, [6, 7, 8]).astype(float), is_holiday=np.isin(month, [11, 12]).astype(float))
    for nm, v, p in [('month', month, 12), ('weekday', wd, 7), ('day_of_year', doy, 365.25)]:
        B[nm + '_sin'] = np.sin(2 * np.pi * v / p)
        B[nm + '_cos'] = np.cos(2 * np.pi * v / p)
    F = pd.DataFrame(B)
    basic_cols = list(F.columns)
    # --- density (pool row counts, duplicates included, as in charlie)
    yk = pd.Series(np.where(np.isfinite(year), year, -1))
    mk = pd.Series(np.where(np.isfinite(month), year * 100 + month, -1))
    dk = raw.where(known_day, '')
    F['same_year_count'] = yk.map(yk[yk != -1].value_counts()).fillna(0).to_numpy()
    F['same_month_count'] = mk.map(mk[mk != -1].value_counts()).fillna(0).to_numpy()
    F['same_date_count'] = dk.map(dk[dk != ''].value_counts()).fillna(0).to_numpy()
    st = pd.Series(list(zip(dk, audio['tempo'].round())))
    F['same_date_tempo_count'] = st.map(st.value_counts()).where(dk != '', 0).fillna(0).to_numpy()
    z = _std(audio.to_numpy())
    near = np.zeros((n, 5))
    near[:, 3:] = np.nan
    means = np.full((n, len(FEAT)), np.nan)
    groups = {k: np.asarray(v) for k, v in dk[dk != ''].groupby(dk[dk != '']).groups.items()}
    for key, ix in groups.items():
        means[ix] = audio.iloc[ix].mean().to_numpy()
        if len(ix) < 2:
            continue
        d = pairwise_distances(z[ix])
        np.fill_diagonal(d, np.inf)
        near[ix, :3] = np.column_stack([(d < r).sum(1) for r in (1., 1.5, 2.)])
        s = np.sort(d, axis=1)
        near[ix, 3] = s[:, 0]
        near[ix, 4] = np.where(np.isfinite(s[:, 1]), s[:, 1], np.nan)
    for j, nm in enumerate(('near_count_1', 'near_count_1_5', 'near_count_2', 'nearest_same_date',
                            'second_nearest_same_date')):
        F[nm] = near[:, j]
    for j, c in enumerate(FEAT):
        F['same_date_mean_' + c] = means[:, j]
        F['same_date_deviation_' + c] = B[c] - means[:, j]
    density_cols = [c for c in F.columns if c not in basic_cols]
    # --- zlog: standardized audio with log10 instrumentalness (similarity.py)
    zlog = pd.DataFrame(z, columns=FEAT)
    li = np.log10(np.maximum(B['instrumentalness'], 1e-5))
    zlog['instrumentalness'] = _std(li[:, None])[:, 0]
    full = known_day
    valid = audio.notna().any(axis=1).to_numpy()
    fv = pd.Series(np.where(full & valid, raw, ''))
    G2 = [np.asarray(v) for k, v in fv[fv != ''].groupby(fv[fv != '']).groups.items()]
    prod = zlog[['energy', 'loudness', 'acousticness', 'instrumentalness']].to_numpy()
    BL = {}
    # subset: nearest / second / mean same-date distance and within-radius counts on three audio subsets
    specs = {'production': ['energy', 'loudness', 'acousticness', 'instrumentalness'],
             'rhythm': ['tempo', 'danceability', 'speechiness'],
             'all_no_key': [c for c in FEAT if c not in ('key', 'mode', 'time_signature')]}
    cols = []
    for tag, cs in specs.items():
        zs = zlog[cs].to_numpy()
        dist = np.full((n, 3), np.nan)
        cnt = np.zeros((n, 3))
        for ix in G2:
            if len(ix) < 2:
                continue
            d = pairwise_distances(zs[ix])
            np.fill_diagonal(d, np.inf)
            s = np.sort(d, axis=1)
            dist[ix, 0] = s[:, 0]
            dist[ix, 1] = np.where(np.isfinite(s[:, 1]), s[:, 1], np.nan)
            dist[ix, 2] = np.nanmean(np.where(np.isfinite(d), d, np.nan), axis=1)
            for j, r in enumerate((1., 2., 3.)):
                cnt[ix, j] = (d < r).sum(1)
        for j, lab in enumerate(('nearest', 'second', 'mean')):
            F[f'{tag}_{lab}'] = dist[:, j]
            cols.append(f'{tag}_{lab}')
        for j, r in enumerate((1, 2, 3)):
            F[f'{tag}_within_{r}'] = np.where(valid, cnt[:, j], np.nan)
            cols.append(f'{tag}_within_{r}')
    BL['subset'] = cols
    BL['subset_mean'] = [f'{t}_mean' for t in specs]
    # cluster: DBSCAN on production audio within each date
    cols = []
    for eps in (1., 2.):
        size, center, spread, num = (np.full(n, np.nan) for _ in range(4))
        for ix in G2:
            lab = DBSCAN(eps=eps, min_samples=2).fit_predict(prod[ix])
            nid = int(lab.max() + 1)
            for j in np.flatnonzero(lab < 0):
                lab[j] = nid
                nid += 1
            num[ix] = len(np.unique(lab))
            for L in np.unique(lab):
                ii = ix[lab == L]
                dd = np.linalg.norm(prod[ii] - prod[ii].mean(0), axis=1)
                size[ii] = len(ii)
                center[ii] = dd
                spread[ii] = dd.mean()
        for nm, v in [(f'cluster_size_{eps}', size), (f'cluster_center_dist_{eps}', center),
                      (f'cluster_spread_{eps}', spread), (f'date_cluster_count_{eps}', num)]:
            F[nm] = v
            cols.append(nm)
        if eps == 1.:
            F['cluster_size_x_year'] = (year - 2000) * size
            cols.append('cluster_size_x_year')
    BL['cluster'] = cols
    # position: within-date percentile rank and median delta
    cols = []
    for c in ('duration_ms', 'energy', 'loudness', 'speechiness'):
        v = audio[c].where(full & valid)
        gb = v.groupby(raw.where(full & valid))
        F[f'date_percentile_{c}'] = gb.rank(pct=True).to_numpy()
        F[f'date_median_delta_{c}'] = (v - gb.transform('median')).to_numpy()
        cols += [f'date_percentile_{c}', f'date_median_delta_{c}']
    BL['position'] = cols
    # crossdate: nearest production-space look-alike released on ANOTHER date (other uri)
    ordn = np.where(date.notna().to_numpy(), date.to_numpy(dtype='datetime64[D]').astype('int64').astype(float), np.nan)
    vd = full & np.isfinite(ordn) & valid
    ids = df['uri'].astype(str).to_numpy()
    dates = raw.to_numpy()
    cross = np.full((n, 3), np.nan)
    count = np.zeros(n)
    for s0 in range(0, n, 256):
        idx = np.arange(s0, min(s0 + 256, n))
        d = pairwise_distances(prod[idx], prod)
        mask = (dates[idx, None] != dates[None, :]) & (ids[idx, None] != ids[None, :]) & vd[idx, None] & vd[None, :]
        d[~mask] = np.inf
        arg = np.argmin(d, axis=1)
        best = d[np.arange(len(idx)), arg]
        good = np.isfinite(best)
        cross[idx[good], 0] = best[good]
        cross[idx[good], 1] = np.abs(ordn[idx[good]] - ordn[arg[good]])
        count[idx] = (d < 1.).sum(1)
        for j, i in enumerate(idx):
            hit = np.flatnonzero(d[j] < 1.)
            if len(hit):
                cross[i, 2] = ordn[hit].min()
    F['nearest_other_date_prod'] = cross[:, 0]
    F['nearest_other_date_gap_days'] = cross[:, 1]
    F['near_other_date_earliest_ordinal'] = cross[:, 2]
    F['near_other_date_count'] = np.where(valid, count, np.nan)
    BL['crossdate'] = ['nearest_other_date_prod', 'nearest_other_date_gap_days', 'near_other_date_earliest_ordinal',
                       'near_other_date_count']
    # tempo-ambiguous rhythm look-alike (half / double tempo)
    tempo = B['tempo']
    dr = zlog[['danceability', 'speechiness']].to_numpy()
    tout = np.full(n, np.nan)
    tcnt = np.zeros(n)
    for ix in G2:
        if len(ix) < 2:
            continue
        dt = np.abs(np.log2(np.maximum(tempo[ix, None], 1) / np.maximum(tempo[None, ix], 1)))
        dt = np.minimum.reduce([dt, np.abs(dt - 1), np.abs(dt + 1)])
        d = np.sqrt(pairwise_distances(dr[ix]) ** 2 + (dt / .25) ** 2)
        np.fill_diagonal(d, np.inf)
        tout[ix] = d.min(1)
        tcnt[ix] = (d < 1.5).sum(1)
    F['tempo_ambiguous_nearest'] = tout
    F['tempo_ambiguous_count'] = np.where(valid, tcnt, np.nan)
    BL['tempo'] = ['tempo_ambiguous_nearest', 'tempo_ambiguous_count']
    # interactions
    F['estimated_beats'] = B['duration_ms'] * B['tempo'] / 60000.
    F['estimated_bars'] = F['estimated_beats'] / np.where(B['time_signature'] > 0, B['time_signature'], np.nan)
    F['energy_x_danceability'] = B['energy'] * B['danceability']
    F['energy_x_low_valence'] = B['energy'] * (1 - B['valence'])
    F['acoustic_x_instrumental'] = B['acousticness'] * B['instrumentalness']
    F['weekday_x_era'] = wd * (year - 2000)
    F['density_x_era'] = F['same_date_count'].to_numpy() * (year - 2000)
    BL['interactions'] = ['estimated_beats', 'estimated_bars', 'energy_x_danceability', 'energy_x_low_valence',
                          'acoustic_x_instrumental', 'weekday_x_era', 'density_x_era']
    # secondary (part of charlie's final model)
    e, l = B['energy'], B['loudness']
    g = np.isfinite(e) & np.isfinite(l)
    sl, ic = np.polyfit(e[g], l[g], 1)
    F['loudness_energy_residual'] = l - (sl * e + ic)
    F['year_density_percentile'] = F['same_date_count'].groupby(F['release_year']).rank(pct=True).to_numpy()
    oh = []
    for c, vals in [('key', range(12)), ('mode', range(2)), ('time_signature', range(1, 8))]:
        for v in vals:
            F[f'{c}_is_{v}'] = np.where(np.isfinite(B[c]), (B[c] == v).astype(float), np.nan)
            oh.append(f'{c}_is_{v}')
    km = []
    for k in range(12):
        for m in range(2):
            F[f'key_{k}_mode_{m}'] = np.where(np.isfinite(B['key']) & np.isfinite(B['mode']),
                                              ((B['key'] == k) & (B['mode'] == m)).astype(float), np.nan)
            km.append(f'key_{k}_mode_{m}')
    kn = np.full(n, np.nan)
    key = B['key']
    fk = pd.Series(np.where(full & valid & np.isfinite(key), raw, ''))
    for _, ix in fk[fk != ''].groupby(fk[fk != '']).groups.items():
        ix = np.asarray(ix)
        if len(ix) < 2:
            continue
        dl = np.abs(key[ix, None] - key[None, ix])
        cyc = np.minimum(dl, 12 - dl) / 3.
        d = np.sqrt(pairwise_distances(prod[ix]) ** 2 + cyc ** 2)
        np.fill_diagonal(d, np.inf)
        kn[ix] = d.min(1)
    F['nearest_same_date_cyclic_key'] = kn
    BL['basic'] = basic_cols
    BL['density'] = density_cols
    BL['secondary'] = ['loudness_energy_residual', 'year_density_percentile'] + oh + km + ['nearest_same_date_cyclic_key']
    BL['onehot'] = oh + km
    BL['cyckey'] = ['nearest_same_date_cyclic_key']
    BL['ydp'] = ['year_density_percentile']
    BL['nn13'] = ['near_count_1', 'near_count_1_5', 'near_count_2', 'nearest_same_date', 'second_nearest_same_date']
    BL['PROJECT'] = basic_cols + density_cols + BL['secondary']
    F = F.replace([np.inf, -np.inf], np.nan)
    return F, BL

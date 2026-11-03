"""Named feature configurations (column lists over the cached pool table). Column order is fixed so the
cached CV results stay valid (LightGBM column sampling depends on column order)."""
import sys
sys.path.insert(0, './work/fe')
from allfeats import *

XP, COLS = group_cols()
UCOLS = [c for c in XP.columns if c.startswith('u_')]


def configs():
    cols = COLS
    base = sel(cols, FULL)
    C = {'full67': base}
    for g in FULL:
        C[f'full-{g}'] = sel(cols, [x for x in FULL if x != g])
    for g in ['precision', 'density2', 'detrended']:
        C[f'full+{g}'] = sel(cols, FULL + [g])
    C['base17'] = sel(cols, [])
    for g in ['trend', 'seasonality', 'precision', 'density', 'density2', 'detrended', 'groupagg', 'albumnn', 'albumnn2']:
        C[f'base17+{g}'] = sel(cols, [g])
    C['full+dupcnt'] = base + ['dup_uri_cnt']
    C['full+dup+twin'] = base + ['dup_uri_cnt', 'twin_n_uri', 'twin_n_dates']
    repl = [c for c in base if c not in cols['groupagg'] + cols['albumnn'] + cols['albumnn2'] + ['n_same_date']]
    C['dedupAlbum+dupcnt'] = repl + UCOLS + ['dup_uri_cnt']
    C['dedupAlbum_(no_dupcnt)'] = repl + UCOLS
    C['full+dedupAlbum+dupcnt'] = base + UCOLS + ['dup_uri_cnt']
    blocks = {'km8': [c for c in XP.columns if c.startswith('km8_')], 'km16min': ['km16_lab', 'km16_dmin'],
              'pca5': [f'pca{j}' for j in range(5)], 'gmm8': [c for c in XP.columns if c.startswith('gmm8_')],
              'resid': [c for c in XP.columns if c.startswith('r_')],
              'relday': ['rd_std_day', 'rd_fri_post', 'rd_year_only'],
              'album25': [c for c in XP.columns if c.startswith('alb_')],
              'album25size': ['alb_size', 'alb_n_on_date', 'alb_share'],
              'typ': [c for c in XP.columns if c.startswith('ty_')],
              'typ_era': ['ty_era_k5', 'ty_era_k20', 'ty_ratio_k5', 'ty_ratio_k20']}
    for k, v in blocks.items():
        C['full+' + k] = base + v
    C['L_noA1_noG'] = sel(cols, ['trend', 'seasonality', 'density', 'albumnn2'])
    C['L_noA1_noG_noT'] = sel(cols, ['seasonality', 'density', 'albumnn2'])
    C['L_noA1_noT'] = sel(cols, ['seasonality', 'density', 'albumnn2', 'groupagg'])
    # detrended-based sets
    C['L1D'] = sel(cols, ['trend', 'seasonality', 'density', 'albumnn2', 'groupagg', 'detrended'])
    C['L2D'] = sel(cols, ['trend', 'seasonality', 'density', 'albumnn2', 'detrended'])
    return C


def matrix(name, C=None):
    C = C or configs()
    c = C[name]
    return XP[c].iloc[:len(tr)].reset_index(drop=True), XP[c].iloc[len(tr):].reset_index(drop=True)


SUB = {
    'seas_dow': ['s_dow', 's_dow_sin', 's_dow_cos', 's_is_friday', 's_is_tuesday'],
    'seas_doy': ['s_doy', 's_week', 's_doy_sin', 's_doy_cos', 's_day'],
    'seas_mon': ['s_quarter', 's_mon_sin', 's_mon_cos', 's_holiday', 's_summer'],
    'g_mean': [f'g_mean_{c}' for c in ['danceability', 'energy', 'loudness', 'acousticness', 'valence',
                                       'speechiness', 'instrumentalness', 'tempo', 'duration_ms']],
    'g_dev': [f'g_dev_{c}' for c in ['danceability', 'energy', 'loudness', 'acousticness', 'valence',
                                     'speechiness', 'instrumentalness', 'tempo', 'duration_ms']],
    'g_std': ['g_std_energy', 'g_std_loudness'],
    'a2_close': ['a_n_close10', 'a_n_close20', 'a_n_close30'],
    'a2_nn2': ['a_nn2_dist'],
    'a2_global': ['a_global_nn_dist', 'a_global_nn5_dist'],
    'dens_date': ['n_same_date', 'n_same_date_audio'],
    'dens_my': ['n_same_month', 'n_same_year'],
    'base_date': ['release_year', 'release_month', 'days_since_release', 'is_jan1'],
}


def drop(base_list, cols_to_drop):
    return [c for c in base_list if c not in set(cols_to_drop)]


def sub_configs(ref_name='full67', C=None):
    C = C or configs()
    out = {}
    for k, v in SUB.items():
        out[f'{ref_name}-{k}'] = drop(C[ref_name], v)
    return out

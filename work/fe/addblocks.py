"""python addblocks.py REF blk1 blk2 ... -> screen (seed set 0) adding each block (list of columns) to REF."""
import sys
sys.path.insert(0, './work/fe')
from configs import *
C = configs()
ref = sys.argv[1]
BL = {
    'dupcnt': ['dup_uri_cnt'],
    'dup+twin': ['dup_uri_cnt', 'twin_n_uri', 'twin_n_dates'],
    'typ': [c for c in XP.columns if c.startswith('ty_')],
    'typ_era': ['ty_era_k5', 'ty_era_k20', 'ty_ratio_k5', 'ty_ratio_k20'],
    'album25': [c for c in XP.columns if c.startswith('alb_')],
    'album25size': ['alb_size', 'alb_n_on_date', 'alb_share'],
    'km8': [c for c in XP.columns if c.startswith('km8_')],
    'gmm8': [c for c in XP.columns if c.startswith('gmm8_')],
    'pca5': [f'pca{j}' for j in range(5)],
    'resid': [c for c in XP.columns if c.startswith('r_')],
    'relday': ['rd_std_day', 'rd_fri_post', 'rd_year_only'],
    'precision': ['p_precision'],
    'density2': ['n_win7', 'n_win30', 'n_win365', 'n_same_week', 'date_rank_in_year'],
    'detrended': ['dt_danceability', 'dt_duration_ms', 'dt_energy', 'dt_valence', 'dt_tempo', 'dt_loudness',
                  'dt_acousticness', 'dt_speechiness', 'dt_instrumentalness', 'dt_liveness'],
    'erank': [c for c in XP.columns if c.startswith('er_rank_')],
    'densratio': ['dr_date_over_month', 'dr_date_over_year', 'dr_date_over_typical'],
}
Xr, _ = matrix(ref, C)
R = cv_cached(ref, Xr)
log(f'\n## Block additions to {ref} (seed set 0)\n| tag | nfeat | AUC | diff vs {ref} | pos | per-rep diff | note |\n|---|---|---|---|---|---|---|')
for k in sys.argv[2:]:
    cols = C[ref] + [c for c in BL[k] if c not in C[ref]]
    X = XP[cols].iloc[:len(tr)].reset_index(drop=True)
    r = cv_cached(f'{ref}+{k}', X)
    report(f'{ref} + {k}', r, R, len(cols), 'block add')

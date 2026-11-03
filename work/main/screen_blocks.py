"""Screen label-free blocks on top of a reference config (reuses fe agent's cached pool table + CV cache).
python -I screen_blocks.py REF SEEDS(comma) blk1 blk2 ..."""
import sys
sys.path.insert(0, './work/fe')
from configs import *
C = configs()
ref = sys.argv[1]
seeds = [int(s) for s in sys.argv[2].split(',')]
BL = {
    'typ': [c for c in XP.columns if c.startswith('ty_')],
    'typ_era': ['ty_era_k5', 'ty_era_k20', 'ty_ratio_k5', 'ty_ratio_k20'],
    'album25size': ['alb_size', 'alb_n_on_date', 'alb_share'],
    'gmm8': [c for c in XP.columns if c.startswith('gmm8_')],
    'gmm8ll': ['gmm8_ll'],
    'pca5': [f'pca{j}' for j in range(5)],
    'resid': [c for c in XP.columns if c.startswith('r_')],
    'relday': ['rd_std_day', 'rd_fri_post', 'rd_year_only'],
    'density2': ['n_win7', 'n_win30', 'n_win365', 'n_same_week', 'date_rank_in_year'],
    'erank': [c for c in XP.columns if c.startswith('er_rank_')],
    'densratio': ['dr_date_over_month', 'dr_date_over_year', 'dr_date_over_typical'],
    'groupagg': [c for c in XP.columns if c.startswith('g_')],
}
def runs(name, cols):
    X = XP[cols].iloc[:len(tr)].reset_index(drop=True)
    return [cv_cached(name, X, seed_off=s) for s in seeds]
R = runs(ref, C[ref])
ra = np.mean([r['aucs'] for r in R], axis=0)
print(f'REF {ref} seeds {seeds}: {ra.mean():.5f}', flush=True)
for k in sys.argv[3:]:
    cols = C[ref] + [c for c in BL[k] if c not in C[ref]]
    Q = runs(f'{ref}+{k}', cols)
    qa = np.mean([q['aucs'] for q in Q], axis=0)
    dd = qa - ra
    print(f'| {ref}+{k} | {len(cols)} | {qa.mean():.5f} | {dd.mean():+.5f} | {(dd>0).sum()}/5 | {" ".join(f"{x:+.4f}" for x in dd)} |', flush=True)

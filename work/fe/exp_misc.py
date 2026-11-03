import sys; sys.path.insert(0, './work/fe')
from probe import *
from fe_lib import *
cols, Xa, Xta = groups_columns()
FULL = ['trend', 'seasonality', 'density', 'albumnn', 'albumnn2', 'groupagg']
def sel(groups): return cols['base'] + [c for g in groups for c in cols[g]]
base_cols = sel(FULL)
base = cv_cached('full67', Xa[base_cols])
al, raw, d, akey = make_pool(tr, te)
n = len(tr)
blocks = {
    'km8': cluster_feats(al, 8),
    'km16min': cluster_feats(al, 16)[['km16_lab', 'km16_dmin']],
    'pca5': pca_feats(al, 5),
    'gmm8': gmm_feats(al, 8),
    'resid': resid_feats(al, d),
    'relday': release_day_feats(d, raw),
}
A, _ = album_feats(al, raw, akey, t=2.5)
blocks['album25'] = A
blocks['album25size'] = A[['alb_size', 'alb_n_on_date', 'alb_share']]
log('\n## 3/4/5. Album reconstruction, genre proxies, misc (screen, seed set 0; each block added to full67)\n| tag | nfeat | AUC | diff vs full67 | pos | per-rep diff | note |\n|---|---|---|---|---|---|---|')
XP = pd.concat([Xa, Xta], ignore_index=True)
for tag, B in blocks.items():
    B = B.reset_index(drop=True)
    X = pd.concat([XP[base_cols], B], axis=1).iloc[:n].reset_index(drop=True)
    r = cv_cached('full+' + tag, X); report('full67 + ' + tag, r, base, X.shape[1])

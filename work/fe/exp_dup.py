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
DUP = dup_feats(al, raw, d, akey)
U = dedup_group_feats(al, raw, d, akey, by='uri')
XA = pd.concat([pd.concat([Xa, Xta], ignore_index=True), DUP, U], axis=1)
Xtr = XA.iloc[:n].reset_index(drop=True)
print(DUP.iloc[:n].describe().T.to_string())
log('\n## 2. Duplicate / twin signals and de-duplicated album features (screen, seed set 0; diff vs full67)\n| tag | nfeat | AUC | diff vs full67 | pos | per-rep diff | note |\n|---|---|---|---|---|---|---|')
exps = {
    'full+dupcnt': base_cols + ['dup_uri_cnt'],
    'full+dup+twin': base_cols + ['dup_uri_cnt', 'twin_n_uri', 'twin_n_dates'],
}
ucols = list(U.columns)
# replace teammate density(n_same_date)/groupagg/albumnn/albumnn2 by de-duplicated versions
repl = [c for c in base_cols if c not in cols['groupagg'] + cols['albumnn'] + cols['albumnn2'] + ['n_same_date']]
exps['dedupAlbum+dupcnt'] = repl + ucols + ['dup_uri_cnt']
exps['dedupAlbum (no dupcnt)'] = repl + ucols
exps['full+dedupAlbum+dupcnt'] = base_cols + ucols + ['dup_uri_cnt']
for tag, c in exps.items():
    r = cv_cached(tag.replace(' ', '_'), Xtr[c]); report(tag, r, base, len(c))

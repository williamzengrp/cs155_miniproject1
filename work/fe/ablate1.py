import sys; sys.path.insert(0, './work/fe')
from probe import *
cols, Xa, Xta = groups_columns()
for g, c in cols.items(): print(g, len(c), c)
FULL = ['trend', 'seasonality', 'density', 'albumnn', 'albumnn2', 'groupagg']
def sel(groups): return cols['base'] + [c for g in groups for c in cols[g]]
base = cv_cached('full67', Xa[sel(FULL)])
log('\n## 1. Group ablation (screen, seed set 0)\n| tag | nfeat | AUC | diff vs full67 | pos | per-rep diff | note |\n|---|---|---|---|---|---|---|')
for g in FULL:
    s = sel([x for x in FULL if x != g]); r = cv_cached(f'full-{g}', Xa[s]); report(f'full67 - {g}', r, base, len(s), 'leave-one-group-out')
for g in ['precision', 'density2', 'detrended']:
    s = sel(FULL + [g]); r = cv_cached(f'full+{g}', Xa[s]); report(f'full67 + {g}', r, base, len(s), 'add to full')
b17 = cv_cached('base17', Xa[sel([])])
report('base17 (audio+4 date)', b17, base, len(sel([])), 'vs full67')
log('| (forward from base17; diff vs base17) | | | | | | |')
for g in ['trend', 'seasonality', 'precision', 'density', 'density2', 'detrended', 'groupagg', 'albumnn', 'albumnn2']:
    s = sel([g]); r = cv_cached(f'base17+{g}', Xa[s]); report(f'base17 + {g}', r, b17, len(s), 'forward add')

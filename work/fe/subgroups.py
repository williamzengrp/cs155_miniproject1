"""python subgroups.py REF sub1 sub2 ... -> screen (seed set 0) removing each SUB group from config REF."""
import sys
sys.path.insert(0, './work/fe')
from configs import *
C = configs()
ref = sys.argv[1]
subs = sys.argv[2:] or list(SUB)
Xr, _ = matrix(ref, C)
R = cv_cached(ref, Xr)
log(f'\n## Sub-group leave-out screen from {ref} (seed set 0)\n| tag | nfeat | AUC | diff vs {ref} | pos | per-rep diff | note |\n|---|---|---|---|---|---|---|')
for k in subs:
    cols = drop(C[ref], SUB[k])
    if len(cols) == len(C[ref]):
        continue
    X = XP[cols].iloc[:len(tr)].reset_index(drop=True)
    r = cv_cached(f'{ref}-{k}', X)
    report(f'{ref} - {k}', r, R, len(cols), 'sub-group leave-out')

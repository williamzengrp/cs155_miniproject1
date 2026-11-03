"""Paired single-model comparison: python -I cmp.py REF cand1 cand2 ...  (same split set)."""
import sys
import numpy as np
sys.path.insert(0, './work')
import harness as H
ref = H.load_res(sys.argv[1])
print(f'  {sys.argv[1]:34s} {ref["aucs"].mean():.5f}')
for c in sys.argv[2:]:
    r = H.load_res(c)
    d = r['aucs'] - ref['aucs']
    print(f'  {c:34s} {r["aucs"].mean():.5f}  diff {d.mean():+.5f}  pos {(d > 0).sum()}/{len(d)}  per-rep {np.round(d, 4)}')

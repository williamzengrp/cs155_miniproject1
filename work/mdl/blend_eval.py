"""Marginal value of each saved mdl_* model in a per-repeat rank-average blend with the 5 baseline trees
(standard 5x5 splits). Fixed weights only (equal 1/6, and 0.1 / 0.2 / 0.3 for the new model).
usage: blend_eval.py mdl_a mdl_b ... [--combo a+b ...]"""
import sys
sys.path.insert(0, './work/mdl')
import numpy as np
from scipy.stats import spearmanr
import common as C

H = C.H
tr, te, y = H.load()
B = C.load_base()
b = [B[n]['oof'] for n in C.BASE]
a0 = C.blend_aucs(y, b, [0.2] * 5)
print(f'5-tree blend {a0.mean():.5f} {np.round(a0, 4)}')
args = sys.argv[1:]
names = [a for a in args if '+' not in a]
combos = [a.split('+') for a in args if '+' in a]
R = {n: H.load_res(n) for n in set(names) | {m for c in combos for m in c}}


def rank_corr(o):
    cs = []
    for r in range(o.shape[0]):
        m = ~np.isnan(o[r])
        base_rank = sum(C._rank01(x[r, m]) for x in b)
        cs.append(spearmanr(o[r, m], base_rank)[0])
    return float(np.mean(cs))


for n in names:
    o = R[n]['oof']
    s = C.standalone(y, o)
    print(f'\n{n}: standalone {s.mean():.5f} {np.round(s, 4)}  rank-corr with tree blend {rank_corr(o):.3f}'
          f'  test_full n={len(R[n]["test_full"])}')
    C.blend_report(y, o, base=B, label=n)

for c in combos:
    # the combo members are first rank-averaged (equal weights) into one extra component
    oofs = [R[m]['oof'] for m in c]
    R_ = oofs[0].shape[0]
    comp = np.full_like(oofs[0], np.nan)
    for r in range(R_):
        m = ~np.isnan(oofs[0][r])
        comp[r, m] = np.mean([C._rank01(o[r, m]) for o in oofs], axis=0)
    s = C.standalone(y, comp)
    print(f'\ncombo {"+".join(c)} (equal rank-avg as one component): standalone {s.mean():.5f} {np.round(s, 4)}')
    C.blend_report(y, comp, base=B, label='+'.join(c), ws=(0.1, 0.2, 0.3, 0.4))

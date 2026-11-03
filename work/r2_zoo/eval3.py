"""Fair 3-seed comparisons on std: candidate (3 seeds) vs its plain counterpart (3 seeds), in the final8 blend.
python -I eval3.py SET cand:ref:member [...]   e.g. cat_alb:cat_ref:fkt_cat  (raw files offsets 0,100,200)"""
import sys, numpy as np
sys.path.insert(0, './work/r2_zoo')
import bl
from sklearn.metrics import roc_auc_score
S = sys.argv[1]
tr, te, y = bl.H.load()
names = bl.MEMBERS[S]
base = [bl.H.load_res(n)['oof'] for n in names]
a = bl.blend_auc(base, y)


def alone(o):
    return np.mean([roc_auc_score(y[~np.isnan(o[r])], o[r][~np.isnan(o[r])]) for r in range(o.shape[0])])


def show(tag, b):
    d = bl.blend_auc(b, y) - a
    print(f'  {tag:44s} {d.mean():+.5f} pos {(d > 0).sum()}/{len(d)} {np.round(d * 1e4).astype(int)}', flush=True)
    return d


print(f'[{S}] final8 {a.mean():.5f}')
for spec in sys.argv[2:]:
    c, r, mem = spec.split(':')
    oc = bl.load_raw(S, f'{c}:0+100+200')
    orf = bl.load_raw(S, f'{r}:0+100+200')
    j = names.index(mem)
    print(f' {c} 3s alone {alone(oc):.5f} | {r} 3s alone {alone(orf):.5f} | {mem} alone {alone(base[j]):.5f}')
    show(f'ADD {c}(3s)', base + [oc])
    show(f'ADD {r}(3s)', base + [orf])
    b = list(base); b[j] = oc; d1 = show(f'REPL {mem} -> {c}(3s)', b)
    b = list(base); b[j] = orf; d2 = show(f'REPL {mem} -> {r}(3s)  [seed-avg only]', b)
    d = d1 - d2
    print(f'  {"pure effect (REPL cand3s - REPL ref3s)":44s} {d.mean():+.5f} pos {(d > 0).sum()}/{len(d)}')

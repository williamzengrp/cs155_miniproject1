"""Post-hoc combinations of the small positive zoo members (std only; selection-biased, for information)."""
import sys, numpy as np
sys.path.insert(0, './work/r2_zoo')
import bl
tr, te, y = bl.H.load()
names = bl.MEMBERS['std']
base = [bl.H.load_res(n)['oof'] for n in names]
a = bl.blend_auc(base, y)
R = lambda s: bl.load_raw('std', s)
CA, LL, C3, LR = R('cat_alb:0+100+200'), R('lgb_linear:0+100+200'), R('cat_cat3:0'), R('lgbm_ref:0+100+200')
def rep(d, extra=()):
    b = list(base)
    for k, v in d.items(): b[names.index(k)] = v
    return b + list(extra)
cases = {
 'REPL lgbm->lgb_linear(3s)': rep({'fkd_lgbm': LL}),
 'REPL lgbm->lgbm_ref(3s) [seed-avg only]': rep({'fkd_lgbm': LR}),
 'ADD lgb_linear(3s)': rep({}, [LL]),
 'REPL cat->cat_alb, lgbm->lgb_linear': rep({'fkt_cat': CA, 'fkd_lgbm': LL}),
 'REPL cat->cat_alb, lgbm->lgb_linear + ADD cat_cat3': rep({'fkt_cat': CA, 'fkd_lgbm': LL}, [C3]),
 'ADD cat_alb + lgb_linear': rep({}, [CA, LL]),
}
for k, b in cases.items():
    d = bl.blend_auc(b, y) - a
    print(f'  {k:52s} {d.mean():+.5f} pos {(d>0).sum()}/5 {np.round(d*1e4).astype(int)}', flush=True)

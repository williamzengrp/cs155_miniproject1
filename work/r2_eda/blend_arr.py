"""r2_eda helper: equal-weight rank blend of final8 (std) with members replaced / added from arbitrary npz files
(e.g. r2_eda/cache probes) -- same computation as blend.evaluate(nested=False).
python -I blend_arr.py 'repl=fkd_lgbm:/path/a.npz' 'add=/path/b.npz' ... ; several specs separated by ';' compare
variant A (first arg) vs variant B (second arg)."""
import sys
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
sys.path.insert(0, './work')
import harness as H

MEM = ['fkd_lgbm', 'fkt_xgb', 'fkt_cat', 'fkd_et', 'fkd_rf', 'mdl_mlp_ple', 'ch_et_proj', 'ch_lgbm_proj']


def blend(oofs, y):
    R = oofs[0].shape[0]
    out = []
    for r in range(R):
        m = ~np.isnan(oofs[0][r])
        O = np.column_stack([rankdata(o[r, m]) / m.sum() for o in oofs])
        out.append(roc_auc_score(y[m], O.mean(1)))
    return np.array(out)


def members(spec):
    oo = {k: H.load_res(k)['oof'] for k in MEM}
    order = list(MEM)
    extra = []
    for part in [p for p in spec.split(';') if p]:
        kind, val = part.split('=', 1)
        if kind == 'repl':
            old, path = val.split(':', 1)
            oo[old] = np.load(path)['oof']
        elif kind == 'add':
            extra.append(np.load(val)['oof'])
    return [oo[k] for k in order] + extra


if __name__ == '__main__':
    tr, te, y = H.load()
    base = blend(members(''), y)
    print('final8', base.mean().round(5), np.round(base, 4))
    res = []
    for spec in sys.argv[1:]:
        b = blend(members(spec), y)
        res.append(b)
        d = b - base
        print(f'{spec[-60:]:60s} {b.mean():.5f} gain {d.mean():+.5f} pos {(d > 0).sum()}/{len(d)} {np.round(d, 4)}')
    if len(res) == 2:
        d = res[1] - res[0]
        print(f'B - A: {d.mean():+.5f} pos {(d > 0).sum()}/{len(d)} {np.round(d, 4)}')

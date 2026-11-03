"""Screening blend gains for raw zoo OOFs (same computation as main/eval_candidate.py / blend.evaluate 'equal').
python -I bl.py SET spec [spec ...] [--rep member]
  SET std|c9 ; spec = model:off[+off...] (raw r2_zoo/raw files averaged over offsets) ; --rep m = also REPLACE m."""
import sys, numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
sys.path.insert(0, './work')
sys.path.insert(0, './work/main')
import harness as H
from eval_candidate import MEMBERS
ZOO = './work/r2_zoo'
SETS = {'std': (1000, 5), 'c9': (9000, 3)}


def blend_auc(oofs, y):
    R = oofs[0].shape[0]
    out = []
    for r in range(R):
        m = ~np.isnan(oofs[0][r])
        O = np.column_stack([rankdata(o[r, m]) / m.sum() for o in oofs])
        out.append(roc_auc_score(y[m], O.mean(1)))
    return np.array(out)


def load_raw(S, spec):
    nm, offs = spec.split(':')
    s0, nr = SETS[S]
    arr = [np.load(f'{ZOO}/raw/{s0}_{nr}_{nm}_off{o}.npz')['oof'] for o in offs.split('+')]
    return np.mean(arr, 0)


if __name__ == '__main__':
    S = sys.argv[1]
    args = sys.argv[2:]
    rep = None
    if '--rep' in args:
        i = args.index('--rep'); rep = args[i + 1]; args = args[:i] + args[i + 2:]
    tr, te, y = H.load()
    base_names = MEMBERS[S]
    base = [H.load_res(n)['oof'] for n in base_names]
    a = blend_auc(base, y)
    print(f'[{S}] final8 {a.mean():.5f}')
    for spec in args:
        o = load_raw(S, spec)
        R = o.shape[0]
        sa = np.array([roc_auc_score(y[~np.isnan(o[r])], o[r][~np.isnan(o[r])]) for r in range(R)])
        d = blend_auc(base + [o], y) - a
        line = f'  {spec:28s} alone {sa.mean():.5f}  ADD {d.mean():+.5f} pos {(d > 0).sum()}/{R} {np.round(d * 1e4).astype(int)}'
        if rep:
            j = [i for i, n in enumerate(base_names) if rep in n][0]
            b2 = list(base); b2[j] = o
            d2 = blend_auc(b2, y) - a
            line += f' | REPL {base_names[j]} {d2.mean():+.5f} pos {(d2 > 0).sum()}/{R}'
        # correlation with blend (rank) on rep 0
        m = ~np.isnan(o[0])
        bl = np.mean([rankdata(b[0, m]) for b in base], 0)
        line += f' | rho(blend) {np.corrcoef(rankdata(o[0, m]), bl)[0, 1]:.3f}'
        print(line, flush=True)

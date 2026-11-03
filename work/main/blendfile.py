"""Build a submission file from a weighted rank average, same tokens as pubsim.py:
  ch = Charlie's submitted file (final_submissions/reference_60_40_confirmed_positive.csv)
  v2 / f8 = equal rank average of their members' test_full;  anything else = results/<name> test_full.
python -I work/main/blendfile.py OUT_NAME "tok:w,tok:w,..."   (run from the repo root) -> submissions/OUT_NAME.csv"""
import sys
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
sys.path.insert(0, './work')
sys.path.insert(0, './work/main')
import harness as H
from pubsim import V2, F8
OUT = './submissions/'
CH = './final_submissions/reference_60_40_confirmed_positive.csv'
tr, te, y = H.load()
n = len(te)
rk = lambda v: rankdata(v) / len(v)
ss = pd.read_csv(H.DATA + 'sample_submission.csv')
ch = pd.read_csv(CH)
assert (ch['ID'].to_numpy() == ss['ID'].to_numpy()).all()


def vec(tok):
    if tok == 'ch':
        return rk(ch['Popularity_Type'].to_numpy())
    if tok in ('v2', 'f8'):
        return rk(np.mean([rk(H.load_res(m)['test_full']) for m in (V2 if tok == 'v2' else F8)], 0))
    t = H.load_res(tok)['test_full']
    assert len(t) == n and np.isfinite(t).all(), tok
    return rk(t)


name, spec = sys.argv[1], sys.argv[2]
toks = [(t.split(':')[0], float(t.split(':')[1])) for t in spec.split(',')]
w = np.array([x for _, x in toks]); w = w / w.sum()
p = sum(wi * vec(t) for (t, _), wi in zip(toks, w))
sub = pd.DataFrame({'ID': ss['ID'].to_numpy(), 'Popularity_Type': p})
assert len(sub) == n and sub['Popularity_Type'].between(0, 1).all()
sub.to_csv(OUT + name + '.csv', index=False)
print(f'wrote {OUT}{name}.csv  weights {dict((t, round(float(x), 3)) for (t, _), x in zip(toks, w))}  '
      f'spearman vs ch {spearmanr(p, ch["Popularity_Type"]).correlation:.4f}')

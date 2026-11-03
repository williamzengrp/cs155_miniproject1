"""Segment-specific blending by labelled date-mate count (agent `shift`), on saved member OOFs.
python -I seg.py SET   (std | c9)
Segments of eligible val rows by cnt_date (labelled date-mates in the fit part; at test time: all train labels):
  s0: 0, s1: 1-3, s2: 4+. For each segment choose one of a few discrete member-weight vectors; the blend score of
a row = its segment's weighted mean of global member ranks. Prints all 27 combos' paired gains vs final8
(IN-SAMPLE choice -> upper bound) so std/c9 consistency can be checked."""
import sys, pickle, itertools
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
W = './work'
sys.path.insert(0, W); sys.path.insert(0, W + '/main')
import harness as H
from eval_candidate import MEMBERS
S = sys.argv[1]
C = W + '/r2_shift/cache/'
tr, te, y = H.load()
n = len(y)
lf = pickle.load(open(C + ('lf_1000_5.pkl' if S == 'std' else 'lf_9000_3.pkl'), 'rb'))
names = MEMBERS[S]
res = [H.load_res(nm) for nm in names]
R = res[0]['oof'].shape[0]
cnt_val = np.full((R, n), np.nan)
for f in lf:
    cnt_val[f['r'], f['val_idx'][f['elig']]] = f['F_val']['cnt_date'].to_numpy()[f['elig']]
lab = np.array([1, 1, 1, 1, 1, 0, 0, 0], float)       # members with label features
VEC = {'eq': np.ones(8), 'lab2': 1 + lab, 'free2': 2 - lab}
segs = lambda c: np.digitize(c, [0.5, 3.5])
base, combos = [], {}
for r in range(R):
    m = ~np.isnan(res[0]['oof'][r])
    O = np.column_stack([rankdata(x['oof'][r, m]) / m.sum() for x in res])
    sg = segs(cnt_val[r, m])
    yr = y[m]
    base.append(roc_auc_score(yr, O.mean(1)))
    for combo in itertools.product(VEC, repeat=3):
        p = np.zeros(m.sum())
        for s_, v in enumerate(combo):
            w = VEC[v] / VEC[v].sum()
            p[sg == s_] = O[sg == s_] @ w
        combos.setdefault(combo, []).append(roc_auc_score(yr, p))
base = np.array(base)
out = sorted(((np.array(v) - base).mean(), k, np.array(v) - base) for k, v in combos.items())[::-1]
print(f'[{S}] final8 {base.mean():.5f}; segment shares', np.round(np.bincount(sg) / len(sg), 3))
for g, k, d in out[:8] + out[-3:]:
    print(f'  {"/".join(k):18s} gain {g:+.5f} pos {(d > 0).sum()}/{R} per-rep {np.round(d, 4)}')

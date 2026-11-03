"""Assemble final test predictions: equal-weight rank average of saved models' test predictions.
python -I assemble.py OUT_NAME [full|foldavg] model1 model2 ..."""
import sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, './work')
import harness as H
OUT = './submissions/'
name, which, models = sys.argv[1], sys.argv[2], sys.argv[3:]
tr, te, y = H.load()
cols = []
for mname in models:
    r = H.load_res(mname)
    t = r['test_full'] if which == 'full' else r['test_foldavg']
    assert len(t) == len(te) and np.isfinite(t).all(), mname
    cols.append(rankdata(t) / len(t))
M = np.column_stack(cols)
print('pairwise rank corr of components (min/mean):', np.round(np.corrcoef(M.T)[np.triu_indices(len(models), 1)].min(), 3),
      np.round(np.corrcoef(M.T)[np.triu_indices(len(models), 1)].mean(), 3))
p = M.mean(axis=1)
sub = pd.DataFrame({'ID': te['ID'].to_numpy(), 'Popularity_Type': p})
ss = pd.read_csv(H.DATA + 'sample_submission.csv')
assert list(sub.columns) == list(ss.columns) and (sub['ID'].to_numpy() == ss['ID'].to_numpy()).all()
sub.to_csv(OUT + name + '.csv', index=False)
p6 = pd.read_csv('./cs155_miniproject1/predictions/prediction_6.csv')
from scipy.stats import spearmanr
print(f'wrote {OUT}{name}.csv  rows={len(sub)}  spearman vs teammates prediction_6 = {spearmanr(p, p6["Popularity_Type"]).correlation:.4f}')
dup = te['uri'].map(te['uri'].value_counts()) > 1
print('mean pct-rank of the 22 within-test duplicate rows:', np.round(rankdata(p)[dup.to_numpy()].mean() / len(p), 3))

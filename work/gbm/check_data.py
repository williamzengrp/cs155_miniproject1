import sys, time
sys.path.insert(0, './work/gbm')
import numpy as np, pandas as pd
import models as M
t = time.time()
tr, te, y, X, Xt = M.load_data()
print('features', X.shape, Xt.shape, f'{time.time()-t:.1f}s')
uri = tr['uri'].to_numpy()
c = M.uri_counts(uri)
print('copies dist', pd.Series(c).value_counts().to_dict())
# label consistency within uri
g = pd.DataFrame({'u': uri, 'y': y}).groupby('u')['y'].agg(['mean', 'size'])
g = g[g['size'] > 1]
print('multi-copy songs', len(g), 'with inconsistent labels', ((g['mean'] > 0) & (g['mean'] < 1)).sum())
print('High rate singletons', y[c == 1].mean().round(3), 'multi', y[c > 1].mean().round(3))
print('NaN counts', X.isna().sum()[X.isna().sum() > 0].to_dict())
# eligible counts for standard splits
import harness as H
sp = H.make_splits(tr, y, n_rep=5)
for r in range(5):
    n_el = sum(s[3].sum() for s in sp if s[0] == r)
    n_dup_el = sum((c[s[2][s[3]]] > 1).sum() for s in sp if s[0] == r)
    print('rep', r, 'eligible', n_el, 'of which multi-copy rows', n_dup_el)

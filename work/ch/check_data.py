"""ch agent: data checks for charlie-branch ideas (twins, reference submission comparison). Label-free except
for the twin label-agreement count."""
import sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, './work')
import harness as H
from time_features import AUDIO

tr, te, y = H.load()
al = pd.concat([tr, te], ignore_index=True)
n = len(tr)
akey = al[AUDIO].round(8).astype(str).agg('|'.join, axis=1)
uri = al['uri'].to_numpy()
# twins = identical full audio vector, different uri
g = pd.DataFrame({'a': akey, 'u': uri, 'is_tr': np.arange(len(al)) < n, 'date': al['track_album_release_date']})
nu = g.groupby('a')['u'].transform('nunique')
tw = nu > 1
print('pool rows in a twin group (identical audio, >1 uri):', int(tw.sum()), ' train', int(tw[:n].sum()), ' test', int(tw[n:].sum()))
# test rows with a twin in train
tr_keys = set(akey[:n]); tr_uris = set(uri[:n])
te_tw = [(akey[i] in tr_keys) for i in range(n, len(al))]
print('test rows with exact-audio twin in train:', int(np.sum(te_tw)), '/', len(te))
same_date_tw = 0
for i in range(n, len(al)):
    if akey[i] in tr_keys:
        m = (akey[:n] == akey[i])
        same_date_tw += int((g['date'][:n][m] == g['date'][i]).any())
print('   of which twin has the same release date:', same_date_tw)
# in CV: eligible val rows with a twin (different uri) in fit
splits = H.make_splits(tr, y, n_rep=5)
cnt = elig_tot = 0
for r, fi, vi, el in splits:
    fk = set(akey[fi])
    v = vi[el]
    elig_tot += len(v)
    cnt += sum(akey[j] in fk for j in v)
print(f'CV: eligible val rows with exact-audio twin in fit: {cnt}/{elig_tot} = {cnt/elig_tot:.4f} (test {np.sum(te_tw)/len(te):.4f})')
# twin label agreement in train (different uri)
trg = g.iloc[:n].copy(); trg['y'] = y
tw_tr = trg[trg.groupby('a')['u'].transform('nunique') > 1]
print('train twin groups:', tw_tr['a'].nunique(), ' rows', len(tw_tr), ' mean y', tw_tr['y'].mean().round(3))

# charlie's reference submission (1 - P(High)) vs main prediction_6 (LB 0.861) -- label-free comparison
ref = pd.read_csv(sys.argv[1]); p6 = pd.read_csv(sys.argv[2])
assert (ref['ID'].to_numpy() == te['ID'].to_numpy()).all() and (p6['ID'].to_numpy() == te['ID'].to_numpy()).all()
print('charlie ref columns', list(ref.columns), ' value range', ref.iloc[:, 1].min().round(4), ref.iloc[:, 1].max().round(4), ' mean', ref.iloc[:, 1].mean().round(4))
print('Spearman(charlie ref, main prediction_6) =', round(spearmanr(ref.iloc[:, 1], p6.iloc[:, 1])[0], 4),
      ' -> Spearman(1-ref, p6) =', round(spearmanr(1 - ref.iloc[:, 1], p6.iloc[:, 1])[0], 4))
# sanity: does P(High) of charlie decrease with release year like the known trend?
yr = pd.to_datetime(te['track_album_release_date'], format='mixed', errors='coerce').dt.year
print('corr(1-ref, year) =', round(spearmanr(1 - ref.iloc[:, 1], yr, nan_policy='omit')[0], 3),
      ' corr(p6, year) =', round(spearmanr(p6.iloc[:, 1], yr, nan_policy='omit')[0], 3))

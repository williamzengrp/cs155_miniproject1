"""r2_eda: per-member AUC by segment (std splits, per-repeat pooled within segment, averaged over repeats)."""
import sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
WORK = './work'
sys.path.insert(0, WORK)
import harness as H

MEM = ['fkd_lgbm', 'fkt_xgb', 'fkt_cat', 'fkd_et', 'fkd_rf', 'mdl_mlp_ple', 'ch_et_proj', 'ch_lgbm_proj']
tr, te, y = H.load()
D = pd.read_pickle(WORK + '/r2_eda/blend_std.pkl')
E2 = pd.read_pickle(WORK + '/r2_eda/blend_std.pkl')
res = {m: H.load_res(m)['oof'] for m in MEM}
R = 5
# LOO date-mate count (diagnostic)
al = pd.concat([tr, te], ignore_index=True)
raw = al['track_album_release_date'].astype(str)
lab = pd.Series(np.r_[np.ones(len(tr)), np.zeros(len(te))])
c_ = raw.map(lab.groupby(raw).sum()) - al['uri'].map(lab.groupby(al['uri']).sum())
segs = {
    'all': np.ones(len(tr), bool),
    'no lab mates': (c_[:len(tr)] == 0).to_numpy(),
    '1-3 lab mates': c_[:len(tr)].between(1, 3).to_numpy(),
    '4+ lab mates': (c_[:len(tr)] >= 4).to_numpy(),
    'instr>0.5': (tr.instrumentalness > 0.5).to_numpy(),
    'year<2010': (D.year < 2010).to_numpy(),
    '2010-2019': D.year.between(2010, 2019).to_numpy(),
    '2020+': (D.year >= 2020).to_numpy(),
}
rows = []
for sname, smask in segs.items():
    row = {'seg': sname}
    for mname in MEM + ['blend']:
        a = []
        for r in range(R):
            m = ~np.isnan(res[MEM[0]][r])
            if mname == 'blend':
                sc = np.column_stack([rankdata(res[x][r, m]) for x in MEM]).mean(1)
            else:
                sc = res[mname][r, m]
            mm = smask[m]
            a.append(roc_auc_score(y[m][mm], sc[mm]))
        row[mname] = np.mean(a)
    row['n/rep'] = int(smask[~np.isnan(res[MEM[0]][0])].sum())
    rows.append(row)
pd.set_option('display.width', 250)
print(pd.DataFrame(rows).round(4).to_string(index=False))

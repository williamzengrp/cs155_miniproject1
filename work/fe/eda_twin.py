import sys; sys.path.insert(0, './work/fe')
sys.path.insert(0, './work')
import numpy as np, pandas as pd
import harness as H
from fe_lib import *
tr, te, y = H.load(); n = len(tr)
al, raw, d, akey = make_pool(tr, te)
yy = np.r_[y, np.full(len(te), np.nan)]
ucnt = al['uri'].map(al['uri'].value_counts()).to_numpy()
first, codes = dedup_index(al, akey, raw, 'uri')
u = al.iloc[first].reset_index(drop=True)
uy = yy[first]; usrc = np.r_[np.zeros(n), np.ones(len(te))][first]; uc = ucnt[first]
dur = u.duration_ms.to_numpy(); key = u.key.to_numpy(); mode = u['mode'].to_numpy(); tempo = u.tempo.to_numpy()
en = u.energy.to_numpy(); da = u.danceability.to_numpy()
near = np.zeros(len(u), int); same_date_near = np.zeros(len(u), int)
ru = raw.iloc[first].to_numpy()
for i in range(len(u)):
    m = (np.abs(dur - dur[i]) < 1500) & (np.abs(tempo - tempo[i]) < 1) & (np.abs(en - en[i]) < 0.04) & (np.abs(da - da[i]) < 0.04) & (key == key[i]) & (mode == mode[i])
    m[i] = False
    near[i] = m.sum(); same_date_near[i] = (m & (ru == ru[i])).sum()
df = pd.DataFrame({'near': near, 'sd_near': same_date_near, 'y': uy, 'src': usrc, 'cnt': uc})
print('songs with a near-twin (diff uri):', (near > 0).sum(), ' of which same date:', (same_date_near > 0).sum())
s = df[(df.src == 0) & (df.cnt == 1)]
print(s.groupby(np.minimum(s.near, 2))['y'].agg(['mean', 'size']))
print('other-date near twins only:')
s2 = s.assign(od=s.near - s.sd_near)
print(s2.groupby(np.minimum(s2.od, 2))['y'].agg(['mean', 'size']))
print('test songs with near twin:', ((df.src == 1) & (df.near > 0)).sum())

import sys; sys.path.insert(0, './work')
import numpy as np, pandas as pd
import harness as H
from time_features import AUDIO, parse_date
tr, te, y = H.load()
al = pd.concat([tr.assign(src='tr', y=y), te.assign(src='te')], ignore_index=True)
n = len(tr)
raw = al['track_album_release_date'].astype(str)
d = parse_date(raw)
al['ucnt'] = al['uri'].map(al['uri'].value_counts())
g = al.groupby(raw)
st = pd.DataFrame({'rows': g.size(), 'songs': g['uri'].nunique(), 'n_tr': g['src'].apply(lambda s: (s == 'tr').sum()),
                   'n_te': g['src'].apply(lambda s: (s == 'te').sum()), 'yrate': g['y'].mean(),
                   'dow': d.dt.day_of_week.groupby(raw).first(),
                   'loud_std': g['loudness'].std(), 'dur_std': g['duration_ms'].std()})
print(st.sort_values('songs', ascending=False).head(40).round(2).to_string())
# how pure are date groups (label rate distribution) for groups with >=5 train singletons
s1 = al[:n][al[:n].ucnt == 1]
gg = s1.groupby(raw[:n][al[:n].ucnt == 1])['y'].agg(['mean', 'size'])
gg = gg[gg['size'] >= 5]
print('groups with >=5 train singletons:', len(gg), ' rows:', gg['size'].sum())
print(np.histogram(gg['mean'], bins=[0, .01, .2, .4, .6, .8, .99, 1.0])[0])
# exact YYYY-01-01 dates
print('YYYY-01-01 full dates:', (raw.str.len().eq(10) & raw.str.endswith('-01-01')).sum(), ' year-only:', raw.str.len().eq(4).sum())

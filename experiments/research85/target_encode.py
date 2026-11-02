"""Leakage-safe, track-group-cross-fitted categorical release-date encodings."""
from collections import defaultdict
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from features85 import InteractionFeatureBuilder


def date_keys(rows):
    values=[]
    for row in rows:
        s=(row.get('track_album_release_date') or '').strip()
        year=s[:4] if len(s)>=4 and s[:4].isdigit() else None
        month=s[:7] if len(s)>=7 and s[4]=='-' and s[5:7].isdigit() else None
        exact=s if len(s)==10 and s[4]=='-' and s[7]=='-' else None
        values.append((year,month,exact))
    return values


class GroupedDateTargetEncoder:
    def __init__(self,smoothing=10.,folds=5,seed=42):
        if smoothing<=0 or folds<2:raise ValueError('Invalid encoder configuration')
        self.smoothing=smoothing;self.folds=folds;self.seed=seed
    def _build(self,keys,y):
        prior=float(np.mean(y));maps=[defaultdict(lambda:[0.,0]) for _ in range(3)]
        for k,label in zip(keys,y):
            for j,key in enumerate(k):
                if key is not None:
                    bucket=maps[j][key];bucket[0]+=int(label);bucket[1]+=1
        return prior,[dict(m) for m in maps]
    def _encode(self,keys,state):
        prior,maps=state;out=np.empty((len(keys),6),dtype=float)
        for i,k in enumerate(keys):
            for j,key in enumerate(k):
                total,count=maps[j].get(key,(0.,0)) if key is not None else (0.,0)
                out[i,j]=(total+self.smoothing*prior)/(count+self.smoothing)
                out[i,j+3]=np.log1p(count)
        return out
    def fit_transform(self,rows,y,groups):
        y=np.asarray(y);groups=np.asarray(groups);keys=date_keys(rows)
        if len(y)!=len(rows) or len(groups)!=len(rows):raise ValueError('Misaligned data')
        out=np.full((len(rows),6),np.nan)
        splitter=StratifiedGroupKFold(n_splits=self.folds,shuffle=True,random_state=self.seed)
        for train,val in splitter.split(np.arange(len(rows)),y,groups):
            assert not set(groups[train])&set(groups[val])
            state=self._build([keys[i] for i in train],y[train])
            out[val]=self._encode([keys[i] for i in val],state)
        if not np.isfinite(out).all():raise ValueError('Missing cross-fitted encodings')
        self.state=self._build(keys,y)
        return out
    def transform(self,rows):return self._encode(date_keys(rows),self.state)


class TargetDateBuilder:
    def __init__(self,smoothing=10.,interactions=False):
        self.smoothing=smoothing;self.interactions=interactions
        self.base=InteractionFeatureBuilder(interactions=interactions)
        self.encoder=GroupedDateTargetEncoder(smoothing=smoothing)
    def fit_transform(self,rows,y,groups):
        self.base.fit(rows);encoded=self.encoder.fit_transform(rows,y,groups)
        return np.column_stack((self.base.transform(rows),encoded))
    def transform(self,rows):return np.column_stack((self.base.transform(rows),self.encoder.transform(rows)))

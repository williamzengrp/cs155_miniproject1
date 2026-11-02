"""Optional row-local audio/calendar interactions; no IDs or target features."""
import sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'research80'))
from features import FeatureBuilderV2

class InteractionFeatureBuilder(FeatureBuilderV2):
    def __init__(self,dates='calendar',onehot=False,interactions=False):
        super().__init__(dates,onehot)
        self.interactions=interactions
    def raw(self,rows):
        data=super().raw(rows)
        if not self.interactions:return data
        def v(name):return data[name].to_numpy(dtype=float)
        def product(a,b):return v(a)*v(b)
        for name in ['instrumentalness','speechiness','acousticness','liveness','duration_ms']:
            data[f'log1p_{name}']=np.log1p(np.maximum(v(name),0)*1000 if name!='duration_ms' else np.maximum(v(name),0))
        for left,right in [('energy','danceability'),('energy','valence'),('energy','acousticness'),
                           ('energy','loudness'),('danceability','valence'),('danceability','speechiness'),
                           ('instrumentalness','acousticness'),('instrumentalness','speechiness'),
                           ('valence','acousticness'),('liveness','speechiness')]:
            data[f'{left}_times_{right}']=product(left,right)
        key=v('key');data['key_sin']=np.sin(2*np.pi*key/12);data['key_cos']=np.cos(2*np.pi*key/12)
        day=v('release_day_of_year');data['release_season_sin']=np.sin(2*np.pi*day/365.25);data['release_season_cos']=np.cos(2*np.pi*day/365.25)
        year=v('release_year')-2000
        for name in ['energy','valence','speechiness','acousticness','instrumentalness']:
            data[f'year_times_{name}']=year*v(name)
        return data

class DateCategoricalBuilder:
    """Numeric audio/date features plus explicit categorical release and music fields."""
    def __init__(self,date_categories='full',music_categories=False,interactions=False):
        if date_categories not in ('none','year_month','full'):raise ValueError('Invalid date categories')
        self.date_categories=date_categories;self.music_categories=music_categories;self.interactions=interactions
        self.numeric=InteractionFeatureBuilder(interactions=interactions)
    def fit(self,rows):
        self.numeric.fit(rows)
        return self
    @property
    def cat_columns(self):
        names=[]
        if self.date_categories in ('year_month','full'):names+=['release_year_category','release_year_month_category']
        if self.date_categories=='full':names+=['release_exact_category']
        if self.music_categories:names+=['key_category','mode_category','time_signature_category']
        return names
    def transform(self,rows):
        import pandas as pd
        raw=pd.DataFrame(rows)
        result=pd.DataFrame(self.numeric.transform(rows),columns=self.numeric.feature_names)
        date=raw['track_album_release_date'].fillna('').astype(str)
        if self.date_categories in ('year_month','full'):
            result['release_year_category']=date.str.slice(0,4).where(date.str.len()>=4,'__missing__')
            result['release_year_month_category']=date.str.slice(0,7).where(date.str.len()>=7,'__missing__')
        if self.date_categories=='full':result['release_exact_category']=date.where(date.str.len()>=10,'__missing__')
        if self.music_categories:
            for name in ('key','mode','time_signature'):
                result[f'{name}_category']=raw[name].fillna('').astype(str).replace('','__missing__')
        return result

"""Reusable training-fitted feature engineering for the Spotify data."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from preprocessing import FEATURES

class FeatureBuilder:
    """Use numeric audio attributes and optional release-date/calendar features.

    Every feature is available from one input row. No target or identifier is used.
    Missing-value statistics are fitted on the training partition only.
    """
    def __init__(self, dates='year', onehot=True):
        if dates not in ('none','year','calendar'):raise ValueError('Unknown date feature set')
        self.dates=dates;self.onehot=onehot
        self.imputer=SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)
    def raw(self,rows):
        frame=pd.DataFrame(rows)
        base=frame[FEATURES].apply(pd.to_numeric,errors='coerce').astype(float)
        base=base.replace([np.inf,-np.inf],np.nan)
        data={c:base[c].to_numpy() for c in FEATURES}
        if self.onehot:
            for col,values in [('key',range(12)),('mode',range(2)),('time_signature',range(1,8))]:
                for value in values:data[f'{col}_is_{value}']=(base[col].to_numpy()==value).astype(float)
        if self.dates!='none':
            dates=pd.to_datetime(frame['track_album_release_date'],format='mixed',errors='coerce')
            data['release_year']=dates.dt.year.to_numpy(dtype=float,na_value=np.nan)
            if self.dates=='calendar':
                data['release_month']=dates.dt.month.to_numpy(dtype=float,na_value=np.nan)
                data['release_day']=dates.dt.day.to_numpy(dtype=float,na_value=np.nan)
                data['release_weekday']=dates.dt.day_of_week.to_numpy(dtype=float,na_value=np.nan)
                data['release_day_of_year']=dates.dt.day_of_year.to_numpy(dtype=float,na_value=np.nan)
                data['release_date_ordinal']=np.where(dates.notna(),dates.astype('int64')/86400000000000,np.nan)
                data['release_precision']=frame['track_album_release_date'].fillna('').str.len().to_numpy(dtype=float)
        return pd.DataFrame(data)
    def fit(self,rows):
        data=self.raw(rows)
        self.imputer.fit(data)
        self.input_feature_names=list(data.columns)
        self.feature_names=list(self.imputer.get_feature_names_out(self.input_feature_names))
        return self
    def transform(self,rows):
        data=self.raw(rows)
        if list(data.columns)!=self.input_feature_names:raise ValueError('Feature layout differs')
        return self.imputer.transform(data)

class FeatureBuilderV2(FeatureBuilder):
    """Calendar features that respect date precision and use actual day ordinals."""
    def raw(self,rows):
        data=super().raw(rows)
        if self.dates!='calendar':return data
        frame=pd.DataFrame(rows)
        strings=frame['track_album_release_date'].fillna('').astype(str)
        dates=pd.to_datetime(strings,format='mixed',errors='coerce')
        month_known=strings.str.fullmatch(r'\d{4}-\d{2}(-\d{2})?').fillna(False)
        day_known=strings.str.fullmatch(r'\d{4}-\d{2}-\d{2}').fillna(False)
        data['release_month']=dates.dt.month.where(month_known).to_numpy(dtype=float,na_value=np.nan)
        data['release_day']=dates.dt.day.where(day_known).to_numpy(dtype=float,na_value=np.nan)
        data['release_weekday']=dates.dt.day_of_week.where(day_known).to_numpy(dtype=float,na_value=np.nan)
        data['release_day_of_year']=dates.dt.day_of_year.where(day_known).to_numpy(dtype=float,na_value=np.nan)
        ordinal=dates.to_numpy(dtype='datetime64[D]').astype('int64').astype(float)
        data['release_date_ordinal']=np.where(day_known & dates.notna(),ordinal,np.nan)
        return data

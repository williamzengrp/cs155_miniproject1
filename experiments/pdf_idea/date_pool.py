"""Unsupervised, transductive release density and same-day audio similarities.

Input pool is original train plus the supplied unlabelled test, as in the PDF.
No label or resource identifier is used in these generated feature values.
"""
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from preprocessing import FEATURES

def build_pool(rows):
 df=pd.DataFrame(rows);raw=df['track_album_release_date'].fillna('').astype(str).str.strip();date=pd.to_datetime(raw,format='mixed',errors='coerce');audio=df[FEATURES].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
 n=len(df);out={k:audio[k].to_numpy(dtype=float) for k in FEATURES};year=date.dt.year.to_numpy(dtype=float,na_value=np.nan);month=date.dt.month.to_numpy(dtype=float,na_value=np.nan);day=date.dt.day.to_numpy(dtype=float,na_value=np.nan);weekday=date.dt.dayofweek.to_numpy(dtype=float,na_value=np.nan);doy=date.dt.dayofyear.to_numpy(dtype=float,na_value=np.nan)
 known_day=raw.str.fullmatch(r'\d{4}-\d{2}-\d{2}').fillna(False).to_numpy();known_month=raw.str.fullmatch(r'\d{4}-\d{2}(-\d{2})?').fillna(False).to_numpy();month=np.where(known_month,month,np.nan);day=np.where(known_day,day,np.nan);weekday=np.where(known_day,weekday,np.nan);doy=np.where(known_day,doy,np.nan)
 out.update(release_year=year,release_month=month,release_day=day,weekday=weekday,day_of_year=doy,release_quarter=np.ceil(month/3),fractional_year=year+(doy-1)/365.25,decade=np.floor(year/10)*10,is_jan1=raw.str.endswith('-01-01').to_numpy(float),is_friday=(weekday==4).astype(float),is_tuesday=(weekday==1).astype(float),is_summer=np.isin(month,[6,7,8]).astype(float),is_holiday=np.isin(month,[11,12]).astype(float))
 for name,v,period in [('month',month,12),('weekday',weekday,7),('day_of_year',doy,365.25)]:out[name+'_sin']=np.sin(2*np.pi*v/period);out[name+'_cos']=np.cos(2*np.pi*v/period)
 basic=pd.DataFrame(out)
 # Keys only group rows; these do not contain labels. Blank/partial dates get
 # missing values for unavailable precision rather than a shared fake date.
 ykey=pd.Series(np.where(np.isfinite(year),year,-1));mkey=pd.Series(np.where(np.isfinite(month),year*100+month,-1));dkey=raw.where(known_day,'')
 density=pd.DataFrame({'same_year_count':ykey.map(ykey[ykey!=-1].value_counts()).fillna(0),'same_month_count':mkey.map(mkey[mkey!=-1].value_counts()).fillna(0),'same_date_count':dkey.map(dkey[dkey!=''].value_counts()).fillna(0)})
 tempo=pd.to_numeric(df['tempo'],errors='coerce').round();same_tempo=pd.Series(list(zip(dkey,tempo)));counts=same_tempo.value_counts();density['same_date_tempo_count']=same_tempo.map(counts).where(dkey!='',0).fillna(0)
 # Standardisation and neighbor pools are explicitly transductive and label-free.
 scaler=StandardScaler();z=scaler.fit_transform(SimpleImputer(strategy='median',keep_empty_features=True).fit_transform(audio));near=np.zeros((n,5));near[:,3:]=np.nan;means=np.full((n,len(FEATURES)),np.nan)
 for key,index in dkey[dkey!=''].groupby(dkey[dkey!='']).groups.items():
  ix=np.asarray(list(index));means[ix]=audio.iloc[ix].mean().to_numpy();
  if len(ix)<2:continue
  d=pairwise_distances(z[ix]);np.fill_diagonal(d,np.inf);near[ix,:3]=np.column_stack([(d<r).sum(axis=1) for r in (1.,1.5,2.)]);sort=np.sort(d,axis=1);near[ix,3]=sort[:,0];near[ix,4]=np.where(np.isfinite(sort[:,1]),sort[:,1],np.nan)
 for j,name in enumerate(('near_count_1','near_count_1_5','near_count_2','nearest_same_date','second_nearest_same_date')):density[name]=near[:,j]
 for j,name in enumerate(FEATURES):density['same_date_mean_'+name]=means[:,j];density['same_date_deviation_'+name]=audio[name].to_numpy()-means[:,j]
 out=pd.concat([basic,density],axis=1).replace([np.inf,-np.inf],np.nan);return out,basic.columns.tolist()

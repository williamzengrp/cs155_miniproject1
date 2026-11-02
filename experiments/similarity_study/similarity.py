"""Label-free features for conservative album-like audio neighborhoods."""
from pathlib import Path
import sys
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'experiments/pdf_idea')]
from preprocessing import FEATURES
from date_pool import build_pool

def features(rows):
 base,_=build_pool(rows);df=pd.DataFrame(rows);audio=df[FEATURES].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan);valid=audio.notna().any(axis=1).to_numpy();dates=df['track_album_release_date'].fillna('').astype(str).str.strip();full=dates.str.fullmatch(r'\d{4}-\d{2}-\d{2}').to_numpy();ids=df['uri'].fillna('').astype(str).to_numpy();n=len(rows)
 z=StandardScaler().fit_transform(SimpleImputer(strategy='median',keep_empty_features=True).fit_transform(audio));z=pd.DataFrame(z,columns=FEATURES);zlog=z.copy();loginst=np.log10(np.maximum(audio['instrumentalness'].to_numpy(dtype=float),1e-5));loginst=np.where(np.isfinite(loginst),loginst,np.nan);loginst=SimpleImputer(strategy='median').fit_transform(loginst[:,None]);zlog['instrumentalness']=StandardScaler().fit_transform(loginst)[:,0]
 # Never treat a completely missing audio row as a synthetic median-sounding
 # neighbor. Keep its label and calendar features and mark its distances missing.
 for i in np.flatnonzero(~valid):
  same=np.flatnonzero((dates.to_numpy()==dates.iloc[i]) & valid & full)
  base.loc[i,['nearest_same_date','second_nearest_same_date','near_count_1','near_count_1_5','near_count_2']]=np.nan
  if len(same):
   d=pairwise_distances(z.iloc[same]);np.fill_diagonal(d,np.inf);ordered=np.sort(d,axis=1)
   base.loc[same,'nearest_same_date']=np.where(np.isfinite(ordered[:,0]),ordered[:,0],np.nan)
   base.loc[same,'second_nearest_same_date']=np.where(np.isfinite(ordered[:,1]),ordered[:,1],np.nan) if len(same)>1 else np.nan
   for radius,column in [(1.,'near_count_1'),(1.5,'near_count_1_5'),(2.,'near_count_2')]:base.loc[same,column]=(d<radius).sum(axis=1)
 specs={'production':['energy','loudness','acousticness','instrumentalness'],'rhythm':['tempo','danceability','speechiness'],'all_no_key':[c for c in FEATURES if c not in ('key','mode','time_signature')]};out={k:{} for k in ('subset','cluster','position','crossdate','tempo','interactions','secondary')}
 for tag,cols in specs.items():
  zsub=zlog[cols].to_numpy();dist=np.full((n,3),np.nan);counts=np.zeros((n,3));
  for _,ixs in dates[full & valid].groupby(dates[full & valid]).groups.items():
   ix=np.asarray(list(ixs));
   if len(ix)<2:continue
   d=pairwise_distances(zsub[ix]);np.fill_diagonal(d,np.inf);ds=np.sort(d,axis=1);dist[ix,0]=ds[:,0];dist[ix,1]=np.where(np.isfinite(ds[:,1]),ds[:,1],np.nan);dist[ix,2]=np.nanmean(np.where(np.isfinite(d),d,np.nan),axis=1)
   for j,r in enumerate((1.,2.,3.)):counts[ix,j]=(d<r).sum(axis=1)
  for j,label in enumerate(('nearest','second','mean')):out['subset'][f'{tag}_{label}']=dist[:,j]
  for j,r in enumerate((1,2,3)):out['subset'][f'{tag}_within_{r}']=np.where(valid,counts[:,j],np.nan)
 prod=zlog[specs['production']].to_numpy()
 for eps in (1.,2.):
  size=np.full(n,np.nan);center=np.full(n,np.nan);spread=np.full(n,np.nan);num=np.full(n,np.nan);labels_all=np.full(n,-1,dtype=int)
  for _,ixs in dates[full & valid].groupby(dates[full & valid]).groups.items():
   ix=np.asarray(list(ixs));lab=DBSCAN(eps=eps,min_samples=2).fit_predict(prod[ix]);nextid=int(lab.max()+1)
   for j in np.flatnonzero(lab<0):lab[j]=nextid;nextid+=1
   num[ix]=len(np.unique(lab));
   for label in np.unique(lab):
    inds=ix[lab==label];distance=np.linalg.norm(prod[inds]-np.mean(prod[inds],axis=0),axis=1);size[inds]=len(inds);center[inds]=distance;spread[inds]=float(np.mean(distance))
  out['cluster'][f'cluster_size_{eps}']=size;out['cluster'][f'cluster_center_dist_{eps}']=center;out['cluster'][f'cluster_spread_{eps}']=spread;out['cluster'][f'date_cluster_count_{eps}']=num
  if eps==1.:
   year=pd.to_datetime(dates,errors='coerce',format='mixed').dt.year.to_numpy(dtype=float,na_value=np.nan);out['cluster']['cluster_size_x_year']=(year-2000)*size
 for field in ('duration_ms','energy','loudness','speechiness'):
  values=audio[field].where(full & valid);gb=values.groupby(dates.where(full & valid));out['position'][f'date_percentile_{field}']=gb.rank(pct=True).to_numpy();out['position'][f'date_median_delta_{field}']=(values-gb.transform('median')).to_numpy()
 # Compute distances in blocks; cross-date matches exclude same URI and missing audio.
 cross=np.full((n,3),np.nan);count=np.zeros(n);ordinal=pd.to_datetime(dates,errors='coerce',format='mixed').to_numpy(dtype='datetime64[D]').astype('int64');validdate=full & (ordinal!=np.iinfo(np.int64).min)
 for start in range(0,n,128):
  idx=np.arange(start,min(start+128,n));d=pairwise_distances(prod[idx],prod);mask=(dates.to_numpy()[idx,None]!=dates.to_numpy()[None,:]) & (ids[idx,None]!=ids[None,:]) & valid[idx,None] & valid[None,:] & validdate[idx,None] & validdate[None,:];d[~mask]=np.inf;arg=np.argmin(d,axis=1);best=d[np.arange(len(idx)),arg];good=np.isfinite(best);cross[idx[good],0]=best[good];cross[idx[good],1]=np.abs(ordinal[idx[good]]-ordinal[arg[good]]);count[idx]=(d<1.).sum(axis=1)
  for j,i in enumerate(idx):
   hit=np.flatnonzero(d[j]<1.)
   if len(hit):cross[i,2]=float(np.min(ordinal[hit]))
 for j,name in enumerate(('nearest_other_date_prod','nearest_other_date_gap_days','near_other_date_earliest_ordinal')):out['crossdate'][name]=cross[:,j]
 out['crossdate']['near_other_date_count']=np.where(valid,count,np.nan)
 # Rhythm distance considers half / original / double tempo, independently of key.
 tempo=audio['tempo'].to_numpy(float);dr=zlog[['danceability','speechiness']].to_numpy();tempo_out=np.full(n,np.nan);tempo_count=np.zeros(n)
 for _,ixs in dates[full & valid].groupby(dates[full & valid]).groups.items():
  ix=np.asarray(list(ixs));
  if len(ix)<2:continue
  dt=np.abs(np.log2(np.maximum(tempo[ix,None],1)/np.maximum(tempo[None,ix],1)));dt=np.minimum.reduce([dt,np.abs(dt-1),np.abs(dt+1)]);d=np.sqrt(pairwise_distances(dr[ix])**2+(dt/.25)**2);np.fill_diagonal(d,np.inf);tempo_out[ix]=np.min(d,axis=1);tempo_count[ix]=(d<1.5).sum(axis=1)
 out['tempo']['tempo_ambiguous_nearest']=tempo_out;out['tempo']['tempo_ambiguous_count']=np.where(valid,tempo_count,np.nan)
 a={c:audio[c].to_numpy(float) for c in FEATURES};out['interactions']['estimated_beats']=a['duration_ms']*a['tempo']/60000.;out['interactions']['estimated_bars']=a['duration_ms']*a['tempo']/60000./np.where(a['time_signature']>0,a['time_signature'],np.nan)
 for name,val in {'energy_x_danceability':a['energy']*a['danceability'],'energy_x_low_valence':a['energy']*(1-a['valence']),'acoustic_x_instrumental':a['acousticness']*a['instrumentalness'],'weekday_x_era':base['weekday'].to_numpy()*(base['release_year'].to_numpy()-2000),'density_x_era':base['same_date_count'].to_numpy()*(base['release_year'].to_numpy()-2000)}.items():out['interactions'][name]=val
 good=np.isfinite(a['energy']) & np.isfinite(a['loudness']);slope,intercept=np.polyfit(a['energy'][good],a['loudness'][good],1);out['secondary']['loudness_energy_residual']=a['loudness']-(slope*a['energy']+intercept)
 out['secondary']['year_density_percentile']=base['same_date_count'].groupby(base['release_year']).rank(pct=True).to_numpy()
 for col,values in [('key',range(12)),('mode',range(2)),('time_signature',range(1,8))]:
  for value in values:out['secondary'][f'{col}_is_{value}']=np.where(np.isfinite(a[col]),(a[col]==value).astype(float),np.nan)
 for key in range(12):
  for mode in range(2):out['secondary'][f'key_{key}_mode_{mode}']=np.where(np.isfinite(a['key'])&np.isfinite(a['mode']),((a['key']==key)&(a['mode']==mode)).astype(float),np.nan)
 key_nearest=np.full(n,np.nan);key=a['key']
 for _,ixs in dates[full & valid & np.isfinite(key)].groupby(dates[full & valid & np.isfinite(key)]).groups.items():
  ix=np.asarray(list(ixs));
  if len(ix)<2:continue
  delta=np.abs(key[ix,None]-key[None,ix]);cycle=np.minimum(delta,12-delta)/3.;d=np.sqrt(pairwise_distances(prod[ix])**2+cycle**2);np.fill_diagonal(d,np.inf);key_nearest[ix]=np.min(d,axis=1)
 out['secondary']['nearest_same_date_cyclic_key']=key_nearest
 out={k:pd.DataFrame(v).replace([np.inf,-np.inf],np.nan) for k,v in out.items()};return base,out

#!/usr/bin/env python3
"""CPU-only, restartable Spotify CSV search. See README.md before running."""
import argparse
from collections import OrderedDict
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import signal
import sys
import time
import traceback

# Set before numeric imports; avoid nested thread oversubscription.
DEFAULT_THREADS = max(1, min(8, (os.cpu_count() or 2) - 1))
_thread_parser = argparse.ArgumentParser(add_help=False)
_thread_parser.add_argument('--threads',type=int,default=DEFAULT_THREADS)
_early_threads, _ = _thread_parser.parse_known_args()
os.environ.setdefault('OMP_NUM_THREADS', str(max(1,_early_threads.threads)))
os.environ.setdefault('OPENBLAS_NUM_THREADS', str(max(1,_early_threads.threads)))
import numpy as np
import pandas as pd
try:
    import optuna
except ImportError:
    optuna = None
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

AUDIO = ['danceability','energy','loudness','tempo','duration_ms','speechiness',
         'acousticness','instrumentalness','liveness','valence','key','mode','time_signature']
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PROJECT_ROOT/'experiments'),str(PROJECT_ROOT/'experiments/research85')]
from target_encode import GroupedDateTargetEncoder
STOP = False

def stop(*_):
    global STOP
    STOP = True
    print('\nStop requested: finishing the current fold, then exporting checkpoints.', flush=True)

signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)


def atomic_json(path, obj):
    tmp = Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def ids(x):
    s = x.get('uri', pd.Series(np.nan, index=x.index)).copy().astype('string').str.strip().replace('',pd.NA)
    for i in np.flatnonzero(s.isna()):
        s.iloc[i] = f'missing_row_{i}'
    return s.astype(str).to_numpy()


def duplicate_weights(frame, power):
    key=ids(frame.reset_index(drop=True))
    _,inverse,counts=np.unique(key,return_inverse=True,return_counts=True)
    w=counts[inverse].astype(float)**(-power)
    return w/w.mean()


def connected_groups(frame):
    """Join equal nonempty URIs and exactly equal complete audio vectors."""
    parent=np.arange(len(frame));seen={};audio=frame[AUDIO].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float,na_value=np.nan)
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    def join(a,b):
        a,b=root(a),root(b)
        if a!=b:parent[b]=a
    for i,uri in enumerate(ids(frame)):
        tokens=[('uri',uri)]
        if np.isfinite(audio[i]).all():tokens.append(('audio',tuple(audio[i])))
        for token in tokens:
            if token in seen:join(i,seen[token])
            else:seen[token]=i
    return np.array([root(i) for i in range(len(frame))])


def base_features(x):
    f = x[AUDIO].apply(pd.to_numeric, errors='coerce').copy()
    raw = x.track_album_release_date.fillna('').astype(str)
    d = pd.to_datetime(raw, format='mixed', errors='coerce')
    month_known = raw.str.fullmatch(r'\d{4}-\d{2}(-\d{2})?').fillna(False)
    day_known = raw.str.fullmatch(r'\d{4}-\d{2}-\d{2}').fillna(False)
    for key, val in [('year',d.dt.year),('month',d.dt.month),('day',d.dt.day),
                     ('weekday',d.dt.day_of_week),('dayofyear',d.dt.day_of_year)]:
        f[key] = val.where(month_known if key == 'month' else day_known) if key != 'year' else val
    f['date_ordinal'] = (d - pd.Timestamp('1970-01-01')).dt.days.where(day_known)
    f['date_precision'] = raw.str.len()
    f['missing_audio'] = f[AUDIO].isna().sum(axis=1)
    for col, period in [('month',12),('weekday',7),('dayofyear',365.25)]:
        f[col+'_sin'] = np.sin(2*np.pi*f[col]/period)
        f[col+'_cos'] = np.cos(2*np.pi*f[col]/period)
    f['key_sin'] = np.sin(2*np.pi*f['key']/12)
    f['key_cos'] = np.cos(2*np.pi*f['key']/12)
    return f


class Features:
    """Fold-local, label-free reference statistics; duplicate IDs count once."""
    def __init__(self, train, test, transductive=False, extra=None, replace_features=False):
        self.tr, self.te, self.transductive = train, test, transductive
        self.extra = extra
        self.replace_features = replace_features
        self.cache = OrderedDict()

    def get(self, fit, apply, view):
        key = (tuple(fit), tuple(apply), view)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        if self.replace_features:
            a,b = self.extra
            f = pd.concat([a.iloc[fit],a.iloc[apply],b],ignore_index=True).replace([np.inf,-np.inf],np.nan).astype(np.float32)
            n,v=len(fit),len(apply)
            value=(f.iloc[:n].copy(),f.iloc[n:n+v].copy(),f.iloc[n+v:].copy())
            self.cache[key]=value
            while len(self.cache)>12:self.cache.popitem(last=False)
            return value
        query = pd.concat([self.tr.iloc[fit], self.tr.iloc[apply], self.te], ignore_index=True)
        ref = pd.concat([self.tr, self.te], ignore_index=True) if self.transductive else self.tr.iloc[fit]
        ref = ref.drop_duplicates('uri').reset_index(drop=True)
        # Missing-ID records cannot establish an identity or an audio neighborhood.
        ref = ref[ref.uri.notna()].reset_index(drop=True)
        f = base_features(query)
        qids = query.uri.fillna('__missing__').to_numpy()
        rids = ref.uri.to_numpy()
        reference_by_id = ref.set_index('uri')
        reference_date = pd.Series(qids).map(reference_by_id['track_album_release_date']).fillna('').astype(str)
        query_date = query.track_album_release_date.fillna('').astype(str).reset_index(drop=True)
        if view != 'basic':
            for width in [10,7,4]:
                rd = ref.track_album_release_date.fillna('').astype(str)
                rk = rd.str[:width].where(rd.str.len() >= width)
                qk = query_date.str[:width].where(query_date.str.len() >= width)
                own_in_group = (reference_date.str.len() >= width) & (reference_date.str[:width] == qk)
                f[f'count_{width}'] = qk.map(rk.value_counts()).fillna(0).to_numpy() - own_in_group.to_numpy(dtype=int)
            rawr = ref.track_album_release_date.fillna('').astype(str)
            rawq = query_date
            same_date = (reference_date == query_date) & query_date.ne('')
            full_query_date = rawq.str.fullmatch(r'\d{4}-\d{2}-\d{2}').fillna(False).to_numpy()
            for col in AUDIO:
                num = pd.to_numeric(ref[col], errors='coerce')
                agg = pd.DataFrame({'date':rawr,'v':num}).groupby('date').v.agg(['sum','count'])
                sm = rawq.map(agg['sum']).fillna(0).to_numpy()
                ct = rawq.map(agg['count']).fillna(0).to_numpy()
                own = f[col].to_numpy()
                ref_own = pd.to_numeric(pd.Series(qids).map(reference_by_id[col]), errors='coerce').to_numpy()
                remove = same_date.to_numpy() & np.isfinite(ref_own)
                mean = (sm-np.nan_to_num(ref_own)*remove)/np.maximum(ct-remove,1)
                mean[(ct-remove<=0) | ~full_query_date] = np.nan
                f[col+'_date_mean'] = mean
                f[col+'_date_delta'] = own-mean
            # Standardization fitted on reference covariates, never labels.
            ref_audio = ref[AUDIO].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf],np.nan)
            query_audio = query[AUDIO].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf],np.nan)
            ref_valid = ref_audio.notna().any(axis=1).to_numpy()
            query_valid = query_audio.notna().any(axis=1).to_numpy()
            if not ref_valid.any():
                raise ValueError('Reference split has no audio features')
            audio_imputer = SimpleImputer(strategy='median', keep_empty_features=True)
            R = audio_imputer.fit_transform(ref_audio.loc[ref_valid])
            Q = audio_imputer.transform(query_audio)
            scaler = StandardScaler().fit(R)
            R, Q = scaler.transform(R), scaler.transform(Q)
            neighbor_ref = ref.loc[ref_valid].reset_index(drop=True)
            neighbor_ids = neighbor_ref.uri.to_numpy()
            neighbor_dates = neighbor_ref.track_album_release_date.astype(str).to_numpy()
            subsets = {'all':list(range(len(AUDIO)))}
            if view in ('multiscale','interactions'):
                subsets['production'] = [AUDIO.index(c) for c in ['energy','loudness','acousticness','instrumentalness']]
                subsets['rhythm'] = [AUDIO.index(c) for c in ['danceability','tempo','speechiness','duration_ms']]
            for name, cols in subsets.items():
                nd = np.full(len(query),20.0)
                nd2 = nd.copy()
                counts = {r:np.zeros(len(query)) for r in [1.,1.5,2.,3.]}
                for date, qi in query.groupby('track_album_release_date').indices.items():
                    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}',str(date)):
                        continue
                    ri = np.flatnonzero(neighbor_dates==str(date))
                    if not len(ri):
                        continue
                    D = cdist(Q[np.ix_(qi,cols)],R[np.ix_(ri,cols)])
                    D[qids[qi,None]==neighbor_ids[None,ri]] = np.inf
                    nd[qi] = np.minimum(D.min(axis=1),20)
                    if len(ri)>1:
                        nd2[qi] = np.minimum(np.partition(D,1,axis=1)[:,1],20)
                    for radius in counts:
                        counts[radius][qi] = (D<radius).sum(axis=1)
                f[name+'_nn1'] = nd
                f[name+'_nn2'] = nd2
                for radius, val in counts.items():
                    f[name+'_close_'+str(radius)] = val
                for col in [name+'_nn1',name+'_nn2']+[name+'_close_'+str(radius) for radius in counts]:
                    f.loc[~query_valid | ~full_query_date,col] = np.nan
        if view == 'interactions':
            cols = ['energy','danceability','valence','acousticness','instrumentalness','speechiness']
            for i,c in enumerate(cols):
                for b in cols[i+1:]:
                    f[c+'__'+b] = f[c]*f[b]
            f['beats'] = f.duration_ms*f.tempo/60000
            f['bars'] = f.beats/f.time_signature.replace(0,np.nan)
        if self.extra is not None:
            a,b = self.extra
            e = pd.concat([a.iloc[fit],a.iloc[apply],b], ignore_index=True)
            for c in e:
                f['external_'+str(c)] = e[c].to_numpy()
        f = f.replace([np.inf,-np.inf],np.nan).astype(np.float32)
        n, v = len(fit),len(apply)
        value = (f.iloc[:n].copy(), f.iloc[n:n+v].copy(), f.iloc[n+v:].copy())
        self.cache[key] = value
        while len(self.cache) > 12:
            self.cache.popitem(last=False)
        return value


def suggest(t, families, smoke=False, project_features=False):
    c = {'family':t.suggest_categorical('family',families),
         'view':t.suggest_categorical('view',['project'] if project_features else ['basic','album','multiscale','interactions']),
         'duplicate_power':t.suggest_categorical('duplicate_power',[0.,.5,1.]),
         'residual_alpha':t.suggest_categorical('residual_alpha',[0.,.15,.35]),
         'residual_bandwidth':t.suggest_categorical('residual_bandwidth',[.75,1.5,3.]),
         'residual_shrinkage':t.suggest_categorical('residual_shrinkage',[1.,3.,10.])}
    family = c['family']
    if family in ['lgbm','xgb','cat']:
        c['rounds'] = t.suggest_int('rounds',300,2200,step=100)
        c['lr'] = t.suggest_float('lr',.008,.12,log=True)
        c['depth'] = t.suggest_int('depth',3,9)
        c['l2'] = t.suggest_float('l2',.1,50.,log=True)
        c['colsample'] = t.suggest_float('colsample',.35,1.)
        if family == 'lgbm':
            c['leaves'] = t.suggest_int('leaves',8,64,log=True)
            c['min_leaf'] = t.suggest_int('min_leaf',8,100,log=True)
            c['bins'] = t.suggest_categorical('bins',[63,127,255])
        if family == 'xgb':
            c['min_child'] = t.suggest_float('min_child',1,30,log=True)
            c['subsample'] = t.suggest_float('subsample',.6,1.)
        if family == 'cat':
            c['random_strength'] = t.suggest_float('random_strength',.01,8,log=True)
            c['borders'] = t.suggest_categorical('borders',[64,128,254])
    else:
        c['rounds'] = t.suggest_int('trees',300,900,step=100)
        c['depth'] = t.suggest_int('forest_depth',5,30)
        c['min_leaf'] = t.suggest_int('forest_min_leaf',1,20,log=True)
        c['colsample'] = t.suggest_float('forest_colsample',.15,1.)
    if smoke:
        c['rounds'] = 20
    return c


def make_model(c, threads, seed):
    common = dict(random_state=seed)
    f = c['family']
    if f == 'lgbm':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=c['rounds'],learning_rate=c['lr'],
            max_depth=c['depth'],num_leaves=min(c['leaves'],2**c['depth']),
            min_child_samples=c['min_leaf'],reg_lambda=c['l2'],
            colsample_bytree=c['colsample'],max_bin=c['bins'],
            verbosity=-1,n_jobs=threads,**common)
    if f == 'xgb':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=c['rounds'],learning_rate=c['lr'],
            max_depth=c['depth'],reg_lambda=c['l2'],colsample_bytree=c['colsample'],
            min_child_weight=c['min_child'],subsample=c['subsample'],
            tree_method='hist',eval_metric='logloss',n_jobs=threads,**common)
    if f == 'cat':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=c['rounds'],learning_rate=c['lr'],
            depth=c['depth'],l2_leaf_reg=c['l2'],rsm=c['colsample'],
            random_strength=c['random_strength'],border_count=c['borders'],
            loss_function='Logloss',verbose=False,allow_writing_files=False,
            thread_count=threads,random_seed=seed)
    cls = ExtraTreesClassifier if f=='extra' else RandomForestClassifier
    return cls(n_estimators=c['rounds'],max_depth=c['depth'],
        min_samples_leaf=c['min_leaf'],max_features=c['colsample'],
        n_jobs=threads,**common)


def add_fold_date_encoding(X,V,T,tr,te,y,fit,apply,groups):
    """Group-cross-fit train encodings and fit validation/test on train only."""
    fit_rows=tr.iloc[fit].astype(object).where(pd.notna(tr.iloc[fit]),'').to_dict('records')
    valid_rows=tr.iloc[apply].astype(object).where(pd.notna(tr.iloc[apply]),'').to_dict('records')
    test_rows=te.astype(object).where(pd.notna(te),'').to_dict('records')
    enc=GroupedDateTargetEncoder(smoothing=2.)
    efit=enc.fit_transform(fit_rows,y[fit],groups[fit]);eval_=enc.transform(valid_rows)
    etest=enc.transform(test_rows) if test_rows else np.empty((0,6))
    cols=['date_te_year','date_te_month','date_te_exact','date_count_year','date_count_month','date_count_exact']
    def attach(frame,array):
        return pd.concat([frame.reset_index(drop=True),pd.DataFrame(array,columns=cols)],axis=1)
    return attach(X,efit),attach(V,eval_),attach(T,etest)


def cv_run(c, folds, features, tr, y, threads, seed, deadline, trial=None, date_target_encoding=False):
    p = np.full(len(tr),np.nan)
    pt = np.zeros(len(features.te))
    scores = []
    for k,(a,b) in enumerate(folds):
        if STOP or time.monotonic() >= deadline:
            raise TimeoutError('Budget/stop reached between folds')
        X,V,T = features.get(a,b,c['view'])
        if date_target_encoding:
            X,V,T=add_fold_date_encoding(X,V,T,tr,features.te,y,a,b,features.groups)
        # Missing values have one consistent representation across all families.
        imp = SimpleImputer(strategy='median',keep_empty_features=True)
        X = imp.fit_transform(X)
        V,T = imp.transform(V),imp.transform(T) if len(T) else np.empty((0,X.shape[1]))
        w = duplicate_weights(tr.iloc[a],c['duplicate_power'])
        m = make_model(c,threads,seed+k)
        # Fixed iterations: outer validation labels are not used for early stopping.
        m.fit(X,y[a],sample_weight=w)
        p[b] = m.predict_proba(V)[:,1]
        if len(T):
            fold_test = m.predict_proba(T)[:,1]
            pt += fold_test/len(folds)
        if c.get('residual_alpha',0)>0:
            # Inner OOF residuals: no outer validation labels enter this correction.
            inner_pred = np.full(len(a),np.nan)
            inner = StratifiedGroupKFold(3,shuffle=True,random_state=seed+100+k)
            for it,iv in inner.split(a,y[a],features.groups[a]):
                if STOP or time.monotonic()>=deadline:raise TimeoutError('Residual budget')
                I,J,_ = features.get(a[it],a[iv],c['view'])
                if date_target_encoding:
                    I,J,_=add_fold_date_encoding(I,J,I.iloc[:0],tr,features.te.iloc[:0],y,a[it],a[iv],features.groups)
                im = SimpleImputer(strategy='median',keep_empty_features=True)
                I=im.fit_transform(I);J=im.transform(J)
                iw=duplicate_weights(tr.iloc[a[it]],c['duplicate_power'])
                model=make_model(c,threads,seed+200+k)
                model.fit(I,y[a[it]],sample_weight=iw)
                inner_pred[iv]=model.predict_proba(J)[:,1]
            residual=y[a]-inner_pred
            # One recording contributes once, with average residual across copies.
            ref=tr.iloc[a].copy();ref['_residual']=residual
            ref=ref.groupby('uri',dropna=True).agg({**{v:'first' for v in AUDIO},
                'track_album_release_date':'first','_residual':'mean'}).reset_index()
            numeric=ref[AUDIO].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
            ref=ref.loc[numeric.notna().any(axis=1)].reset_index(drop=True)
            numeric=ref[AUDIO].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
            if ref.empty:
                raise ValueError('Residual reference contains no usable audio rows')
            audio_imp=SimpleImputer(strategy='median',keep_empty_features=True)
            R=audio_imp.fit_transform(numeric)
            scaler=StandardScaler().fit(R);R=scaler.transform(R)
            def correction(query):
                out=np.zeros(len(query));numeric=query[AUDIO].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
                valid=numeric.notna().any(axis=1).to_numpy();Q=scaler.transform(audio_imp.transform(numeric))
                dates=ref.track_album_release_date.astype(str).to_numpy()
                for date,qi in query.reset_index(drop=True).groupby('track_album_release_date').indices.items():
                    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}',str(date)):continue
                    ri=np.flatnonzero(dates==str(date))
                    if not len(ri):continue
                    D=cdist(Q[qi],R[ri]);W=np.exp(-.5*(D/c['residual_bandwidth'])**2)
                    W[~valid[qi]]=0
                    out[qi]=(W@ref._residual.to_numpy()[ri])/(c['residual_shrinkage']+W.sum(axis=1))
                return out
            p[b]=np.clip(p[b]+c['residual_alpha']*correction(tr.iloc[b]),0,1)
            if len(T):pt+=(np.clip(fold_test+c['residual_alpha']*correction(features.te),0,1)-fold_test)/len(folds)
        scores.append(float(roc_auc_score(y[b],p[b])))
        if trial is not None:
            trial.report(float(np.mean(scores)),k)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return p,np.clip(pt,0,1),float(np.mean(scores)),scores


def export(out, tr, te, y, records, sample, prediction_column, invert):
    """Save conservative equal/diversity and greedy blends, with explicit selection bias."""
    if not records:
        return
    records = sorted(records,key=lambda r:r['score'],reverse=True)
    # Keep one best candidate per family/view, maximum 12.
    pool=[]; seen=set()
    for r in records:
        k=(r['config']['family'],r['config']['view'],r['config'].get('residual_alpha',0)>0)
        if k not in seen:
            pool.append(r);seen.add(k)
        if len(pool)>=12:
            break
    arrays=[np.load(out/r['file']) for r in pool]
    O=np.column_stack([rankdata(a['oof'],method='average')/len(y) for a in arrays])
    T=np.column_stack([rankdata(a['test'],method='average')/len(te) for a in arrays]) if len(te) else np.empty((0,len(pool)))
    # Greedy ensemble selection with replacement; caps complexity at 25 steps.
    total=np.zeros(len(y));weights=np.zeros(len(pool));best_score=-1.;best_w=None
    for step in range(25):
        values=[roc_auc_score(y,total+O[:,j]) for j in range(len(pool))]
        j=int(np.argmax(values));total+=O[:,j];weights[j]+=1
        if values[j]>best_score:
            best_score=float(values[j]);best_w=weights/weights.sum()
    oof=O@best_w
    pd.DataFrame({'row':np.arange(len(tr)),'uri':tr.uri,'label':y,
                  'ensemble':oof}).to_csv(out/'oof.csv',index=False)
    def write(name,pred):
        if not len(te):return
        pred = 1-pred if invert else pred
        if sample is not None:
            frame=sample.copy();frame[prediction_column]=pred
        else:
            frame=pd.DataFrame({'ID':te['ID'].to_numpy(),prediction_column:pred})
        frame.to_csv(out/name,index=False)
    write('submission_best.csv',arrays[0]['test'])
    write('submission_equal.csv',T.mean(axis=1))
    write('submission_greedy.csv',T@best_w)
    atomic_json(out/'summary.json',{
        'warning':'All CV figures are selection-biased. Second seed reuses labels; not an independent test.',
        'prediction_column':prediction_column,'output_inverted':invert,
        'best_mean_fold_auc':records[0]['score'],
        'greedy_pooled_oof_auc_selection_biased':best_score,
        'candidate_weights':[{'config':r['config'],'weight':float(w),'score':r['score']} for r,w in zip(pool,best_w)],
        'best_candidate':records[0]})
    print(f"Checkpoint: best mean-fold AUC={records[0]['score']:.6f}; selected blend pooled AUC={best_score:.6f} (optimistic)",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',default=str(PROJECT_ROOT/'data/train.csv'))
    p.add_argument('--test',default=str(PROJECT_ROOT/'data/test.csv'))
    p.add_argument('--sample-submission');p.add_argument('--hours',type=float,default=24)
    p.add_argument('--out',default=str(PROJECT_ROOT/'experiments/day_search'));p.add_argument('--threads',type=int,default=DEFAULT_THREADS)
    p.add_argument('--prediction-column',default='Popularity_Type',help='Prediction column in output or sample submission')
    p.add_argument('--invert',action='store_true',help='Write 1 - predicted High score; CV metrics remain normal AUC')
    p.add_argument('--optimizer',choices=['auto','optuna','random'],default='auto',help='Optuna TPE if installed; otherwise resumable random search')
    p.add_argument('--models',default='lgbm,xgb,cat,extra,rf')
    p.add_argument('--search-folds',type=int,default=3);p.add_argument('--confirm-folds',type=int,default=5)
    p.add_argument('--finalists',type=int,default=12);p.add_argument('--max-trials',type=int,default=100000)
    p.add_argument('--seed',type=int,default=20261029)
    p.add_argument('--transductive',action='store_true',help='Use all train/test covariates for label-free album features')
    p.add_argument('--feature-builder',help='Optional local file.py:function; see README')
    p.add_argument('--project-features',action='store_true',help='Use project audio/date/group features with the unlabeled test pool')
    p.add_argument('--date-target-encoding',action='store_true',help='Add group-cross-fitted date label summaries inside every fold')
    p.add_argument('--smoke',action='store_true',help='20 trees, 2+2 folds, at most 5 trials')
    a=p.parse_args()
    if a.project_features:
        if a.feature_builder:p.error('--project-features cannot be combined with --feature-builder')
        a.feature_builder=str(Path(__file__).with_name('project_features.py'))+':make'
        a.transductive=True
        a.date_target_encoding=True
        print('Project features: using unlabeled train + test covariates and group-cross-fitted date encodings.',flush=True)
    if a.hours<=0 or a.threads<1:p.error('hours and threads must be positive')
    if a.prediction_column=='ID':p.error('The prediction column must not overwrite the test ID')
    if a.search_folds<2 or a.confirm_folds<2 or a.finalists<1 or a.max_trials<1:p.error('fold counts must be at least 2; finalists and max-trials must be positive')
    if a.optimizer=='optuna' and optuna is None:p.error('Optuna is not installed; install requirements.txt or use --optimizer random')
    optimizer='optuna' if a.optimizer=='optuna' or (a.optimizer=='auto' and optuna is not None) else 'random'
    if a.smoke:a.search_folds=a.confirm_folds=2;a.max_trials=min(a.max_trials,5)
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    tr=pd.read_csv(a.train);te=pd.read_csv(a.test)
    for label,frame in [('train',tr),('test',te)]:
        missing=set(AUDIO+['uri','track_album_release_date'])-set(frame.columns)
        if missing:p.error(f'{label} CSV missing columns: {sorted(missing)}')
        frame['uri']=frame['uri'].astype('string').str.strip().replace('',pd.NA)
    if 'Popularity_Type' not in tr:p.error('Missing Popularity_Type target')
    tr=tr.loc[tr.Popularity_Type.notna()].reset_index(drop=True)
    if not set(tr.Popularity_Type.unique())<= {'High','Low'}:p.error('Expected High/Low labels')
    if len(te)<1:p.error('Test CSV is empty')
    if not a.sample_submission and 'ID' not in te:p.error('Test CSV has no ID; supply --sample-submission or an ID column')
    y=tr.Popularity_Type.eq('High').to_numpy().astype(int)
    group=connected_groups(tr)
    families=a.models.split(',')
    modules={'lgbm':'lightgbm','xgb':'xgboost','cat':'catboost','extra':'sklearn','rf':'sklearn'}
    for f in families:
        if f not in modules:p.error('Unknown model '+f)
        if importlib.util.find_spec(modules[f]) is None:p.error('Install missing package '+modules[f])
    extra=None
    if a.feature_builder:
        path,fn=a.feature_builder.rsplit(':',1)
        spec=importlib.util.spec_from_file_location('custom_features',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        extra=getattr(module,fn)(tr.drop(columns='Popularity_Type').copy(),te.drop(columns='Popularity_Type',errors='ignore').copy())
        if len(extra)!=2 or len(extra[0])!=len(tr) or len(extra[1])!=len(te):p.error('Invalid feature-builder lengths')
        if any(not pd.api.types.is_numeric_dtype(t) for frame in extra for t in frame.dtypes):p.error('External features must be numeric')
    sample=pd.read_csv(a.sample_submission) if a.sample_submission else None
    if sample is not None and len(sample)!=len(te):p.error('Sample submission row count differs')
    if sample is not None and a.prediction_column not in sample:p.error(f'Sample submission lacks {a.prediction_column!r}; use --prediction-column')
    if sample is not None and 'ID' in sample and 'ID' in te and not sample.ID.astype(str).equals(te.ID.astype(str)):p.error('Sample submission IDs are not aligned with test rows')
    signature=hashlib.sha256()
    for path in [a.train,a.test]:signature.update(Path(path).read_bytes())
    if a.feature_builder:signature.update(Path(a.feature_builder.rsplit(':',1)[0]).read_bytes())
    if a.sample_submission:signature.update(Path(a.sample_submission).read_bytes())
    if a.project_features:
        for path in [PROJECT_ROOT/'experiments/similarity_study/similarity.py',PROJECT_ROOT/'experiments/pdf_idea/date_pool.py',PROJECT_ROOT/'experiments/research85/target_encode.py',PROJECT_ROOT/'experiments/preprocessing.py',Path(__file__).with_name('random_search.py')]:
            signature.update(path.read_bytes())
    meta={'version':3,'data_hash':signature.hexdigest(),'seed':a.seed,'families':families,
          'search_folds':a.search_folds,'confirm_folds':a.confirm_folds,'transductive':a.transductive,'smoke':a.smoke,
          'optimizer':optimizer,'prediction_column':a.prediction_column,'invert':a.invert,
          'project_features':a.project_features,'date_target_encoding':a.date_target_encoding,
          'feature_builder':a.feature_builder,'script_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    manifest=out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=meta:p.error('Output directory belongs to a different configuration/data/script. Use a new --out.')
    atomic_json(manifest,meta)
    folds=list(StratifiedGroupKFold(a.search_folds,shuffle=True,random_state=a.seed).split(tr,y,group))
    finalfolds=list(StratifiedGroupKFold(a.confirm_folds,shuffle=True,random_state=a.seed+71).split(tr,y,group))
    features=Features(tr,te,a.transductive,extra,replace_features=a.project_features)
    features.groups=np.asarray(group)
    search_records=json.loads((out/'search_candidates.json').read_text()) if (out/'search_candidates.json').exists() else []
    confirmed=json.loads((out/'confirmed_candidates.json').read_text()) if (out/'confirmed_candidates.json').exists() else []
    start=time.monotonic();deadline=start+a.hours*3600;search_deadline=start+a.hours*3600*.7
    storage='sqlite:///'+str(out/'optuna.db')
    if optimizer=='optuna':
        study=optuna.create_study(study_name='day',storage=storage,direction='maximize',load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=a.seed+len(search_records),n_startup_trials=30),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=25,n_warmup_steps=1))
    else:
        from random_search import RandomStudy
        study=RandomStudy(out/'random_study.json',a.seed)
        print('Optuna TPE unavailable or disabled: using resumable random search (no pruning).',flush=True)
    if not study.trials and not (optimizer=='random' and study.state['queue']):
        for f in families:
            preset={'family':f,'view':'project' if a.project_features else 'album','duplicate_power':.5,'residual_alpha':0.}
            if f in ['lgbm','xgb','cat']:
                preset.update(rounds=1100,lr=.025,depth=6,l2=5.,colsample=.6)
            if f=='lgbm':preset.update(leaves=30,min_leaf=20,bins=255)
            if f=='xgb':preset.update(min_child=5.,subsample=.9)
            if f=='cat':preset.update(random_strength=1.,borders=254)
            if f=='cat' and a.project_features:
                # Matched-fold diagnostic: this fixed configuration scored
                # 0.8474 with these features and grouped date encoding.
                preset.update(duplicate_power=0.,rounds=1300,lr=.015929186619987528,
                    depth=5,l2=4.389663704201526,colsample=.39019251353562023,
                    random_strength=.2603695359797555,borders=64)
            if f in ['extra','rf']:
                preset.update(trees=600,forest_depth=19,forest_min_leaf=3,forest_colsample=.3)
            study.enqueue_trial(preset)
        for f in ['lgbm','cat','extra']:
            if f in families:study.enqueue_trial({'family':f,'view':'project' if a.project_features else 'album','duplicate_power':.5,'residual_alpha':.15})
    def objective(t):
        if STOP:
            study.stop()
            raise TimeoutError('Stop requested')
        c=suggest(t,families,a.smoke,a.project_features);t.set_user_attr('config',c)
        o,tt,score,fs=cv_run(c,folds,features,tr,y,a.threads,a.seed,search_deadline,t,a.date_target_encoding)
        file=f'search_{t.number}.npz';np.savez_compressed(out/file,oof=o,test=tt)
        search_records.append({'id':t.number,'config':c,'score':score,'fold_scores':fs,'file':file})
        atomic_json(out/'search_candidates.json',search_records)
        # Always leave a usable completed-candidate checkpoint.
        if score>=max(r['score'] for r in search_records):export(out,tr,te,y,search_records,sample,a.prediction_column,a.invert)
        return score
    try:
        study.optimize(objective,n_trials=a.max_trials,timeout=max(0,search_deadline-time.monotonic()),catch=(TimeoutError,))
    except KeyboardInterrupt:stop()
    # Recheck diverse candidates rather than spending the entire day tuning one view.
    ranked=sorted(search_records,key=lambda r:r['score'],reverse=True)
    shortlist=[];used=set()
    for r in ranked:
        k=(r['config']['family'],r['config']['view'],r['config'].get('residual_alpha',0)>0)
        if k not in used:shortlist.append(r);used.add(k)
        if len(shortlist)>=a.finalists:break
    if len(shortlist)<a.finalists:
        known={r['id'] for r in shortlist}
        shortlist += [r for r in ranked if r['id'] not in known][:a.finalists-len(shortlist)]
    done={r['id'] for r in confirmed}
    try:
        for r in shortlist:
            if STOP or time.monotonic()>=deadline:break
            if r['id'] in done:continue
            o,tt,score,fs=cv_run(r['config'],finalfolds,features,tr,y,a.threads,a.seed+71,deadline,date_target_encoding=a.date_target_encoding)
            file=f"confirmed_{r['id']}.npz";np.savez_compressed(out/file,oof=o,test=tt)
            confirmed.append(dict(r,score=score,fold_scores=fs,file=file))
            atomic_json(out/'confirmed_candidates.json',confirmed)
            export(out,tr,te,y,confirmed,sample,a.prediction_column,a.invert)
    except (TimeoutError,KeyboardInterrupt):pass
    export(out,tr,te,y,confirmed or search_records,sample,a.prediction_column,a.invert)
    study.trials_dataframe().to_csv(out/'trials.csv',index=False)
    print('Finished. Output:',out,flush=True)
    if not search_records:raise SystemExit('No trial completed. Increase budget or use --smoke.')

if __name__=='__main__':
    main()

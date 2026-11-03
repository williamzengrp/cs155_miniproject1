"""Screen label-feature blocks with a LightGBM probe (final_params lgbm, 58 label-free feats + block).
python -I screen.py cfg1,cfg2,...   env OFFS=0,100 (model seed offsets averaged), SEED0=1000, NREP=5, MODEL=lgbm|et
Paired per-repeat AUC diff vs config 'base' (= te BEST_SPEC) with the same seeds; OOFs saved to out/."""
import json, os, sys, time, warnings
import numpy as np, pandas as pd
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/main'); sys.path.insert(0, WORK + '/r2_lab')
import harness as H
from features_final import build_features_final
import labx as LX
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
OUT = WORK + '/r2_lab/out'
B = list(LX.BEST_SPEC)
CONFIGS = {
    'base': dict(spec=B),
    'inner10': dict(spec=B, n_inner=10),
    'msk': dict(spec=B + ['knnw_yr_10', 'knnw_yr_150']),
    'msw': dict(spec=B + ['knnw_yr05_40', 'knnw_yr3_40']),
    'sub': dict(spec=B + ['knnw_yrA_40', 'knnw_yrB_40']),
    'dedup': dict(spec=B, dedup=True),
    'dmin': dict(spec=B + ['dmin_hi', 'dmin_lo']),
    'tek': dict(spec=B + ['tek_date_m10']),
    'chain': dict(spec=B + ['chain_year-ym-date-alb_m10']),
    'yc': dict(spec=B + ['teh_yc8_m10']),
    'kall': dict(spec=B + ['knnw_yr_10', 'knnw_yr_150', 'knnw_yr05_40', 'knnw_yr3_40', 'knnw_yrA_40', 'knnw_yrB_40']),
    'dext': dict(spec=B + ['dmin_hi', 'dmin_lo', 'tek_date_m10']),
    'kx': dict(spec=B + ['knnwx_yr_40']),
    'kxr': dict(spec=['knnwx_yr_40', 'teh_date_m10', 'cnt_date']),
    'chyc': dict(spec=B + ['chain_year-ym-date-alb_m10', 'teh_yc8_m10']),
}
names = sys.argv[1].split(',')
OFFS = [int(x) for x in os.environ.get('OFFS', '0,100').split(',')]
MODEL = os.environ.get('MODEL', 'lgbm')
SEED0, NREP = int(os.environ.get('SEED0', '1000')), int(os.environ.get('NREP', '5'))
P = json.load(open(WORK + '/main/final_params.json'))
tr, te, y = H.load()
X, Xt = build_features_final(tr, te)
XA = pd.concat([X, Xt], ignore_index=True)
n = len(tr)
splits = H.make_splits(tr, y, n_rep=NREP, seed0=SEED0)
groups = {}
for c in names:
    cf = CONFIGS[c]
    groups.setdefault((cf.get('n_inner', 5), cf.get('dedup', False)), []).append(c)
oof = {c: np.full((NREP, n), np.nan) for c in names}
fill = X.median()


def mk(seed):
    if MODEL == 'lgbm':
        return LGBMClassifier(**P['lgbm'], random_state=seed, n_jobs=2, verbose=-1)
    return ExtraTreesClassifier(**P['et'], random_state=seed, n_jobs=2)


t0 = time.time()
for k, (r, fi, vi, el) in enumerate(splits):
    for (ni, dd), cs in groups.items():
        spec = list(dict.fromkeys(sum([CONFIGS[c]['spec'] for c in cs], [])))
        Ff, (Fv,) = LX.label_features(XA, y, fi, [vi], tr, te, seed=k, spec=spec, n_inner=ni, dedup=dd)
        for c in cs:
            cols = CONFIGS[c]['spec']
            Mf = pd.concat([X.iloc[fi].reset_index(drop=True), Ff[cols]], axis=1)
            Mv = pd.concat([X.iloc[vi].reset_index(drop=True), Fv[cols]], axis=1)
            if MODEL == 'et':
                f2 = pd.concat([fill, Ff[cols].median()])
                Mf, Mv = Mf.fillna(f2), Mv.fillna(f2)
            pv = 0
            for off in OFFS:
                pv = pv + mk(k + off).fit(Mf, y[fi]).predict_proba(Mv)[:, 1] / len(OFFS)
            oof[c][r, vi[el]] = pv[el]
    if k % 5 == 4:
        print(f'fold {k} {time.time() - t0:.0f}s', flush=True)

tag = f'{MODEL}_s{SEED0}_o' + '-'.join(map(str, OFFS))


def aucs(o):
    return np.array([roc_auc_score(y[~np.isnan(o[r])], o[r][~np.isnan(o[r])]) for r in range(len(o))])


for c in names:
    np.save(f'{OUT}/scr_{c}_{tag}.npy', oof[c])
bp = f'{OUT}/scr_base_{tag}.npy'
ab = aucs(np.load(bp)) if os.path.exists(bp) else None
for c in names:
    a = aucs(oof[c])
    line = f'{tag} {c:10s} AUC {a.mean():.5f} per-rep {np.round(a, 4)}'
    if ab is not None:
        d = a - ab
        line += f'  diff {d.mean():+.5f} pos {(d > 0).sum()}/{len(d)} per-rep {np.round(d, 4)}'
    print(line, flush=True)
    with open(f'{OUT}/screen_results.txt', 'a') as f:
        f.write(line + '\n')

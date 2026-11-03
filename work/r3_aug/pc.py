"""r3_aug screening helper: pubcheck (work/main/pubcheck.py, imported read-only) on screening runs kept in
r3_aug/cache/runs (token 'run:<file stem>') or chaug assemblies ('chaug:<cat tag>|<et tag>|<nrep>') without writing
to results/. Also prints the paired per-repeat AUC vs chrep_bl on the repeats the run has.
python -I pc.py TOKEN [TOKEN ...]"""
import os, sys
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/main')
import harness as H
import pubsim as PS
import pubcheck as PC

RUNS = WORK + '/r3_aug/cache/runs/'
_orig = H.load_res


def _load(name):
    if name.startswith('run:') or name.startswith('run2:'):
        d = np.load((RUNS + name[4:] if name.startswith('run:') else RUNS[:-1] + '2/' + name[5:]) + '.npz')
        oof = d['oof']
        full = np.full((5, oof.shape[1]), np.nan)
        full[:oof.shape[0]] = oof
        return dict(oof=full)
    if name.startswith('chaug:'):
        return dict(oof=chaug_oof(name[6:]))
    return _orig(name)


def chaug_oof(spec):
    """'cattag|ettag|nrep' -> 0.6*cat + 0.4*et corrected OOF (tag 'orig' = replica cache)."""
    ct, et, nrep = spec.split('|')
    tr, te, y = H.load()
    splits = H.make_splits(tr, y, n_rep=5)
    oof = np.full((5, len(y)), np.nan)
    for k in range(5 * int(nrep)):
        r, fit, val, elig = splits[k]
        C = dict(np.load(f'{WORK}/ch/cache/chfull_std_k{k}.npz'))
        pc = C['cat_v'] if ct == 'orig' else np.load(f'{WORK}/r3_aug/cache/chaug_cat_{ct}_k{k}.npz')['v']
        pe = C['et_v'] if et == 'orig' else np.load(f'{WORK}/r3_aug/cache/chaug_et_{et}_k{k}.npz')['v']
        oof[r, val[elig]] = (0.6 * pc + 0.4 * pe)[elig]
    return oof


H.load_res = _load
PS.H.load_res = _load


def rk(v):
    return rankdata(v) / len(v)


def main():
    tr, te, y = H.load()
    ch = _orig('chrep_bl')['oof']
    for nm in sys.argv[1:]:
        o = _load(nm)['oof']
        reps = [r for r in range(5) if not np.isnan(o[r]).all()]
        a = np.array([roc_auc_score(y[~np.isnan(o[r])], o[r][~np.isnan(o[r])]) for r in reps])
        c = np.array([roc_auc_score(y[~np.isnan(ch[r])], ch[r][~np.isnan(ch[r])]) for r in reps])
        bl = {}
        for w in (0.2, 0.3, 0.5):
            bl[w] = np.mean([roc_auc_score(y[~np.isnan(o[r])], (1 - w) * rk(ch[r][~np.isnan(o[r])]) + w * rk(o[r][~np.isnan(o[r])])) for r in reps]) - c.mean()
        print(f'{nm}: reps {reps} AUC {a.mean():.5f} vs chrep_bl {c.mean():.5f} ({a.mean() - c.mean():+.5f})  '
              f'blend gain w.2 {bl[0.2]:+.5f} w.3 {bl[0.3]:+.5f} w.5 {bl[0.5]:+.5f}', flush=True)
    if os.environ.get('NOPC') != '1':
        PC.check(sys.argv[1:])
    if os.environ.get('EXTRA') == '1':
        extra(sys.argv[1:])


def extra(names, npub=431):
    """solo / heavy-weight candidates (same simulator, same conditioning as pubcheck)."""
    from scipy.stats import norm
    cands = dict(PC.OBS_DEF)
    for nm in names:
        cands[f'solo {nm}'] = [(nm, 1.0)]
        cands[f'ch.3+{nm}.7'] = [('ch', 0.3), (nm, 0.7)]
        cands[f'ch.45+catalb.1+hc2lgbm.15+{nm}.3'] = [('ch', 0.45), ('zoo_cat_alb', 0.1), ('graph_hc2_lgbm', 0.15), (nm, 0.3)]
    res = PS.run(cands, npub, verbose=False, extra_obs=PC.EXTRA)
    g = npub / 3521; fac = np.sqrt((1 + g) / (1 - g))
    for name, cvg, cm, sd, p in res:
        pc = 1 - norm.cdf((PS.TARGET - PS.CH_PUB - cm) / (sd * fac))
        print(f'{name[:60]:60s} CV gain vs ch {cvg:+.5f}  E[public] {PS.CH_PUB + cm:.5f} (sd {sd * fac:.5f})  P(>={PS.TARGET}) {pc:.3f}')


if __name__ == '__main__':
    main()

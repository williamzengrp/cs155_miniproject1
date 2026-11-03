"""ch agent: stage 2 of the residual-correction test (see resid.py). Cheap: loads cached base / inner-OOF predictions.

usage:
  python -I resid_eval.py tune 0                 -> grid over (alpha, bw, shrink) on the tuning splits
  python -I resid_eval.py std 0,100,200 SETTINGS -> chosen settings on the standard splits, per seed set and
                                                    3-seed-set average (per-repeat AUC averaged over seed sets)
  SETTINGS = 'a:bw:s;a:bw:s;...'   (optional --save NAME a:bw:s  -> H.save for offset 0 with fold-avg test)
"""
import pickle, sys
import numpy as np
WORK = './work'
sys.path.insert(0, WORK)
sys.path.insert(0, WORK + '/ch')
import harness as H
from ch_common import ResidualCorrector

kind = sys.argv[1]
offs = [int(s) for s in sys.argv[2].split(',')]
MODEL = 'lgbm'
tr, te, y = H.load()
splits = H.make_splits(tr, y, n_rep=5) if kind == 'std' else H.make_splits(tr, y, n_rep=2, seed0=7000)


def prep(off):
    C = pickle.load(open(f'{WORK}/ch/cache/resid_{MODEL}_{kind}_off{off}.pkl', 'rb'))
    F = []
    for k, (r, fit_idx, val_idx, elig) in enumerate(splits):
        f = C['folds'][k]
        rc = ResidualCorrector(tr.iloc[fit_idx], y[fit_idx] - f['inner'])
        F.append(dict(pv=f['pv'], pt=f['pt'], iv=rc.prepare(tr.iloc[val_idx]), it=rc.prepare(te)))
    return F


def score(F, a, bw, s, with_test=False):
    def fp(fit_idx, val_idx, k):
        f = F[k]
        if a == 0:
            return f['pv'], (f['pt'] if with_test else None)
        pv = np.clip(f['pv'] + a * ResidualCorrector.correction(f['iv'], bw, s), 0, 1)
        pt = np.clip(f['pt'] + a * ResidualCorrector.correction(f['it'], bw, s), 0, 1) if with_test else None
        return pv, pt
    return H.run_cv(fp, splits, y, verbose=False)


def parse(st):
    return [tuple(float(x) for x in t.split(':')) for t in st.split(';') if t]


if kind == 'tune':
    F = prep(offs[0])
    base = score(F, 0, 1, 1)
    print(f'base (alpha=0) {base["auc"]:.5f} per-rep {np.round(base["aucs"], 4)}')
    cov = np.mean([np.mean([it is not None for it in f['iv']]) for f in F])
    print(f'coverage: share of val rows with >=1 same-date fit song: {cov:.3f}')
    rows = []
    for a in (0.15, 0.35, 0.5, 0.75, 1.0):
        for bw in (0.75, 1.5, 3.0, 5.0):
            for s in (1.0, 3.0, 10.0):
                res = score(F, a, bw, s)
                d = res['aucs'] - base['aucs']
                rows.append((d.mean(), a, bw, s, res['auc'], d))
    rows.sort(key=lambda t: -t[0])
    print('alpha bw shrink | AUC | diff | per-rep diff')
    for dm, a, bw, s, auc, d in rows:
        print(f'{a:5.2f} {bw:4.2f} {s:5.1f} | {auc:.5f} | {dm:+.5f} | {" ".join(f"{x:+.4f}" for x in d)}')
else:
    settings = parse(sys.argv[3])
    per = {st: [] for st in settings}
    bases = []
    for off in offs:
        F = prep(off)
        b = score(F, 0, 1, 1)
        bases.append(b['aucs'])
        print(f'[off {off}] base {b["auc"]:.5f} per-rep {np.round(b["aucs"], 4)}', flush=True)
        for st in settings:
            res = score(F, *st)
            per[st].append(res['aucs'])
            d = res['aucs'] - b['aucs']
            print(f'[off {off}] a={st[0]} bw={st[1]} s={st[2]}: {res["auc"]:.5f}  diff {d.mean():+.5f} '
                  f'pos {(d > 0).sum()}/5  [{" ".join(f"{x:+.4f}" for x in d)}]', flush=True)
        if '--save' in sys.argv and off == offs[0]:
            name = sys.argv[sys.argv.index('--save') + 1]
            st = parse(sys.argv[sys.argv.index('--save') + 2])[0]
            res = score(F, *st, with_test=True)
            H.save(name, res)
            print('saved', name, res['auc'])
    B = np.mean(bases, axis=0)
    print(f'\n3-seed-set average: base {B.mean():.5f} per-rep {np.round(B, 4)}')
    print('| setting | AUC | diff | pos | per-rep diff | per-seed-set mean diffs |')
    for st in settings:
        A = np.mean(per[st], axis=0)
        d = A - B
        ps = [float(np.mean(per[st][i] - bases[i])) for i in range(len(offs))]
        print(f'| a={st[0]} bw={st[1]} s={st[2]} | {A.mean():.5f} | {d.mean():+.5f} | {(d > 0).sum()}/5 | '
              f'{" ".join(f"{x:+.4f}" for x in d)} | {" ".join(f"{x:+.4f}" for x in ps)} |')

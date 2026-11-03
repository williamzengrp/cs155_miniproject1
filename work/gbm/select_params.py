"""Robust pick from an Optuna study, then 5x5 confirmation vs the teammates' params.

usage: python -I select_params.py <model> <scheme> [n_top]
1. candidates = top-n trials (tuning splits seed0=7000) + 'centroid' of the top-10 (geometric mean for
   log-scaled params, mean otherwise, mode for categoricals).
2. re-score candidates on fresh selection splits (n_rep=2, seed0=8000); pick the best mean over the
   4 tuning+selection repeats (centroid is also scored on the tuning splits).
3. confirm on the standard 5x5 evaluation splits vs baseline params (same scheme), paired per repeat.
Writes the result to gbm/best_params_newcv.json under key <model> (merged), and gbm/sel_<model>.json.
"""
import sys
import os
import json
sys.path.insert(0, './work/gbm')
import numpy as np
import optuna
import lib as L

sys.argv = sys.argv[:1] + [sys.argv[1], '0'] + sys.argv[2:]     # reuse tune.suggest without running
name, scheme = sys.argv[1], (sys.argv[3] if len(sys.argv) > 3 else 'none')
n_top = int(sys.argv[4]) if len(sys.argv) > 4 else 5
from tune import suggest, study   # noqa: E402  (n_trials=0 -> no new trials)

optuna.logging.set_verbosity(optuna.logging.WARNING)
done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
done.sort(key=lambda t: -t.value)
dists = done[0].distributions


def centroid(trials):
    c = {}
    for k, d in dists.items():
        vals = [t.params[k] for t in trials]
        if isinstance(d, optuna.distributions.CategoricalDistribution):
            c[k] = max(set(vals), key=vals.count)
            continue
        v = np.exp(np.mean(np.log(vals))) if d.log else float(np.mean(vals))
        if isinstance(d, optuna.distributions.IntDistribution):
            v = int(round(v))
        c[k] = v
    return c


cands = [(f't{t.number}', t.params, t.value) for t in done[:n_top]]
cands.append(('centroid10', centroid(done[:10]), None))
tune_splits = L.H.make_splits(L.tr, L.y, n_rep=2, seed0=7000)
sel_splits = L.H.make_splits(L.tr, L.y, n_rep=2, seed0=8000)
base_trial = study.trials[0]
L.log(f'\n### select {name} (scheme {scheme}): {len(done)} trials; baseline params on tuning splits '
      f'{base_trial.value:.5f}; best {done[0].value:.5f} (t{done[0].number})')
rows = []
for cname, cp, tv in cands:
    p = suggest(name, optuna.trial.FixedTrial(cp))
    if tv is None:
        tv = L.run(name, p, tune_splits, scheme=scheme, verbose=False)['auc']
    sv = L.run(name, p, sel_splits, scheme=scheme, verbose=False)['auc']
    rows.append((cname, p, tv, sv, (tv + sv) / 2))
    L.log(f'- {cname:11s} tune {tv:.5f} select {sv:.5f} mean {(tv + sv) / 2:.5f}  '
          + json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in p.items()}))
pb = L.BASE[name]
bsel = L.run(name, pb, sel_splits, scheme=scheme, verbose=False)['auc']
L.log(f'- baseline    tune {base_trial.value:.5f} select {bsel:.5f} mean {(base_trial.value + bsel) / 2:.5f}')
best = max(rows, key=lambda r: r[4])
L.log(f'picked {best[0]}')

# confirmation on the standard evaluation splits
ev = L.H.make_splits(L.tr, L.y, n_rep=5)
rt = L.run(name, best[1], ev, scheme=scheme, verbose=False)
rb = L.run(name, pb, ev, scheme=scheme, verbose=False)
L.log(f'- 5x5 confirm {name}: tuned {rt["auc"]:.5f} {np.round(rt["aucs"], 4)} | baseline params '
      f'{rb["auc"]:.5f} {np.round(rb["aucs"], 4)} | diff {L.pdiff(rt, rb)}')
out = dict(name=name, scheme=scheme, picked=best[0], params=best[1], tune=best[2], select=best[3],
           eval_aucs=list(map(float, rt['aucs'])), base_eval_aucs=list(map(float, rb['aucs'])))
json.dump(out, open(os.path.join(L.M.GBM, f'sel_{name}_{scheme}.json'), 'w'), indent=1)
np.savez(os.path.join(L.M.GBM, f'eval_{name}_{scheme}.npz'), tuned_oof=rt['oof'], base_oof=rb['oof'],
         tuned_aucs=rt['aucs'], base_aucs=rb['aucs'])

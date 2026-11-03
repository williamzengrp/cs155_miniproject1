"""Small torch MLP (seed-averaged) under the shared harness. Runs in work/mdl/venv.

usage:
  torch_mlp.py tune                        -> grid on H.make_splits(n_rep=2, seed0=7000); predictions are
                                              snapshotted at several epochs in one run (epoch count tuned)
  torch_mlp.py std '<json params>' <save_name>
Inputs: per-fold median fill + quantile-normal scaling (fit rows only); optional piecewise-linear
encoding (PLE, quantile bins fit on fit rows) of every feature.
"""
import os, sys, json, time, warnings
sys.path.insert(0, './work/mdl')
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import common as C
warnings.filterwarnings('ignore')
torch.set_num_threads(int(os.environ.get('NJ', '2')))
H = C.H
tr, te, y, X0, Xt0 = C.data()
COMPACT_DROP = ['release_year', 'days_since_release', 't_log_age', 't_decade', 's_doy', 's_week', 's_quarter',
                's_mon_sin', 's_mon_cos', 's_dow_sin', 's_dow_cos', 's_doy_sin', 's_doy_cos', 's_is_friday',
                's_is_tuesday', 's_holiday', 's_summer'] + [c for c in X0.columns if c.startswith('g_dev_')]


def feats(name):
    if name == 'v2':
        return C.data_v2()
    return (X0, Xt0) if name == 'all' else (X0.drop(columns=COMPACT_DROP), Xt0.drop(columns=COMPACT_DROP))


def ple(A, others, n_bins):
    """Piecewise-linear encoding: per feature, quantile bin edges from fit rows; each bin -> value in [0,1]."""
    outs = [[] for _ in range(len(others) + 1)]
    for j in range(A.shape[1]):
        e = np.unique(np.quantile(A[:, j], np.linspace(0, 1, n_bins + 1)))
        if len(e) < 3:
            for o, M in zip(outs, [A] + others):
                o.append(M[:, [j]])
            continue
        lo, hi = e[:-1], e[1:]
        for o, M in zip(outs, [A] + others):
            v = M[:, [j]]
            o.append(np.clip((v - lo) / (hi - lo), 0, 1))
    return [np.hstack(o).astype(np.float32) for o in outs]


class Net(nn.Module):
    def __init__(self, d, h, p):
        super().__init__()
        layers, prev = [], d
        for w in h:
            layers += [nn.Linear(prev, w), nn.ReLU(), nn.Dropout(p)]
            prev = w
        layers += [nn.Linear(prev, 1)]
        self.f = nn.Sequential(*layers)

    def forward(self, x):
        return self.f(x).squeeze(-1)


def train_predict(P, A, yf, others, seed, snaps):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net(A.shape[1], P['h'], P['drop'])
    opt = torch.optim.AdamW(net.parameters(), lr=P.get('lr', 1e-3), weight_decay=P['wd'])
    lossf = nn.BCEWithLogitsLoss()
    Xa, ya = torch.tensor(A, dtype=torch.float32), torch.tensor(yf, dtype=torch.float32)
    Os = [torch.tensor(o, dtype=torch.float32) for o in others]
    bs, n = P.get('bs', 128), len(A)
    out = {}
    for ep in range(1, max(snaps) + 1):
        net.train()
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(net(Xa[b]), ya[b])
            loss.backward()
            opt.step()
        if ep in snaps:
            net.eval()
            with torch.no_grad():
                out[ep] = [net(o).numpy() for o in Os]
    return out


def prep(P, Xf, Xo):
    A, others = C.prep(Xf, Xo, scaler='quantile')
    if P.get('ple'):
        A, *others = ple(A, others, P['ple'])
    return A.astype(np.float32), [o.astype(np.float32) for o in others]


def cv(P, which, snaps):
    X, Xt = feats(P['feats'])
    splits = H.make_splits(tr, y, n_rep=2, seed0=7000) if which == 'tune' else H.make_splits(tr, y, n_rep=5)
    with_test = which == 'std'
    R = max(s[0] for s in splits) + 1
    oof = {e: np.full((R, len(y)), np.nan) for e in snaps}
    tst = {e: 0 for e in snaps}
    t0 = time.time()
    for k, (r, fi, vi, el) in enumerate(splits):
        A, others = prep(P, X.iloc[fi], [X.iloc[vi]] + ([Xt] if with_test else []))
        acc = {e: [0, 0] for e in snaps}
        for s in range(P['seeds']):
            o = train_predict(P, A, y[fi], others, seed=1000 * k + s, snaps=snaps)
            for e in snaps:
                acc[e][0] = acc[e][0] + o[e][0] / P['seeds']
                if with_test:
                    acc[e][1] = acc[e][1] + o[e][1] / P['seeds']
        for e in snaps:
            oof[e][r, vi[el]] = acc[e][0][el]
            if with_test:
                tst[e] = tst[e] + acc[e][1] / len(splits)
    dt = (time.time() - t0) / len(splits)
    res = {}
    for e in snaps:
        aucs = np.array([roc_auc_score(y[~np.isnan(oof[e][r])], oof[e][r][~np.isnan(oof[e][r])]) for r in range(R)])
        res[e] = dict(oof=oof[e], aucs=aucs, auc=float(aucs.mean()), test_foldavg=tst[e] if with_test else None)
        print(f'{which} ep={e:3d} AUC {aucs.mean():.5f} per-rep {np.round(aucs, 4)}  {dt:.1f}s/fold  {json.dumps(P)}'
              + ('  ' + C.tune_gain(y, oof[e]) if which == 'tune' else ''), flush=True)
    return res


GRID = [dict(feats=f, h=h, drop=d, wd=wd, seeds=3, ple=pl)
        for f in ('all',) for pl in (0, 16) for h in ([256, 128],) for d in (0.2, 0.4) for wd in (1e-3, 1e-1)]

if __name__ == '__main__':
    if sys.argv[1] == 'tune':
        for P in GRID:
            cv(P, 'tune', snaps=(10, 20, 40, 80))
    else:
        P = json.loads(sys.argv[2])
        ep = P['epochs']
        res = cv(P, 'std', snaps=(ep,))[ep]
        C.blend_report(y, res['oof'], label='torch_mlp')
        if len(sys.argv) > 3:
            X, Xt = feats(P['feats'])
            A, (At,) = prep(P, X, [Xt])
            tf = np.mean([train_predict(P, A, y, [At], seed=77 + s, snaps=(ep,))[ep][0] for s in range(max(5, P['seeds']))], axis=0)
            print('corr(full, foldavg)', np.corrcoef(tf, res['test_foldavg'])[0, 1].round(4))
            res = dict(res, oof=1 / (1 + np.exp(-res['oof'])), test_foldavg=1 / (1 + np.exp(-res['test_foldavg'])))
            H.save(sys.argv[3], res, 1 / (1 + np.exp(-tf)))   # logits -> probabilities (monotone)
            print('saved', sys.argv[3], flush=True)

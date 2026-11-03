"""ch agent: evaluate the Charlie replica vs final8 (equal-weight rank blend of 8 members), per repeat.
usage: python -I chfull_eval.py {std|conf}"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
WORK = './work'
sys.path.insert(0, WORK)
import harness as H

kind = sys.argv[1]
tr, te, y = H.load()
F8 = ['fkd_lgbm', 'fkt_xgb', 'fkt_cat', 'fkd_et', 'fkd_rf', 'mdl_mlp_ple', 'ch_et_proj', 'ch_lgbm_proj']
if kind == 'conf':
    F8 = ['conf_fkd_lgbm', 'conf_fkt_xgb', 'conf_fkt_cat', 'conf_fkd_et', 'conf_fkd_rf', 'conf_mlp_ple',
          'conf_ch_et', 'conf_ch_lgbm']
pre = 'ch_full_' if kind == 'std' else 'conf_ch_full_'
REP = {t: H.load_res(pre + t) for t in ('cat', 'et', 'blend')}
MEM = {m: H.load_res(m) for m in F8}
R = REP['blend']['oof'].shape[0]


def rank_blend(names_w, r, m):
    s = np.zeros(m.sum())
    for res, w in names_w:
        s += w * rankdata(res['oof'][r, m]) / m.sum()
    return s


rows = {}
base, sp = [], []
for r in range(R):
    m = ~np.isnan(REP['blend']['oof'][r])
    for res in list(MEM.values()) + list(REP.values()):
        assert np.array_equal(m, ~np.isnan(res['oof'][r])), 'different splits'
    b = rank_blend([(MEM[k], 1 / 8) for k in F8], r, m)
    base.append(roc_auc_score(y[m], b))
    sp.append(spearmanr(b, REP['blend']['oof'][r, m])[0])
    combos = {
        '+blend (eq 1/9)': [(MEM[k], 1) for k in F8] + [(REP['blend'], 1)],
        '+cat (eq 1/9)': [(MEM[k], 1) for k in F8] + [(REP['cat'], 1)],
        '+et (eq 1/9)': [(MEM[k], 1) for k in F8] + [(REP['et'], 1)],
        '+cat +et (eq 1/10)': [(MEM[k], 1) for k in F8] + [(REP['cat'], 1), (REP['et'], 1)],
        '+blend (w=0.2, final8 0.8)': [(MEM[k], 0.1) for k in F8] + [(REP['blend'], 0.2)],
    }
    for tag, nw in combos.items():
        rows.setdefault(tag, []).append(roc_auc_score(y[m], rank_blend(nw, r, m)))
base = np.array(base)
print(f'[{kind}] repeats: {R}')
print(f'final8 equal rank blend: {base.mean():.5f} per-rep {np.round(base, 4)}')
print('| model | AUC | per-rep | diff vs final8 (standalone) |')
for t in ('blend', 'cat', 'et'):
    a = REP[t]['aucs']
    print(f'| replica {t} | {a.mean():.5f} | {" ".join(f"{x:.4f}" for x in a)} | {a.mean() - base.mean():+.5f} |')
print(f'mean Spearman(final8 OOF blend, replica blend OOF) over repeats: {np.mean(sp):.4f}')
print('| added to final8 | AUC | diff | pos | per-rep diff |')
for tag, v in rows.items():
    d = np.array(v) - base
    print(f'| {tag} | {np.mean(v):.5f} | {d.mean():+.5f} | {(d > 0).sum()}/{R} | {" ".join(f"{x:+.4f}" for x in d)} |')

if kind == 'std':
    ref = pd.read_csv('./reference/submission_inverted.csv')   # Charlie's file (this branch: reference/)
    assert (ref['ID'].to_numpy() == te['ID'].to_numpy()).all()
    ch = 1 - ref['Popularity_Type'].to_numpy()
    print('\n(c) Spearman with Charlie\'s actual submission (flipped, 1 - p):')
    for t in ('blend', 'cat', 'et'):
        print(f'   replica {t:5s}: test_full {spearmanr(REP[t]["test_full"], ch)[0]:.4f}   '
              f'test_foldavg {spearmanr(REP[t]["test_foldavg"], ch)[0]:.4f}')
    print(f'   replica blend test_full vs test_foldavg: {spearmanr(REP["blend"]["test_full"], REP["blend"]["test_foldavg"])[0]:.4f}')
    try:
        T = np.column_stack([rankdata(MEM[k]['test_full']) for k in F8]).mean(1)
        print(f'   final8 test (equal rank avg of members\' test_full) vs Charlie actual: {spearmanr(T, ch)[0]:.4f}; '
              f'vs replica blend test_full: {spearmanr(T, REP["blend"]["test_full"])[0]:.4f}')
    except Exception as e:
        print('   final8 test reconstruction unavailable:', e)

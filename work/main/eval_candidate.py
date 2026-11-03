"""Standard evaluation of a candidate against the current final blend (final8), on one split set.

python -I eval_candidate.py SET [--add n1,n2,...] [--replace old:new,old2:new2]
  SET = std  (H.make_splits(n_rep=5), seed0=1000)        members: fkd_lgbm fkt_xgb fkt_cat fkd_et fkd_rf mdl_mlp_ple ch_et_proj ch_lgbm_proj
        c9   (H.make_splits(n_rep=3, seed0=9000))         members: conf_* versions
        c11  (H.make_splits(n_rep=3, seed0=11000))        members: c11_* versions   (VERIFICATION ONLY)
Candidates are result names in work/results/ produced on the SAME split set.
Prints standalone AUCs and the per-repeat AUC of the equal-weight rank blend with the candidate added / swapped in,
paired against final8 on the same splits.
"""
import sys
import numpy as np
sys.path.insert(0, './work')
import harness as H
import blend as B

MEMBERS = {
    'std': ['fkd_lgbm', 'fkt_xgb', 'fkt_cat', 'fkd_et', 'fkd_rf', 'mdl_mlp_ple', 'ch_et_proj', 'ch_lgbm_proj'],
    'c9': ['conf_fkd_lgbm', 'conf_fkt_xgb', 'conf_fkt_cat', 'conf_fkd_et', 'conf_fkd_rf', 'conf_mlp_ple',
           'conf_ch_et', 'conf_ch_lgbm'],
    'c11': ['c11_lgbm', 'c11_xgb', 'c11_cat', 'c11_et', 'c11_rf', 'c11_mlp_ple', 'c11_ch_et', 'c11_ch_lgbm'],
}


def main(argv):
    S = argv[1]
    add, rep = [], {}
    for i, a in enumerate(argv):
        if a == '--add':
            add = [x for x in argv[i + 1].split(',') if x]
        if a == '--replace':
            rep = dict(p.split(':') for p in argv[i + 1].split(',') if p)
    tr, te, y = H.load()
    base = MEMBERS[S]
    new = [rep.get(m, m) for m in base] + add
    for n in sorted(set(new) - set(base)):
        print(f'  standalone {n:32s} {H.load_res(n)["aucs"].mean():.5f}')
    a = B.evaluate(base, y, nested=False, verbose=False)['equal']
    b = B.evaluate(new, y, nested=False, verbose=False)['equal']
    d = b - a
    print(f'[{S}] final8 {a.mean():.5f} -> candidate blend {b.mean():.5f}  gain {d.mean():+.5f}  pos {(d > 0).sum()}/{len(d)}  '
          f'per-rep {np.round(d, 4)}')
    return d


if __name__ == '__main__':
    main(sys.argv)

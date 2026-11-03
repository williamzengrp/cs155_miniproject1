"""r3_chp: paired screens on cached folds (vs the replica chrep_bl on the same repeats).
python -I -B chp_screen.py sanity | seed | var V1 [V2 ...]"""
import sys
sys.dont_write_bytecode = True
import numpy as np
sys.path.insert(0, './work/r3_chp')
import chp_lib as L

mode = sys.argv[1]


def ev(name, cat, et, w=0.4, ccorr=None, ecorr=None):
    oof, aucs, tfa, have = L.assemble(lambda k: L.blend(cat, et, k, w, ccorr, ecorr))
    if not have:
        print(name, 'no complete repeats'); return None
    return L.report(name, aucs, have)


def ev_fam(name, srcs, tag, corr=None):
    def fn(k):
        f = L.fam(srcs, tag, k, corr)
        return f
    oof, aucs, tfa, have = L.assemble(fn)
    if not have:
        return None
    ref = L.H.load_res('chrep_cat' if tag == 'cat' else 'chrep_et')['aucs']
    return L.report(name + f' [{tag} vs chrep_{tag}]', aucs, have, ref)


if mode == 'sanity':
    ev('replica recomputed', ['rep'], ['rep'])
elif mode == 'seed':
    SD = sys.argv[2] if len(sys.argv) > 2 else 'sd:500:3000:1'
    ev('replica', ['rep'], ['rep'])
    for it in (1000, 1500, 2000, 2500, 3000):
        ev_fam(f'{SD} cat it{it} (single seed)', [f'{SD}:{it}'], 'cat')
    ev_fam(f'{SD} et (single seed)', [SD], 'et')
    for it in (2000, 2500, 3000):
        ev(f'2-seed cat(rep + {SD}:{it}) et(rep + sd)', ['rep', f'{SD}:{it}'], ['rep', SD])
    ev('2-seed cat only', ['rep', f'{SD}:2000'], ['rep'])
    ev('2-seed et only', ['rep'], ['rep', SD])
    for w in (0.3, 0.35, 0.45, 0.5):
        ev(f'2-seed w_et {w}', ['rep', f'{SD}:2000'], ['rep', SD], w=w)
    ev('2-seed no correction', ['rep', f'{SD}:2000'], ['rep', SD], ccorr=(0, 1, 1), ecorr=(0, 1, 1))
    ev('2-seed cat corr (.2,1.5,1)', ['rep', f'{SD}:2000'], ['rep', SD], ccorr=(.2, 1.5, 1.))
    ev('2-seed cat (.2,1.5,1) et (.35,2,3)', ['rep', f'{SD}:2000'], ['rep', SD], ccorr=(.2, 1.5, 1.), ecorr=(.35, 2., 3.))
elif mode == 'var':
    ev('replica', ['rep'], ['rep'])
    for V in sys.argv[2:]:
        d0 = L._load(f'{L.CACHE}/var_{V}_o0_k0.npz')
        its = sorted(int(x.split('_')[-1]) for x in d0 if x.startswith('cat_v_'))
        has_et = 'et_v' in d0
        E = [f'var:{V}:0'] if has_et else ['rep']
        for it in its:
            ev(f'{V} it{it} blend', [f'var:{V}:0:{it}'], E)
        if not its:
            ev(f'{V} (ET only) blend', ['rep'], E)
            ev_fam(f'{V} et', E, 'et')
            ev(f'{V} blend et no corr', ['rep'], E, ecorr=(0, 1, 1))
            ev(f'{V} 2-et avg (rep+var)', ['rep'], ['rep'] + E)
            continue
        it = 2000 if 2000 in its else its[-1]
        src = f'var:{V}:0:{it}'
        ev_fam(f'{V} cat it{it}', [src], 'cat')
        ev_fam(f'{V} cat it{it} raw', [src], 'cat', corr=(0, 1, 1))
        if has_et:
            ev_fam(f'{V} et', E, 'et')
            ev(f'{V} cat(var)+et(rep)', [src], ['rep'])
        ev(f'{V} blend no corr', [src], E, ccorr=(0, 1, 1), ecorr=(0, 1, 1))
        ev(f'{V} blend cat corr (.2,1.5,1)', [src], E, ccorr=(.2, 1.5, 1.))
        for w in (0.3, 0.5):
            ev(f'{V} blend w_et {w}', [src], E, w=w)
        ev(f'{V} 2-cat avg (rep+var) et(rep+var)', ['rep', src], ['rep'] + E if has_et else ['rep'])

"""r3_chp: save blocks. python -I -B chp_save.py NAME [NAME ...]
NAME in BLOCKS -> results/r3_chp_<NAME>.npz (OOF over the repeats whose 5 folds are all cached; other repeats NaN).
Blocks whose full-refit sources are not all cached are saved as r3_chp_scr_<NAME> with an EMPTY test_full
(screening only: usable by pubcheck, skipped by assess_r3)."""
import sys
sys.dont_write_bytecode = True
import os
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, './work/r3_chp')
import chp_lib as L

SD = 'sd:500:3000:1'
SD2 = 'sd:700:2000:1'
R3 = ['full:0', 'full:1', 'full:2']
BLOCKS = {
    # name: (cv cat srcs, cv et srcs, w_et, cat corr, et corr, full cat srcs, full et srcs)
    'seed2': (['rep', f'{SD}:2000'], ['rep', SD], 0.4, None, None, R3, R3),
    'seed3': (['rep', f'{SD}:2000', f'{SD2}:2000'], ['rep', SD, SD2], 0.4, None, None,
              R3 + ['sd:3:2000:1', 'sd:4:2000:1'], R3 + ['sd:3:2000:1', 'sd:4:2000:1']),
}


def add_var(V, it=2000, seeds=(0,), full_seeds=(0, 1, 2)):
    cat = [f'var:{V}:{s}:{it}' for s in seeds]
    et = [f'var:{V}:{s}' for s in seeds]
    fc = [f'var:{V}:{s}:{it}' for s in full_seeds]
    fe = [f'var:{V}:{s}' for s in full_seeds]
    return (cat, et, 0.4, None, None, fc, fe)


for V in ('lab', 'alb', 'dup05', 'teh', 'copy'):
    BLOCKS[V] = add_var(V)

BLOCKS['labcat'] = (['var:lab:0:2000'], ['rep'], 0.4, None, None, ['var:lab:0:2000', 'var:lab:1:2000', 'var:lab:2:2000'], R3)
BLOCKS['labcatonly'] = (['var:lab:0:2000'], ['rep'], 0.0, None, None, ['var:lab:0:2000', 'var:lab:1:2000', 'var:lab:2:2000'], R3)
BLOCKS['labet'] = (['rep'], ['var:lab:0'], 0.4, None, None, R3, ['var:lab:0', 'var:lab:1', 'var:lab:2'])
BLOCKS['catonly'] = (['rep'], ['rep'], 0.0, None, None, R3, R3)
BLOCKS['seed2catonly'] = (['rep', f'{SD}:2000'], ['rep'], 0.0, None, None, R3, R3)
CC = (0.2, 1.5, 1.0)   # CatBoost residual correction re-tuned on the replica folds (+0.00013, 5/5) and lab reps 0-1
BLOCKS['seed2c'] = (['rep', f'{SD}:2000'], ['rep', SD], 0.4, CC, None, R3, R3)
BLOCKS['labc'] = (['var:lab:0:2000'], ['var:lab:0'], 0.4, CC, None, ['var:lab:0:2000', 'var:lab:1:2000', 'var:lab:2:2000'],
                  ['var:lab:0', 'var:lab:1', 'var:lab:2'])
LF = ['var:lab:0', 'var:lab:1', 'var:lab:2', 'var:lab-tev:3', 'var:lab-tev:4']
BLOCKS['labs2c'] = (['var:lab:0:2000', 'var:lab-tev:500:2000'], ['var:lab:0', 'var:lab-tev:500'], 0.4, CC, None,
                    [f + ':2000' for f in LF], LF)
BLOCKS['labs2'] = (['var:lab:0:2000', 'var:lab-tev:500:2000'], ['var:lab:0', 'var:lab-tev:500'], 0.4, None, None,
                   [f + ':2000' for f in LF], LF)
BLOCKS['albc'] = (['var:alb:0:2000'], ['rep'], 0.4, CC, None, ['var:alb:0:2000', 'var:alb:1:2000', 'var:alb:2:2000'], R3)
AF = ['var:alb:0', 'var:alb:1', 'var:alb:2', 'var:alb-tev:3', 'var:alb-tev:4']
BLOCKS['albs2'] = (['var:alb:0:2000', 'var:alb-tev:500:2000'], ['rep'], 0.4, None, None, [f + ':2000' for f in AF],
                   R3)
BLOCKS['albs2c'] = (['var:alb:0:2000', 'var:alb-tev:500:2000'], ['rep'], 0.4, CC, None, [f + ':2000' for f in AF],
                    R3)
BLOCKS['labalb'] = add_var('lab-alb')
_c, _e, _w, _cc, _ec, _fc, _fe = BLOCKS['labalb']
BLOCKS['labalbc'] = (_c, _e, _w, CC, None, _fc, _fe)
# Charlie-pipeline bag: CatBoost = mean of the album-categorical CatBoosts (2 seeds) and the label-feature CatBoost,
# ExtraTrees = label-feature ET; residual corrections as above (Cat re-tuned CC), w_et 0.4
ALBF = ['var:alb:0:2000', 'var:alb:1:2000', 'var:alb:2:2000', 'var:alb-tev:3:2000', 'var:alb-tev:4:2000']
LABF = ['var:lab:0', 'var:lab:1', 'var:lab:2']
BLOCKS['mix'] = (['var:alb:0:2000', 'var:alb-tev:500:2000', 'var:lab:0:2000'], ['var:lab:0'], 0.4, CC, None,
                 ALBF + [f + ':2000' for f in LABF], LABF)
LABF2 = LABF + ['var:lab-tev:3', 'var:lab-tev:4']
BLOCKS['mix2'] = (['var:alb:0:2000', 'var:alb-tev:500:2000', 'var:lab:0:2000', 'var:lab-tev:500:2000'],
                  ['var:lab:0', 'var:lab-tev:500'], 0.4, CC, None, ALBF + [f + ':2000' for f in LABF2], LABF2)
LAF = ['var:lab-alb:0:2000', 'var:lab-alb:1:2000', 'var:lab-alb:2:2000']
BLOCKS['mix3'] = (['var:alb:0:2000', 'var:alb-tev:500:2000', 'var:lab:0:2000', 'var:lab-tev:500:2000', 'var:lab-alb:0:2000'],
                  ['var:lab:0', 'var:lab-tev:500'], 0.4, CC, None, ALBF + [f + ':2000' for f in LABF2] + LAF, LABF2)
BLOCKS['mixa'] = (['var:alb:0:2000', 'var:alb-tev:500:2000', 'var:lab-alb:0:2000'], ['var:lab:0', 'var:lab-tev:500'], 0.4,
                  CC, None, ALBF + LAF, LABF2)
LAF2 = LAF + ['var:lab-alb-tev:3:2000', 'var:lab-alb-tev:4:2000']
LAE2 = ['var:lab-alb:0', 'var:lab-alb:1', 'var:lab-alb:2', 'var:lab-alb-tev:3', 'var:lab-alb-tev:4']
BLOCKS['labalbs2c'] = (['var:lab-alb:0:2000', 'var:lab-alb-tev:500:2000'], ['var:lab-alb:0', 'var:lab-alb-tev:500'], 0.4, CC,
                       None, LAF2, LAE2)
BLOCKS['mixa2'] = (['var:alb:0:2000', 'var:alb-tev:500:2000', 'var:lab-alb:0:2000', 'var:lab-alb-tev:500:2000'],
                   ['var:lab:0', 'var:lab-tev:500'], 0.4, CC, None, ALBF + LAF2, LABF2)

if __name__ == '__main__':
    ch = L.H.load_res('chrep_bl')['test_full']
    for name in sys.argv[1:]:
        cat, et, w, cc, ec, fc, fe = BLOCKS[name]
        oof, aucs, tfa, have = L.assemble(lambda k: L.blend(cat, et, k, w, cc, ec))
        L.report(name, aucs, have)
        full_ok = len(have) == 5
        if full_ok:
            try:
                import chp_full as F
                tf = F.blend_full(fc, fe, w, cc, ec)
            except FileNotFoundError as e:
                print('  full refit missing:', e)
                full_ok = False
        if full_ok:
            nm = f'r3_chp_{name}'
            L.save(nm, oof, aucs, tfa, tf)
            print(f'  saved {nm}: test_full spearman vs Charlie file {spearmanr(tf, ch)[0]:.5f}  '
                  f'foldavg vs Charlie {spearmanr(tfa, ch)[0]:.5f}', flush=True)
        else:
            nm = f'r3_chp_scr_{name}'
            L.H.save(nm, dict(oof=oof, aucs=aucs, test_foldavg=tfa), None)
            print(f'  saved SCREEN block {nm} (no test_full); foldavg vs Charlie {spearmanr(tfa, ch)[0]:.5f}', flush=True)

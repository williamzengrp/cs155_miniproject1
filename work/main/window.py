"""Submission-window helper: observed public scores by file number -> re-run assess_r3 with them -> build the next
round's files into final_submissions/for_4pm_window/ (R<round>_<nn>_*.csv).
python -I window.py ROUND BATCH NN=score [NN=score ...]   e.g.  window.py 2 4 01=0.8712 02=0.8705 03=0.8698"""
import os, re, subprocess, sys
W = '.'
PY = W + '/venv/bin/python'
OUT = os.environ.get('WIN_OUT', './final_submissions/for_4pm_window/')
BLOCKS = ('r3_aug_lgbm_plganm_2s,r3_aug_lgbm_plgnm_2s,r3_chp_labalbs2c,r3_chp_mixa2,r3_chp_labalb,r3_chg_hc2_w6,'
          'r3_chg_hc2,r3_chx_mix_ca')
NAMES = {'01': 'r3_chp_labalbs2c.6+ch.2+catalb.2', '02': 'r3_chg_hc2_w6.6+ch.2+catalb.2',
         '03': 'r3_aug_lgbm_plganm_2s@0.7+ch0.3', '04': 'ch0.6+zoo_lgb_linear@0.4', '05': 'ch0.5+zoo_cat_alb@0.5',
         '06': 'r3_chx_mix_ca@1.0+ch0.0', '07': 'ch0.6+graph_hc2_cat@0.4', '08': 'r3_chp_labalb@1.0+ch0.0',
         '09': 'ch0.6+te_lgbm_best_v2@0.4', '10': 'r3_chg_hc2_w6@0.7+ch0.3'}
rnd, batch = sys.argv[1], sys.argv[2]
obs = []
for a in sys.argv[3:]:
    k, v = a.split('=')
    k = k if k in NAMES else k.zfill(2)
    obs.append(f'{NAMES[k] if k in NAMES else k}={v}')   # also accepts names printed by a previous round
env = dict(os.environ, R3_BLOCKS=BLOCKS, OMP_NUM_THREADS='4')
r = subprocess.run([PY, '-I', W + '/work/main/assess_r3.py', batch, '431'] + obs, capture_output=True, text=True,
                   env=env, cwd=W + '/work/main')
print(r.stdout[-6000:], r.stderr[-2000:])
picks = re.findall(r'^\s+(\d+)\. (\S+).*\n\s+spec: (\S+)$', r.stdout, flags=re.M)
for i, name, spec in picks:
    fname = f'R{rnd}_{int(i):02d}_' + re.sub(r'[^A-Za-z0-9.+@_-]', '', name)[:80].replace('+', '_').replace('@', '_')
    b = subprocess.run([PY, '-I', W + '/work/main/blendfile.py', fname, spec], capture_output=True, text=True)
    src = W + '/submissions/' + fname + '.csv'
    if os.path.exists(src):
        os.replace(src, OUT + fname + '.csv')
        print('ready:', OUT + fname + '.csv', '| candidate name for next round:', name)
    else:
        print('FAILED to build', name, b.stderr[-500:])

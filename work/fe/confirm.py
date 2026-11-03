"""python confirm.py name1 name2 ... -> runs seed sets 0/100/200 (cached) and compares with full67."""
import sys
sys.path.insert(0, './work/fe')
from configs import *
C = configs()
SEEDS = (0, 100, 200)
ref = sys.argv[1] if sys.argv[1].startswith('ref=') else None
names = sys.argv[2:] if ref else sys.argv[1:]
ref = ref[4:] if ref else 'full67'
def runs(name):
    X, _ = matrix(name, C)
    return [cv_cached(name, X, seed_off=s) for s in SEEDS]
R = runs(ref)
ra = np.mean([r['aucs'] for r in R], axis=0)
for name in names:
    Q = runs(name)
    qa = np.mean([r['aucs'] for r in Q], axis=0)
    dd = qa - ra
    per_seed = [float(np.mean(q['aucs'] - r['aucs'])) for q, r in zip(Q, R)]
    line = (f'| {name} (3 seed sets) | {len(C[name])} | {qa.mean():.5f} | {dd.mean():+.5f} | {(dd > 0).sum()}/5 | '
            f'{" ".join(f"{x:+.4f}" for x in dd)} | vs {ref} {ra.mean():.5f}; per-seed-set means {" ".join(f"{x:+.4f}" for x in per_seed)} |')
    print(line, flush=True); log(line)

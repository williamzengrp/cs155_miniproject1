#!/usr/bin/env python3
"""Reproduce the five-fold 60% CatBoost / 40% Extra Trees submission."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

# Set thread limits before importing numerical libraries.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--threads', type=int, default=8)
parser.add_argument('--out', type=Path, default=Path('output'))
args = parser.parse_args()
if args.threads < 1:
    parser.error('--threads must be positive')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', str(args.threads))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'spotify_day_search/spotify_day_search'))
import tune_project as pipeline
from project_features import make

config = json.loads((ROOT / 'reference/model_config.json').read_text())
train = pd.read_csv(ROOT / 'data/train.csv')
test = pd.read_csv(ROOT / 'data/test.csv')
train = train.loc[train.Popularity_Type.notna()].reset_index(drop=True)
for frame in (train, test):
    frame['uri'] = frame.uri.astype('string').str.strip().replace('', pd.NA)
y = train.Popularity_Type.eq('High').to_numpy().astype(int)
groups = pipeline.connected_groups(train)
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=20261171).split(train, y, groups))
for fit, valid in folds:
    if set(groups[fit]) & set(groups[valid]):
        raise ValueError('Duplicate group crosses a fold')
extra = make(train.drop(columns='Popularity_Type'), test.drop(columns='Popularity_Type', errors='ignore'))
features = pipeline.Features(train, test, True, extra, replace_features=True)
features.groups = groups
out = args.out.resolve()
out.mkdir(parents=True, exist_ok=True)
# Cache is tied to data, code, model config, thread count, and package versions.
digest = hashlib.sha256()
for file in [ROOT / 'data/train.csv', ROOT / 'data/test.csv', ROOT / 'reference/model_config.json',
             Path(__file__), ROOT / 'spotify_day_search/spotify_day_search/tune_project.py',
             ROOT / 'spotify_day_search/spotify_day_search/project_features.py',
             ROOT / 'experiments/similarity_study/similarity.py', ROOT / 'experiments/pdf_idea/date_pool.py',
             ROOT / 'experiments/research85/target_encode.py', ROOT / 'experiments/preprocessing.py']:
    digest.update(file.read_bytes())
from importlib.metadata import version
identity = {'sha256': digest.hexdigest(), 'threads': args.threads,
            'packages': {name:version(name) for name in ('numpy','pandas','scipy','scikit-learn','catboost')}}
manifest_path = out / 'manifest.json'
if manifest_path.exists() and json.loads(manifest_path.read_text()) != identity:
    raise SystemExit('Output directory belongs to a different setup; choose a new --out.')
pipeline.atomic_json(manifest_path, identity)
results = {}
for family in ('cat', 'extra'):
    oof = np.full(len(train), np.nan)
    tests = []
    for k, (fit, valid) in enumerate(folds):
        path = out / f'{family}_fold{k+1}.npz'
        if path.exists():
            with np.load(path) as saved:
                pred, test_pred = saved['valid'], saved['test']
            print(f'Loaded {family} fold {k+1}/5', flush=True)
        else:
            print(f'Training {family} fold {k+1}/5', flush=True)
            predictions, test_pred, _, _ = pipeline.cv_run(
                config[family], [(fit, valid)], features, train, y, args.threads,
                20261171 + k, float('inf'), date_target_encoding=True)
            pred = predictions[valid]
            temp = path.with_suffix('.tmp.npz')
            np.savez_compressed(temp, valid=pred, test=test_pred)
            os.replace(temp, path)
        oof[valid] = pred
        tests.append(test_pred)
    results[family] = (oof, np.mean(tests, axis=0))
w = config['extra_weight']
oof = (1-w)*results['cat'][0] + w*results['extra'][0]
pred = (1-w)*results['cat'][1] + w*results['extra'][1]
if len(pred) != len(test) or not np.isfinite(pred).all() or np.any((pred < 0) | (pred > 1)):
    raise ValueError('Invalid test predictions')
for label, values in [('positive', pred), ('inverted', 1-pred)]:
    pd.DataFrame({'ID': test.ID, 'Popularity_Type': values}).to_csv(out / f'submission_{label}.csv', index=False)
fold_scores = [float(roc_auc_score(y[v], oof[v])) for _, v in folds]
reference = pd.read_csv(ROOT / 'reference/submission_inverted.csv')
if not reference.ID.equals(test.ID):
    raise ValueError('Reference IDs differ from test IDs')
max_diff = float(np.max(np.abs(reference.Popularity_Type.to_numpy() - (1-pred))))
report = {'mean_fold_auc': float(np.mean(fold_scores)), 'fold_auc': fold_scores,
          'max_absolute_difference_from_reference': max_diff,
          'note': 'Selection used these labels; test features were pooled without labels.'}
pipeline.atomic_json(out / 'metrics.json', report)
print(json.dumps(report, indent=2))

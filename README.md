# CS155 miniproject 1: Spotify popularity

This package reproduces the project's best observed submission: a five-fold ensemble of **60% CatBoost and 40% Extra Trees**. The saved inverted submission received **0.13113 AUC** on the public leaderboard (equivalent to 0.86887 AUC for the corresponding non-inverted High scores, for the same rows and labels). The five-fold mean validation AUC for the High scores was **0.86234**. The validation result is selection-biased: the same training labels informed earlier model and feature choices.

## Setup

The reproduced run used Linux x86-64, Python **3.14.8**, 8 CPU threads, an Intel Core Ultra 5 125H, and 30 GiB of system RAM. It did not use a GPU. These describe the machine used, not minimum hardware requirements. The exact Python package versions are in `requirements.txt`. Numerical results may differ across package versions, platforms, and hardware.

From this directory, with Python 3.14 available:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 python -u reproduce.py --threads 8 --out output
```

The script runs on CPU. Training and the inner models used for residual correction can take several minutes; time depends on the machine. It checkpoints each completed model fold in `output/`. If interrupted, run the same command again to reuse those folds. Use a new output directory if changing the data, model code, dependencies, or thread count.

The input files are provided at `data/train.csv` and `data/test.csv`. Their SHA-256 hashes are:

```text
d14b372deff47aa791242875ac5c134ce2497695bb190479b159b8ec49d68407  data/train.csv
70f1502ccae76396b45961a0e0c9f96c5ba213d5b99a101015c32ddc65d020e3  data/test.csv
```

The output files are:

- `output/submission_inverted.csv`: the leaderboard submission format, with `ID` and `Popularity_Type` columns.
- `output/submission_positive.csv`: corresponding High-class scores before inversion.
- `output/metrics.json`: fold AUCs and the largest absolute difference from the included reference submission.
- `reference/submission_inverted.csv`: the exact previously submitted file, included for comparison. Its SHA-256 is `33f5be1734f2fe579814f6fa349cbbf5613a470c2dfd17168603ed9ced738756`.

The reference file can be inspected immediately; executing the script retrains the ten outer-fold models rather than loading fitted models. The output comparison reports numerical differences instead of assuming identical results across environments.

## How the model works

The training data has 3,864 labeled rows; the test data has 863 rows with no labels. The core inputs include 13 numerical Spotify audio attributes and release-date information. The pipeline derives calendar features, same-date song counts, audio neighborhood statistics, and music-category features. It also builds smoothed target encodings for release year, month, and exact date.

The label-free feature builder **pools training and test covariates** to compute some neighborhood, density, and scaling features. No test labels are used. Check whether this use of test covariates is permitted under the applicable project rules. Missing audio is imputed for model fitting; a row with all audio missing is excluded as an audio neighbor.

Rows sharing a Spotify URI or an identical complete audio vector are grouped. `StratifiedGroupKFold` makes five folds using split seed `20261171`; each song group remains in one fold. The year/month/date target encodings are fitted on the training portion of each outer fold, with grouped cross-fitting for its training rows. The residual correction also uses grouped inner cross-fitting, with encodings rebuilt inside each inner training split.

For each fold, one CatBoost classifier and one Extra Trees classifier train on the other four folds. Their out-of-fold scores provide the validation AUC. Each model also predicts every test row. The five CatBoost test predictions are averaged, as are the five Extra Trees predictions, then the two averages are combined:

```text
High score = 0.60 × mean(CatBoost fold scores) + 0.40 × mean(Extra Trees fold scores)
submission score = 1 − High score
```

Both components include a same-release-date residual correction based on similar songs. For Extra Trees, copies of a song are downweighted by inverse occurrence count; CatBoost uses equal row weights. These are fold models trained on roughly 80% of rows each, not models refitted on the full training set. The project instructions describe High as class 1; the inverted file is included because the submitted objective was to minimize the AUC of inverted scores. Use the positive file if the evaluation expects larger scores for High.

The exact configurations are in `reference/model_config.json`. Key settings:

| Setting | CatBoost | Extra Trees |
|---|---:|---:|
| Trees | 2,000 | 900 |
| Depth | 9 | 15 |
| Learning rate | 0.0088846 | — |
| L2 | 9.8833 | — |
| Feature subsampling | 0.4717 | 0.1827 |
| Minimum leaf size | — | 2 |
| Residual strength | 0.35 | 0.15 |
| Duplicate weight power | 0 | 1 |
| Blend weight | 0.60 | 0.40 |

## Files

- `reproduce.py`: runs the folds, blend, score inversion, and reference comparison.
- `spotify_day_search/spotify_day_search/`: model training, grouped splitting, and feature adapter.
- `experiments/`: feature construction and grouped target-encoding code used by the model.
- `data/`: provided train and test CSVs.
- `reference/`: model configuration, historical validation results, and submitted prediction file.

No Spotify links or external data are needed to run the pipeline.

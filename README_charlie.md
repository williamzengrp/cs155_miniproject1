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

### CatBoost and Extra Trees

A decision tree predicts by asking a sequence of questions about the inputs, such as whether energy is above a threshold or whether the release was before a given year. A leaf at the end of the path supplies a prediction.

**CatBoost** is a gradient-boosted tree model. It builds trees sequentially: each new tree adjusts the combined prediction to reduce the training loss. In this project the loss is binary log loss, while models are evaluated using ROC AUC. The reference uses 2,000 trees, depth 9, and a small learning rate of approximately 0.008885. L2 regularization and feature subsampling control how the trees are fitted. CatBoost outputs a score for the High class.

**Extra Trees** fits many trees independently. At each split it considers a randomly selected subset of features and randomized candidate thresholds, then selects a useful split from those candidates. It averages its trees' class probabilities. The reference uses 900 trees, maximum depth 15, and a minimum of two training rows per leaf. Scikit-learn's `ExtraTreesClassifier` uses `bootstrap=False` here: each tree is fitted using the fold's training rows with the configured sample weights. This randomness is separate from the five-fold ensemble.

CatBoost and Extra Trees can make different ranking errors, so the submission averages their corrected predictions at weights 0.60 and 0.40. The 0.60/0.40 split was selected through experiments on the training dataset; it is not a learned probability or a guarantee that the blend always improves AUC.

### The 121 input features

Each model receives **115 label-free features plus six target-encoding features**. The list below describes every input by its exact name or by an explicitly expanded naming pattern. The two models receive the same columns.

1. **13 audio and music inputs:** `time_signature`, `speechiness`, `danceability`, `duration_ms`, `energy`, `mode`, `instrumentalness`, `valence`, `key`, `tempo`, `loudness`, `acousticness`, and `liveness`.

2. **19 calendar inputs:** `release_year`, `release_month`, `release_day`, `weekday`, `day_of_year`, `release_quarter`, `fractional_year`, `decade`, `is_jan1`, `is_friday`, `is_tuesday`, `is_summer`, `is_holiday`, `month_sin`, `month_cos`, `weekday_sin`, `weekday_cos`, `day_of_year_sin`, and `day_of_year_cos`. The sine/cosine pairs represent cyclical positions, such as December being close to January. Missing or partial release dates have unavailable calendar components represented as missing.

3. **Four release-density inputs:** `same_year_count`, `same_month_count`, `same_date_count`, and `same_date_tempo_count`. These count rows in the pooled train/test feature data sharing the corresponding date component; the tempo count uses tempo rounded to the nearest integer. Counts do not use labels and can include repeated rows. For an applicable date, the queried row is included in its own count.

4. **Five same-date audio-neighbor inputs:** `near_count_1`, `near_count_1_5`, and `near_count_2` count other rows on the same complete release date within standardized audio distances of 1, 1.5, and 2, respectively. `nearest_same_date` and `second_nearest_same_date` are the distances to the closest and second-closest same-date rows. Distances use the 13 median-imputed, standardized audio/music inputs; the row itself is excluded. Repeated rows may still be neighbors. A row with all audio missing is not used as an audio neighbor.

5. **26 same-date audio summary inputs:** for each of the 13 names in item 1, the model receives `same_date_mean_<name>` and `same_date_deviation_<name>`. The first is the mean value among rows with that complete release date; the second is the song's value minus that mean. For example, `same_date_deviation_energy` measures how far the song's energy is from its date's mean. The mean includes the row itself and uses the pooled train/test feature rows.

6. **Two additional numeric inputs:** `loudness_energy_residual` is actual loudness minus the value predicted by a straight-line fit from energy, using available pooled train/test feature values. `year_density_percentile` is the percentile rank of `same_date_count` among rows from the same release year.

7. **45 music-category indicators:** `key_is_0` through `key_is_11` (12 columns), `mode_is_0` and `mode_is_1` (2), `time_signature_is_1` through `time_signature_is_7` (7), and all 24 `key_<k>_mode_<m>` combinations for `k` from 0 through 11 and `m` in `{0, 1}`. An indicator is 1 when its category matches and 0 otherwise; missing source values remain missing.

8. **One cyclic key-neighbor input:** `nearest_same_date_cyclic_key` is the smallest combined production-and-key distance to another same-date row with a known key. It combines standardized differences in energy, loudness, acousticness, and log-transformed instrumentalness with the shorter key distance around the 12-key circle, divided by 3. Thus keys 0 and 11 are one step apart in the key component.

9. **Six grouped date-target inputs:** `date_te_year`, `date_te_month`, and `date_te_exact` are smoothed estimates of the High-label fraction for the song's release year, year/month, and exact date. Each uses the training-fold High fraction as a prior with smoothing strength 2. `date_count_year`, `date_count_month`, and `date_count_exact` are `log(1 + count)` for those same groups. The counts and estimates use only the relevant labeled training partition. Training-row values are group-cross-fitted, so a song and its grouped duplicates cannot contribute their labels to their own encoding.

The counts add up to `13 + 19 + 4 + 5 + 26 + 2 + 45 + 1 + 6 = 121`. The feature adapter selects the base and `secondary` groups from `similarity.py`. Other experimental groups calculated in that file, such as cluster, cross-date, and tempo-ambiguity features, are **not** passed to this reference model. Rows with missing numeric inputs are median-imputed using the relevant training fold before model fitting.

### Residual correction

A **prediction residual** is the observed label minus a model's predicted High score. For example, if a song's label is 1 and the model predicted 0.7, its residual is +0.3: the model scored that song lower than its label. A negative residual means the score was higher than the label. These residuals are distinct from the `loudness_energy_residual` input described above.

For each outer fold and each model family, the pipeline:

1. Splits that fold's training rows into three **grouped inner folds**. It trains on two inner folds and predicts the third, repeating until every outer-training row has a prediction from an inner model that did not train on that row's song group. Date target encodings are rebuilt inside each inner fit.
2. Calculates `label - inner_prediction` for those outer-training rows. It averages residuals for repeated copies of a URI so one URI contributes one reference residual. Rows without a URI or usable audio are excluded from the correction reference.
3. For a validation or test song with a complete release date and usable audio, finds reference songs on the **same exact date**. It median-imputes and standardizes the 13 audio/music columns using the correction reference, then measures Euclidean audio distance. A neighbor at distance `d` receives weight `exp(-0.5 × (d / bandwidth)²)`.
4. Computes `sum(weight × residual) / (shrinkage + sum(weight))`, multiplies by the correction strength, adds it to the model's High score, and clips the result to `[0, 1]`. With no eligible neighbors, the correction is zero.

The CatBoost correction uses bandwidth **0.75**, shrinkage **10**, and strength **0.35**. Extra Trees uses bandwidth **3**, shrinkage **3**, and strength **0.15**. Larger shrinkage pulls corrections toward zero when little neighbor evidence is available. The two corrected model scores are then blended as described above. Similar same-date audio is an approximation; it does not establish that tracks share an album.


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

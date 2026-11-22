# Detecting Fake Reviews by Combining Text, Behavior and Graph Signals

IT4868E Web Mining — capstone, **Method** project. Checkpoint scope: a clean
pipeline over YelpChi plus the first baselines (E1, E2, E4) and the two
leakage demos. GNNs come after the checkpoint.

**Task.** One review is one example. `label = 1` if Yelp's filter flagged it.
Inputs: text, reviewer ID, product ID, star, date. The `useful`, `funny`,
`cool` votes are **never** inputs — they leak the label (see below).

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python src/download.py
```

```bash
python src/load_data.py && python src/splits.py
```

```bash
python src/features/behavior.py && python src/features/text.py && python src/graph/build_edges.py
```

```bash
python src/eda.py
```

```bash
python scripts/run_baselines.py --seeds 42 43 44 45 46 && python scripts/summarize.py
```

```bash
python -m pytest tests/ -q
```

A single-seed smoke run, if you just want numbers quickly:

```bash
python scripts/run_baselines.py --seeds 42 --only E2-TREE Leak-1 Leak-2
```

## What the pipeline produces

| Artifact | Built by | Contents |
|---|---|---|
| `data/processed/reviews.parquet` | `src/load_data.py` | 67,395 reviews, one row per review, `row_id` 0..67394 |
| `splits/*.json` | `src/splits.py` | `row_id` lists per split, committed; nobody re-splits in a notebook |
| `data/processed/behavior.parquet` | `src/features/behavior.py` | 19 behavior features |
| `data/processed/text_stats.parquet` | `src/features/text.py` | 11 style features |
| `data/processed/edges.npz` | `src/graph/build_edges.py` | R-U-R / R-T-R / R-S-R edge lists |
| `results/eda_report.md` + `figures/*.png` | `src/eda.py` | every number on slides 5–7, three figures |
| `results/runs.csv` | `scripts/run_baselines.py` | one row per (run, seed) |
| `results/summary.md` | `scripts/summarize.py` | slide-11 table, mean ± std, hypothesis read-out |
| `docs/experiments.md` | — | mô tả từng thí nghiệm: input → khởi tạo → output |

## Verified data facts

All reproduced by `pytest` and `src/eda.py`:

- 67,395 reviews · 8,919 fake (13.2%) · 38,063 reviewers · 201 products
  (72 hotels / 5,854 reviews, 129 restaurants / 61,541 reviews) · Oct 2004 – Oct 2012.
- Every meta line has exactly 9 fields and every review file line parses; text
  and meta are aligned by line number, so blank lines are **not** dropped.
- Edge counts: R-U-R 110,112 · R-T-R 466,352 · R-S-R 8,166,840.
  Label homophily 0.997 / 0.792 / 0.804.

## The two leaks, and what we do about them

1. **Votes leak the label.** 0 of 8,919 flagged reviews carry any
   useful/funny/cool vote, against 49.9% of genuine reviews. Dropping them is
   enforced by an assert inside every feature builder, and `Leak-1` measures
   what including them would buy.
2. **Labels are nearly per-reviewer.** Only 135 of 38,063 reviewers carry both
   labels, and two reviews by the same reviewer share a label 99.7% of the
   time. A random split therefore lets reviewer features carry the label from
   train to test. The main split is **user-disjoint** (60/20/20 by reviewer);
   `Leak-2` measures the inflation a random split gives.

## Splits

| Split | Design | Sizes (train/val/test) | Used for |
|---|---|---|---|
| `user_disjoint` | 60/20/20 by **reviewer**, stratified on whether the reviewer ever wrote a fake | 40,258 / 13,667 / 13,470 | main result |
| `random` | 40/20/40 by review, as in the papers | 26,958 / 13,479 / 26,958 | H3 comparison |
| `time` | train ≤ 2010, test ≥ 2011 | 32,663 / 5,765 / 28,967 | robustness |
| `domain` | restaurants → hotels, hotel reviewers seen in training removed | 52,309 / 9,232 / 4,506 | RQ4 early check |

Seeds 42–46 of the user-disjoint split are committed as
`splits/user_disjoint_seed4*.json`.

## Metrics

AUC-ROC, average precision, macro-F1 (threshold tuned on **validation**), and
precision@top-1%. Not accuracy — predicting all-genuine already scores ~86.8%.
Every metric is also reported separately for one-review and repeat reviewers,
because 76% of fakes come from one-review accounts.

## Notes on the implementation

- Seed 42 everywhere; `--seeds` controls the repeats.
- TF-IDF, the scaler and the F1 threshold are fitted on the **train split
  only**. For the stacked run E4 the train-split text score is
  **out-of-fold** (5-fold, vectorizers refitted per fold), so the tree never
  sees a text score fitted on its own label.
- The tree model is XGBoost (3.4.1) when it is installed, otherwise
  scikit-learn's `HistGradientBoostingClassifier`. `scripts/summarize.py`
  prints which one produced the numbers, so the table never claims XGBoost
  when it did not run. The two agree closely (0.787 vs 0.784 AUC on E2).
- TF-IDF scores are cached per (split, seed) under
  `data/processed/text_cache/`, keyed to a fingerprint of the split file.
  E1a and E4 fit the identical text model and E4's out-of-fold pass is five
  more fits, so a re-run that only changes the tree takes about a minute
  instead of half an hour. Editing a split file invalidates the cache.

## After the checkpoint

DistilRoBERTa/RoBERTa text embeddings · unsupervised link analysis (FraudEagle,
SpEagle, Fraudar via UGFraud) · the proposed relation-aware GNN with ego
separation and top-k filtered product neighbors · CARE-GNN / PC-GNN / BWGNN
via GADBench · restaurant → hotel transfer (E8) and the DGL reproduction (E9).

"""Metrics and the run log.

Not accuracy: an all-genuine model scores ~86.8% on this data. We report
AUC-ROC, average precision, macro-F1 at a threshold tuned on validation, and
precision@top-1%, each also split by one-review vs repeat reviewers.
"""
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from load_data import ROOT

RUNS = ROOT / "results/runs.csv"
FIELDS = ["exp", "hypothesis", "model", "features", "split", "seed",
          "auc", "ap", "macro_f1", "p_at_1pct", "auc_singleton", "auc_repeat",
          "n_train", "n_test", "threshold", "date"]


def best_threshold(y_val, p_val):
    grid = np.linspace(0.05, 0.95, 91)
    return float(max(grid, key=lambda t: f1_score(y_val, p_val >= t, average="macro")))


def precision_at(y, p, frac=0.01):
    k = max(1, int(len(p) * frac))
    return float(np.asarray(y)[np.argsort(-np.asarray(p))[:k]].mean())


def _safe_auc(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def evaluate(y_val, p_val, y_test, p_test, singleton_test=None):
    """singleton_test: boolean array marking test reviews from one-review accounts."""
    t = best_threshold(y_val, p_val)
    y_test, p_test = np.asarray(y_test), np.asarray(p_test)
    m = dict(auc=_safe_auc(y_test, p_test),
             ap=float(average_precision_score(y_test, p_test)),
             macro_f1=float(f1_score(y_test, p_test >= t, average="macro")),
             p_at_1pct=precision_at(y_test, p_test),
             threshold=round(t, 3))
    if singleton_test is not None:
        s = np.asarray(singleton_test).astype(bool)
        m["auc_singleton"] = _safe_auc(y_test[s], p_test[s])
        m["auc_repeat"] = _safe_auc(y_test[~s], p_test[~s])
    return m


def log_run(row, path=RUNS):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {**{k: "" for k in FIELDS}, **row}
    row["date"] = row["date"] or datetime.now().isoformat(timespec="seconds")
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row[k] for k in FIELDS})
    return row

"""Behavior features, from `date, user_id, product_id, star` only.

No labels and no vote columns go in, so these can be computed once on the whole
dataset without leaking anything across the split boundary.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from load_data import ROOT, load_processed

OUT = ROOT / "data/processed/behavior.parquet"
LEAK = ["useful", "funny", "cool"]


def _star_entropy(s):
    p = s.value_counts(normalize=True)
    return float(-(p * np.log2(p)).sum())


def behavior_features(df):
    d = df[["row_id", "date", "user_id", "product_id", "star"]].copy()
    d["extreme"] = d.star.isin([1, 5]).astype(int)

    # --- product context ---
    p = d.groupby("product_id")
    d["p_mean_star"] = p.star.transform("mean")
    d["p_n"] = p.star.transform("size")
    d["dev"] = d.star - d.p_mean_star
    d["abs_dev"] = d.dev.abs()
    d["rank_pct"] = p.date.rank(method="first") / d.p_n
    d["prod_same_day"] = d.groupby(["product_id", "date"]).star.transform("size")
    d["p_star_entropy"] = d.product_id.map(p.star.apply(_star_entropy))

    # --- reviewer behavior ---
    g = d.groupby("user_id")
    d["u_n"] = g.star.transform("size")
    d["u_singleton"] = (d.u_n == 1).astype(int)
    d["u_max_per_day"] = (d.groupby(["user_id", "date"]).star.transform("size")
                          .groupby(d.user_id).transform("max"))
    d["u_pos_ratio"] = g.star.transform(lambda s: (s >= 4).mean())
    d["u_neg_ratio"] = g.star.transform(lambda s: (s <= 2).mean())
    d["u_mean_abs_dev"] = g.abs_dev.transform("mean")
    d["u_extreme_ratio"] = g.extreme.transform("mean")
    d["u_active_days"] = (g.date.transform("max") - g.date.transform("min")).dt.days
    d["u_mean_star"] = g.star.transform("mean")
    d["u_std_star"] = g.star.transform("std").fillna(0)

    d = d.drop(columns=["date", "user_id", "product_id"])
    assert not set(d.columns) & set(LEAK), "vote column leaked into behavior features"
    return d


FEATURES = None   # filled on import below


if __name__ == "__main__":
    df = load_processed()
    b = behavior_features(df)
    b.to_parquet(OUT, index=False)
    cols = [c for c in b.columns if c != "row_id"]
    print(f"wrote {OUT}  rows={len(b):,}  features={len(cols)}")
    print("  " + ", ".join(cols))
    assert (b.row_id.to_numpy() == df.row_id.to_numpy()).all(), "row order changed"
    assert b.notna().all().all(), b.isna().sum()[lambda s: s > 0]
    print("  row order matches parquet, no NaNs")

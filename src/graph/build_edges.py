"""Build the three review-review relations and report edge homophily.

Edges link reviews (`row_id`) that share a group:
    R-U-R  same reviewer
    R-T-R  same product, same month
    R-S-R  same product, same star rating

Homophily = share of edges whose two endpoints carry the same label. It is
computed from group counts, so we never need the materialised edge list:
a group of n reviews with n1 fakes contributes C(n,2) edges, of which
C(n1,2) + C(n-n1,2) are same-label.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from load_data import ROOT, load_processed

OUT = ROOT / "data/processed/edges.npz"

RELATIONS = {
    "RUR": ["user_id"],
    "RTR": ["product_id", "month"],
    "RSR": ["product_id", "star"],
}


def pairs_by_group(df, keys):
    src, dst = [], []
    for _, idx in df.groupby(keys, observed=True).row_id:
        a = idx.to_numpy()
        if len(a) < 2:
            continue
        i, j = np.triu_indices(len(a), k=1)
        src.append(a[i])
        dst.append(a[j])
    return (np.concatenate(src).astype(np.int32),
            np.concatenate(dst).astype(np.int32))


def homophily(df, keys):
    """(n_edges, same-label share) without materialising the edges."""
    c2 = lambda n: n * (n - 1) // 2
    g = df.groupby(keys, observed=True).label.agg(["size", "sum"])
    n, n1 = g["size"].to_numpy(dtype=np.int64), g["sum"].to_numpy(dtype=np.int64)
    total = c2(n).sum()
    same = (c2(n1) + c2(n - n1)).sum()
    return int(total), float(same / total)


def neighbor_fake_rate(df, keys):
    """Mean share of FAKE neighbors for a fake review, under this relation.
    This is the camouflage number: low means fakes sit among genuine reviews."""
    g = df.groupby(keys, observed=True).label.agg(["size", "sum"])
    d = df.join(g, on=keys)
    d = d[(d["size"] > 1) & (d.label == 1)]
    # exclude the review itself from its own neighborhood
    return float(((d["sum"] - 1) / (d["size"] - 1)).mean())


def build(df, save=True):
    df = df.copy()
    df["month"] = df.date.dt.to_period("M").astype(str)
    out, report = {}, {}
    for name, keys in RELATIONS.items():
        n_edges, hom = homophily(df, keys)
        report[name] = dict(edges=n_edges, homophily=hom)
        if name != "RUR":
            report[name]["fake_neighbor_share_of_fakes"] = neighbor_fake_rate(df, keys)
        if save:
            s, d = pairs_by_group(df, keys)
            assert len(s) == n_edges, (name, len(s), n_edges)
            out[f"{name}_src"], out[f"{name}_dst"] = s, d
    if save:
        np.savez_compressed(OUT, **out)
    return report


if __name__ == "__main__":
    rep = build(load_processed())
    print(f"edges -> {OUT}\n")
    for name, r in rep.items():
        extra = r.get("fake_neighbor_share_of_fakes")
        extra = "" if extra is None else f"   fake neighbors of a fake: {extra:.1%}"
        print(f"  {name}  edges={r['edges']:>9,}   homophily={r['homophily']:.3f}{extra}")

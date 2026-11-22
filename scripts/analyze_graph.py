"""Does the review graph actually carry usable label signal under the
user-disjoint split?

Three questions the GNN plan depends on:
  1. Is each relation's homophily above what random labels would give?
  2. Do its edges cross the train/test boundary at all? (R-U-R connects reviews
     of the SAME reviewer, and the main split puts a reviewer entirely in one
     part -- so those edges may never reach a labeled neighbor at test time.)
  3. Used alone, how well does "share of my labeled neighbors that are fake"
     score test reviews? That is the E3 graph-only baseline.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
from load_data import ROOT, load_processed   # noqa: E402

RELS = ["RUR", "RTR", "RSR"]


def neighbor_stats(src, dst, label, from_mask, n):
    """For every node: how many neighbors lie in `from_mask`, and how many of
    those are fake. Both edge directions are counted."""
    cnt = np.zeros(n); pos = np.zeros(n)
    for a, b in ((src, dst), (dst, src)):
        keep = from_mask[b]
        cnt += np.bincount(a[keep], minlength=n)
        pos += np.bincount(a[keep], weights=label[b][keep].astype(float), minlength=n)
    return cnt, pos


def main():
    df = load_processed()
    n = len(df)
    label = df.label.to_numpy()
    p = label.mean()
    baseline = p ** 2 + (1 - p) ** 2

    z = np.load(ROOT / "data/processed/edges.npz")
    sp = json.loads((ROOT / "splits/user_disjoint.json").read_text())
    tr = np.zeros(n, bool); te = np.zeros(n, bool)
    tr[sp["train"]] = True
    te[sp["test"]] = True
    y_te = label[te]

    print(f"fake rate {p:.3f}  ->  homophily of RANDOM labels = {baseline:.3f}")
    print(f"user-disjoint: train {tr.sum():,}  test {te.sum():,}\n")

    hdr = (f"{'rel':<5}{'homophily':>10}{'vs random':>11}{'edges':>11}"
           f"{'cross tr-te':>13}{'test w/ lab nb':>16}{'AUC':>8}{'AP':>8}")
    print(hdr); print("-" * len(hdr))

    for r in RELS:
        s, d = z[f"{r}_src"].astype(np.int64), z[f"{r}_dst"].astype(np.int64)
        same = (label[s] == label[d]).mean()
        cross = ((tr[s] & te[d]) | (te[s] & tr[d])).sum()

        cnt, pos = neighbor_stats(s, d, label, tr, n)       # labeled = train only
        has = cnt[te] > 0
        score = np.divide(pos[te], cnt[te], out=np.full(te.sum(), p), where=cnt[te] > 0)
        auc = roc_auc_score(y_te, score) if len(np.unique(y_te)) > 1 else float("nan")
        ap = average_precision_score(y_te, score)
        print(f"{r:<5}{same:>10.3f}{same - baseline:>+11.3f}{len(s):>11,}"
              f"{cross:>13,}{has.mean():>15.1%}{auc:>8.3f}{ap:>8.3f}")

    # all relations pooled
    cnt = np.zeros(n); pos = np.zeros(n)
    for r in RELS:
        c, q = neighbor_stats(z[f"{r}_src"].astype(np.int64),
                              z[f"{r}_dst"].astype(np.int64), label, tr, n)
        cnt += c; pos += q
    score = np.divide(pos[te], cnt[te], out=np.full(te.sum(), p), where=cnt[te] > 0)
    print(f"\nE3 (all 3 relations pooled): AUC {roc_auc_score(y_te, score):.3f}  "
          f"AP {average_precision_score(y_te, score):.3f}  "
          f"test reviews with >=1 labeled neighbor: {(cnt[te] > 0).mean():.1%}")

    compare_splits(df, z, label, p, n)


def compare_splits(df, z, label, p, n):
    """The same graph-only score under both splits.

    R-U-R connects reviews by the same reviewer and same-reviewer reviews share
    a label 99.7% of the time. A RANDOM split therefore puts a labeled twin of
    many test reviews into training, and simply copying a same-user neighbor's
    label becomes a strong classifier. The user-disjoint split removes that
    entirely. This is our explanation for the high published GNN numbers on
    YelpChi, and E9 should report it.
    """
    print(f"\n{'split':<16}{'rel':<6}{'test w/ labeled nb':>20}{'AUC':>8}{'AP':>8}")
    print("-" * 58)
    for split in ("user_disjoint", "random"):
        sp = json.loads((ROOT / f"splits/{split}.json").read_text())
        tr = np.zeros(n, bool); te = np.zeros(n, bool)
        tr[sp["train"]] = True; te[sp["test"]] = True
        y = label[te]
        for r in RELS:
            s, d = z[f"{r}_src"].astype(np.int64), z[f"{r}_dst"].astype(np.int64)
            cnt, pos = neighbor_stats(s, d, label, tr, n)
            sc = np.divide(pos[te], cnt[te], out=np.full(te.sum(), p), where=cnt[te] > 0)
            print(f"{split:<16}{r:<6}{(cnt[te] > 0).mean():>19.1%}"
                  f"{roc_auc_score(y, sc):>8.3f}{average_precision_score(y, sc):>8.3f}")
        print()


if __name__ == "__main__":
    main()

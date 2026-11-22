"""EDA: reproduce every number the checkpoint report and slides claim, and
write the three figures. Prints a markdown report to stdout and to
results/eda_report.md.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import ROOT, load_processed
from graph.build_edges import build as build_graph_report

FIG = ROOT / "figures"
FAKE, GEN = "#d1495b", "#8d99ae"      # one accent for fake, neutral for genuine


def fmt(x, pct=False, nd=1):
    if pct:
        return f"{100 * x:.{nd}f}%"
    return f"{x:,.2f}" if isinstance(x, float) else f"{x:,}"


def table(rows, head):
    w = [max(len(str(r[i])) for r in [head] + rows) for i in range(len(head))]
    line = lambda r: "| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |"
    return "\n".join([line(head), "|" + "|".join("-" * (x + 2) for x in w) + "|"]
                     + [line(r) for r in rows])


def main():
    df = load_processed()
    df["n_words"] = df.text.str.split().str.len()
    df["extreme"] = df.star.isin([1, 5])
    df["has_vote"] = (df[["useful", "funny", "cool"]].sum(axis=1) > 0)
    df["p_mean_star"] = df.groupby("product_id").star.transform("mean")
    df["abs_dev"] = (df.star - df.p_mean_star).abs()
    df["u_n"] = df.groupby("user_id").star.transform("size")

    f, g = df[df.label == 1], df[df.label == 0]
    out = ["# EDA — YelpChi", "",
           f"`{len(df):,}` reviews · `{df.label.sum():,}` fake "
           f"({df.label.mean():.1%}) · `{df.user_id.nunique():,}` reviewers · "
           f"`{df.product_id.nunique():,}` products · "
           f"{df.date.min():%b %Y} – {df.date.max():%b %Y}", ""]

    # ---------- 1. dataset shape ----------
    out += ["## 1. Dataset shape", "",
            table([[d, f"{len(s):,}", f"{s.label.sum():,}", f"{s.label.mean():.1%}",
                    f"{s.product_id.nunique()}"]
                   for d, s in df.groupby("domain")],
                  ["domain", "reviews", "fake", "fake rate", "products"]), ""]

    # ---------- 2. fake vs genuine ----------
    rows = [
        ["Reviews with any useful/funny/cool vote", fmt(f.has_vote.mean(), True), fmt(g.has_vote.mean(), True)],
        ["Median length (words)", f"{f.n_words.median():.0f}", f"{g.n_words.median():.0f}"],
        ["Mean length (words)", f"{f.n_words.mean():.0f}", f"{g.n_words.mean():.0f}"],
        ["Extreme rating (1 or 5 stars)", fmt(f.extreme.mean(), True), fmt(g.extreme.mean(), True)],
        ["1-star share", fmt((f.star == 1).mean(), True), fmt((g.star == 1).mean(), True)],
        ["5-star share", fmt((f.star == 5).mean(), True), fmt((g.star == 5).mean(), True)],
        ["Mean abs. deviation from product mean", f"{f.abs_dev.mean():.2f}", f"{g.abs_dev.mean():.2f}"],
        ["Mean reviews by the same reviewer", f"{f.u_n.mean():.2f}", f"{g.u_n.mean():.2f}"],
        ["Mean star", f"{f.star.mean():.2f}", f"{g.star.mean():.2f}"],
    ]
    out += ["## 2. Fake vs genuine", "", table(rows, ["feature", "fake", "genuine"]), ""]

    # ---------- 3. the four findings ----------
    u = df.groupby("user_id").label.agg(["size", "sum", "max", "min"])
    mixed = int(((u["max"] == 1) & (u["min"] == 0)).sum())
    single_users = (u["size"] == 1).sum()
    single_rev = df[df.u_n == 1]
    repeat_rev = df[df.u_n > 1]
    graph = build_graph_report(df, save=False)
    rur_hom = graph["RUR"]["homophily"]

    findings = [
        ["Votes leak the label",
         f"{int(f.has_vote.sum())} of {len(f):,} fakes carry a vote "
         f"(genuine: {g.has_vote.mean():.1%})", "drop useful/funny/cool"],
        ["Labels are nearly per-reviewer",
         f"{mixed:,} of {df.user_id.nunique():,} reviewers have both labels; "
         f"same-reviewer review pairs share a label {rur_hom:.1%} of the time",
         "main split is user-disjoint"],
        ["Most fakes come from one-review accounts",
         f"{single_users / len(u):.1%} of reviewers are singletons; they are "
         f"{len(single_rev) / len(df):.1%} of reviews and "
         f"{single_rev.label.sum() / f.shape[0]:.1%} of fakes; "
         f"fake rate {single_rev.label.mean():.1%} vs {repeat_rev.label.mean():.1%}",
         "model must decide from a review's own features"],
        ["Product relations are weak",
         f"a fake's same-product-month neighbors are only "
         f"{graph['RTR']['fake_neighbor_share_of_fakes']:.1%} fake "
         f"(same-star: {graph['RSR']['fake_neighbor_share_of_fakes']:.1%})",
         "filter product neighbors, separate weight per relation"],
    ]
    out += ["## 3. Four findings that shape the design", "",
            table(findings, ["finding", "number", "design decision"]), ""]

    # ---------- 4. graph ----------
    out += ["## 4. Review graph", "",
            table([[k, f"{v['edges']:,}", f"{v['homophily']:.3f}"] for k, v in graph.items()],
                  ["relation", "edges", "label homophily"]), ""]

    # ---------- 5. reviewer activity ----------
    buckets = pd.cut(df.u_n, [0, 1, 2, 3, 5, 10, 1e9],
                     labels=["1", "2", "3", "4-5", "6-10", "11+"])
    act = df.groupby(buckets, observed=True).agg(
        reviews=("label", "size"), fake=("label", "sum"), fake_rate=("label", "mean"))
    act["share_of_fakes"] = act.fake / f.shape[0]
    out += ["## 5. Fake rate by reviewer activity", "",
            table([[i, f"{int(r.reviews):,}", f"{int(r.fake):,}", f"{r.fake_rate:.1%}",
                    f"{r.share_of_fakes:.1%}"] for i, r in act.iterrows()],
                  ["reviews by reviewer", "reviews", "fake", "fake rate", "share of all fakes"]), ""]

    # ---------- figures ----------
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9})

    # fig 1 — star distribution by label
    fig, ax = plt.subplots(figsize=(5, 3))
    w = 0.38
    for i, (lab, s, c) in enumerate([("genuine", g, GEN), ("fake", f, FAKE)]):
        share = s.star.value_counts(normalize=True).reindex([1, 2, 3, 4, 5], fill_value=0)
        ax.bar(np.arange(5) + (i - 0.5) * w, share, w, label=lab, color=c)
    ax.set(xticks=range(5), xticklabels=[1, 2, 3, 4, 5], xlabel="star rating",
           ylabel="share of reviews", title="Star rating by label")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "fig1_star_by_label.png"); plt.close(fig)

    # fig 2 — length distribution by label
    fig, ax = plt.subplots(figsize=(5, 3))
    bins = np.linspace(0, 600, 61)
    for lab, s, c in [("genuine", g, GEN), ("fake", f, FAKE)]:
        ax.hist(s.n_words.clip(upper=600), bins=bins, density=True, alpha=0.65,
                label=f"{lab} (median {s.n_words.median():.0f})", color=c)
    ax.set(xlabel="words in review", ylabel="density", title="Review length by label")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "fig2_length_by_label.png"); plt.close(fig)

    # fig 3 — reviews per reviewer + fake rate
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(range(len(act)), act.reviews, color=GEN, label="reviews")
    ax.set(xticks=range(len(act)), xticklabels=act.index.astype(str),
           xlabel="reviews written by the reviewer", ylabel="reviews",
           title="Reviewer activity and fake rate")
    ax2 = ax.twinx()
    ax2.plot(range(len(act)), 100 * act.fake_rate, color=FAKE, marker="o", label="fake rate")
    ax2.set_ylabel("fake rate (%)", color=FAKE); ax2.set_ylim(0, None)
    ax.legend(loc="upper right", frameon=False); ax2.legend(loc="center right", frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "fig3_reviews_per_reviewer.png"); plt.close(fig)

    out += ["## Figures", "",
            "- `figures/fig1_star_by_label.png`",
            "- `figures/fig2_length_by_label.png`",
            "- `figures/fig3_reviews_per_reviewer.png`", ""]

    report = "\n".join(out)
    (ROOT / "results/eda_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()

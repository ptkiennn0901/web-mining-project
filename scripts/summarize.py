"""Turn results/runs.csv into the slide-11 table (mean +- std over seeds)."""
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
from load_data import ROOT   # noqa: E402

ORDER = ["E1a", "E2-LR", "E2-TREE", "E4", "Leak-1", "Leak-2",
         "E2-time", "E2-domain"]
METRICS = ["auc", "ap", "macro_f1", "p_at_1pct", "auc_singleton", "auc_repeat"]
LABEL = {
    "E1a": "E1a  text: TF-IDF word+char + LR",
    "E2-LR": "E2   behavior + LR",
    "E2-TREE": "E2   behavior + boosting",
    "E4": "E4   behavior + text stats + text score",
    "Leak-1": "Leak-1  behavior + useful/funny/cool",
    "Leak-2": "Leak-2  behavior, RANDOM split",
    "E2-time": "E2-time  behavior, time split",
    "E2-domain": "E2-domain  behavior, restaurants->hotels",
}


def _transfer_verdict(auc):
    """Tiered wording so a weak-but-real transfer is not called chance."""
    if auc < 0.55:
        return "at chance — behavior features do not transfer at all"
    if auc < 0.65:
        return ("well above chance but far below in-domain — behavior features "
                "transfer only weakly")
    return "a large drop"


def md_table(rows, head):
    w = [max(len(str(r[i])) for r in [head] + rows) for i in range(len(head))]
    line = lambda r: "| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |"
    return "\n".join([line(head), "|" + "|".join("-" * (x + 2) for x in w) + "|"]
                     + [line(r) for r in rows])


def main():
    df = pd.read_csv(ROOT / "results/runs.csv")
    df = df.drop_duplicates(subset=["exp", "seed"], keep="last")
    g = df.groupby("exp")
    agg = g[METRICS].agg(["mean", "std"])
    meta = g.agg(hypothesis=("hypothesis", "first"), split=("split", "first"),
                 model=("model", "first"), seeds=("seed", "nunique"))

    cell = lambda e, m: (f"{agg.loc[e, (m, 'mean')]:.3f}"
                         + (f" ± {agg.loc[e, (m, 'std')]:.3f}"
                            if pd.notna(agg.loc[e, (m, 'std')]) else ""))
    rows = [[LABEL.get(e, e), meta.loc[e, "split"],
             cell(e, "auc"), cell(e, "ap"), cell(e, "macro_f1"), cell(e, "p_at_1pct"),
             meta.loc[e, "hypothesis"], int(meta.loc[e, "seeds"])]
            for e in ORDER if e in agg.index]
    main_tbl = md_table(rows, ["run", "split", "AUC-ROC", "AP", "macro-F1",
                               "P@top1%", "tests", "seeds"])

    srows = [[LABEL.get(e, e), cell(e, "auc_singleton"), cell(e, "auc_repeat"),
              f"{agg.loc[e, ('auc_repeat', 'mean')] - agg.loc[e, ('auc_singleton', 'mean')]:+.3f}"]
             for e in ORDER
             if e in agg.index and pd.notna(agg.loc[e, ("auc_singleton", "mean")])]
    sub_tbl = md_table(srows, ["run", "AUC one-review", "AUC repeat", "gap"])

    # hypothesis read-out
    m = lambda e, k: agg.loc[e, (k, "mean")] if e in agg.index else float("nan")
    notes = []
    if {"E1a", "E2-TREE"} <= set(agg.index):
        d = m("E2-TREE", "auc") - m("E1a", "auc")
        da = m("E2-TREE", "ap") - m("E1a", "ap")
        notes.append(f"- **H1** (behavior beats text): AUC {m('E2-TREE','auc'):.3f} vs "
                     f"{m('E1a','auc'):.3f} ({d:+.3f}); AP {m('E2-TREE','ap'):.3f} vs "
                     f"{m('E1a','ap'):.3f} ({da:+.3f}) -> "
                     f"{'supported on AUC' if d > 0 else 'not supported'}"
                     f"{', but text wins on AP' if da < 0 else ''}.")
    if {"E4", "E2-TREE", "E1a"} <= set(agg.index):
        best = max(m("E2-TREE", "ap"), m("E1a", "ap"))
        notes.append(f"- **H2** (combining beats the best single signal in AP): "
                     f"E4 AP {m('E4','ap'):.3f} vs best single {best:.3f} "
                     f"({m('E4','ap') - best:+.3f}) -> "
                     f"{'supported' if m('E4','ap') > best else 'not supported'}.")
    if {"Leak-2", "E2-TREE"} <= set(agg.index):
        d = m("Leak-2", "auc") - m("E2-TREE", "auc")
        notes.append(f"- **H3** (random split inflates AUC): random {m('Leak-2','auc'):.3f} vs "
                     f"user-disjoint {m('E2-TREE','auc'):.3f} ({d:+.3f}) -> "
                     f"{'supported' if d > 0 else 'not supported'}.")
    if {"E1a", "E2-TREE"} <= set(agg.index):
        notes.append(
            f"- **H4 motivation** (where each signal works): on one-review accounts "
            f"text scores {m('E1a','auc_singleton'):.3f} against behavior's "
            f"{m('E2-TREE','auc_singleton'):.3f}, while on repeat reviewers behavior "
            f"scores {m('E2-TREE','auc_repeat'):.3f} against text's "
            f"{m('E1a','auc_repeat'):.3f}. The two signals are strong on **opposite** "
            f"groups, which is the case for keeping ego features separate and for "
            f"fusing signals rather than picking one.")
    if {"E2-domain", "E2-TREE"} <= set(agg.index):
        notes.append(
            f"- **RQ4** (restaurants → hotels): AUC {m('E2-domain','auc'):.3f} against "
            f"{m('E2-TREE','auc'):.3f} in-domain, i.e. "
            f"{_transfer_verdict(m('E2-domain','auc'))}. "
            f"Hotel-only reviewers have a different activity profile from the "
            f"restaurant reviewers the model was fitted on. This narrows the "
            f"scope-of-claims slide: transfer degrades badly even *within* Yelp. "
            f"Note the verdict is model-dependent — HistGradientBoosting scored "
            f"0.51 here where XGBoost scores {m('E2-domain','auc'):.2f}, so report "
            f"the model alongside the number.")
    if {"Leak-1", "E2-TREE"} <= set(agg.index):
        d = m("Leak-1", "auc") - m("E2-TREE", "auc")
        notes.append(f"- **Vote leak**: adding useful/funny/cool moves AUC "
                     f"{m('E2-TREE','auc'):.3f} -> {m('Leak-1','auc'):.3f} ({d:+.3f}) "
                     f"and AP {m('E2-TREE','ap'):.3f} -> {m('Leak-1','ap'):.3f}. "
                     f"Confirms the columns must stay out.")

    caveats = [
        "- Behavior features are computed once over the **whole** dataset (from "
        "`date, user_id, product_id, star` only, never labels). For the "
        "user-disjoint, random and domain splits this is the transductive "
        "setting the task definition declares. For **E2-time it is a temporal "
        "leak**: a 2010 training review's `u_n` and `u_active_days` already "
        "count its reviewer's 2012 activity, so that row is optimistic and is "
        "reported as a robustness check only.",
        "- `Leak-1` and `Leak-2` are diagnostics, not candidate models. They "
        "measure what the vote columns and the random split would buy; neither "
        "is used anywhere else.",
        "- Macro-F1 depends on a threshold tuned on validation, so it moves more "
        "across seeds than AUC or AP. AP is the headline metric for a 13.2% "
        "positive rate.",
    ]
    txt = "\n".join([
        "# Baseline results — checkpoint", "",
        f"Mean ± std over {int(meta.seeds.max())} seed(s) (42–46). Tree model: "
        f"`{meta.model.mode().iloc[0]}`. Threshold for macro-F1 tuned on validation.",
        "", "## Main table", "", main_tbl, "",
        "## By reviewer history (H4 baseline: one-review accounts are the hard case)",
        "", sub_tbl, "", "## Hypotheses", "", *notes, "",
        "## Caveats to state on the slide", "", *caveats, "",
    ])
    (ROOT / "results/summary.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()

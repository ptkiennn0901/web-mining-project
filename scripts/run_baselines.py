"""Run the checkpoint baselines and fill results/runs.csv.

    python scripts/run_baselines.py                  # all runs, seeds 42-46
    python scripts/run_baselines.py --seeds 42       # one seed (fast check)
    python scripts/run_baselines.py --only E2-TREE E4

Runs
  E1a         TF-IDF word+char + LR            user-disjoint   H1
  E2-LR       behavior + LR                    user-disjoint   H1
  E2-TREE     behavior + gradient boosting     user-disjoint   H1
  E4          behavior + text stats + OOF text score (stacked) user-disjoint  H2
  Leak-1      behavior + useful/funny/cool     user-disjoint   vote leak
  Leak-2      behavior, same model             random          H3
  E2-time     behavior                         time            robustness
  E2-domain   behavior                         domain          RQ4 early check
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from evaluate import evaluate, log_run                       # noqa: E402
from features.behavior import behavior_features              # noqa: E402
from features.text import oof_text_scores, text_stats, tfidf_lr   # noqa: E402
from load_data import ROOT, load_processed                    # noqa: E402
from models.baselines import logreg, tree, tree_name          # noqa: E402
from splits import BUILDERS, check                            # noqa: E402

VOTES = ["useful", "funny", "cool"]
ALL_RUNS = ["E1a", "E2-LR", "E2-TREE", "E4", "Leak-1", "Leak-2", "E2-time", "E2-domain"]


def get_split(df, name, seed):
    """Load the committed split file, or build it once and commit it."""
    suffix = "" if seed == 42 else f"_seed{seed}"
    path = ROOT / "splits" / f"{name}{suffix}.json"
    if path.exists():
        return json.loads(path.read_text())
    sp = BUILDERS[name](df, seed=seed)
    check(df, name, sp)
    path.write_text(json.dumps(sp))
    return sp


def fit_score(model, Xtr, ytr, Xva, Xte):
    model.fit(Xtr, ytr)
    return model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1]


CACHE = ROOT / "data/processed/text_cache"


def _fingerprint(tr, va, te):
    """Guards the cache against edited split files."""
    return np.array([len(tr), len(va), len(te),
                     int(tr.sum()), int(va.sum()), int(te.sum())], dtype=np.int64)


def cached_text_scores(split_name, seed, text_all, tr, va, te, ytr, want_oof):
    """TF-IDF+LR scores for val/test (full fit) and, optionally, out-of-fold
    scores for train.

    Cached per (split, seed) because E1a and E4 fit the exact same text model,
    and the out-of-fold pass is five more fits on top. The cache is keyed to a
    fingerprint of the split, so editing a split file invalidates it instead of
    silently reusing stale scores.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    fp = _fingerprint(tr, va, te)
    full_p = CACHE / f"full_{split_name}_{seed}.npz"
    oof_p = CACHE / f"oof_{split_name}_{seed}.npz"

    pva = pte = None
    if full_p.exists():
        z = np.load(full_p)
        if np.array_equal(z["fp"], fp):
            pva, pte = z["val"], z["test"]
    if pva is None:
        (pva, pte), _ = tfidf_lr(text_all.loc[tr].to_numpy(), ytr,
                                 [text_all.loc[va].to_numpy(),
                                  text_all.loc[te].to_numpy()], seed=seed)
        np.savez_compressed(full_p, val=pva, test=pte, fp=fp)

    oof = None
    if want_oof:
        if oof_p.exists():
            z = np.load(oof_p)
            if np.array_equal(z["fp"], fp):
                oof = z["oof"]
        if oof is None:
            oof = oof_text_scores(text_all.loc[tr].to_numpy(), ytr, seed=seed)
            np.savez_compressed(oof_p, oof=oof, fp=fp)
    return oof, pva, pte


def main(seeds, only):
    t0 = time.time()
    df = load_processed()
    beh = behavior_features(df).set_index("row_id")
    txt = text_stats(df).set_index("row_id")
    votes = df.set_index("row_id")[VOTES]
    y_all = df.set_index("row_id").label
    text_all = df.set_index("row_id").text
    singleton = beh.u_singleton

    beh_cols = list(beh.columns)
    print(f"tree model: {tree_name()}   behavior features: {len(beh_cols)}   "
          f"text stats: {txt.shape[1]}\n")

    for seed in seeds:
        for exp in only:
            split_name = {"Leak-2": "random", "E2-time": "time",
                          "E2-domain": "domain"}.get(exp, "user_disjoint")
            sp = get_split(df, split_name, seed)
            tr, va, te = (np.array(sp[k]) for k in ("train", "val", "test"))
            ytr, yva, yte = y_all.loc[tr].to_numpy(), y_all.loc[va].to_numpy(), y_all.loc[te].to_numpy()
            pos_w = (ytr == 0).sum() / max(1, (ytr == 1).sum())
            sing_te = singleton.loc[te].to_numpy()
            tag = f"[seed {seed}] {exp:<12}"
            t1 = time.time()

            if exp == "E1a":
                hyp, mdl, feats = "H1", "TF-IDF+LR", "tfidf_word+char"
                _, pva, pte = cached_text_scores(split_name, seed, text_all,
                                                 tr, va, te, ytr, want_oof=False)

            elif exp in ("E2-LR", "E2-TREE", "Leak-2", "E2-time", "E2-domain"):
                cols = beh_cols
                X = beh[cols]
                if exp == "E2-LR":
                    hyp, model, mdl = "H1", logreg(seed), "LR"
                else:
                    hyp = {"Leak-2": "H3"}.get(exp, "H1")
                    model, mdl = tree(pos_w, seed), tree_name()
                feats = "behavior"
                pva, pte = fit_score(model, X.loc[tr], ytr, X.loc[va], X.loc[te])

            elif exp == "Leak-1":
                hyp, feats, mdl = "leak", "behavior+votes", tree_name()
                X = pd.concat([beh[beh_cols], votes], axis=1)
                pva, pte = fit_score(tree(pos_w, seed), X.loc[tr], ytr, X.loc[va], X.loc[te])

            elif exp == "E4":
                hyp, feats, mdl = "H2", "behavior+text_stats+oof_text_score", tree_name()
                # out-of-fold text score on train, full-fit text score on val/test
                oof, pva_t, pte_t = cached_text_scores(split_name, seed, text_all,
                                                       tr, va, te, ytr, want_oof=True)
                base = pd.concat([beh[beh_cols], txt], axis=1)
                Xtr = base.loc[tr].assign(text_score=oof)
                Xva = base.loc[va].assign(text_score=pva_t)
                Xte = base.loc[te].assign(text_score=pte_t)
                pva, pte = fit_score(tree(pos_w, seed), Xtr, ytr, Xva, Xte)

            else:
                raise ValueError(exp)

            m = evaluate(yva, pva, yte, pte, singleton_test=sing_te)
            log_run(dict(exp=exp, hypothesis=hyp, model=mdl, features=feats,
                         split=split_name, seed=seed, n_train=len(tr), n_test=len(te), **m))
            print(f"{tag} {split_name:<13} auc={m['auc']:.4f} ap={m['ap']:.4f} "
                  f"f1={m['macro_f1']:.4f} p@1%={m['p_at_1pct']:.3f} "
                  f"auc_single={m['auc_singleton']:.4f} auc_repeat={m['auc_repeat']:.4f} "
                  f"({time.time() - t1:.0f}s)")

    print(f"\ndone in {time.time() - t0:.0f}s -> {ROOT / 'results/runs.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--only", nargs="+", default=ALL_RUNS, choices=ALL_RUNS)
    a = ap.parse_args()
    main(a.seeds, a.only)

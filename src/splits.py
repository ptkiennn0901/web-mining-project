"""Build the four split files. Lists of `row_id`, written once, never re-split
inside a notebook.

user_disjoint  main split      — 60/20/20 by reviewer (no reviewer in two parts)
random         comparison      — 40/20/40 by review, as used in the papers (H3)
time           robustness      — train <=2010, test >=2011
domain         transfer (E8)   — restaurants -> hotels, unseen reviewers only
"""
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from load_data import ROOT, load_processed

SEED = 42
OUT = ROOT / "splits"


def user_disjoint(df, seed=SEED):
    """Split reviewers, not reviews. Stratified on whether the reviewer ever
    wrote a flagged review, so each part keeps a similar fake rate."""
    users = df.groupby("user_id")["label"].max().rename("has_fake").reset_index()
    tr, tmp = train_test_split(users, test_size=0.4, stratify=users.has_fake, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.5, stratify=tmp.has_fake, random_state=seed)
    pick = lambda u: df.loc[df.user_id.isin(set(u.user_id)), "row_id"].tolist()
    return {"train": pick(tr), "val": pick(va), "test": pick(te)}


def random_split(df, seed=SEED):
    tr, tmp = train_test_split(df, test_size=0.6, stratify=df.label, random_state=seed)
    va, te = train_test_split(tmp, test_size=2 / 3, stratify=tmp.label, random_state=seed)
    return {"train": tr.row_id.tolist(), "val": va.row_id.tolist(), "test": te.row_id.tolist()}


def time_split(df, seed=SEED):
    past = df[df.date.dt.year <= 2010]
    tr, va = train_test_split(past, test_size=0.15, stratify=past.label, random_state=seed)
    te = df[df.date.dt.year >= 2011]
    return {"train": tr.row_id.tolist(), "val": va.row_id.tolist(), "test": te.row_id.tolist()}


def domain_split(df, seed=SEED):
    res = df[df.domain == "restaurant"]
    tr, va = train_test_split(res, test_size=0.15, stratify=res.label, random_state=seed)
    # drop hotel reviewers who also wrote restaurant reviews, else the reviewer
    # features carry the label straight from training into the test set
    te = df[(df.domain == "hotel") & ~df.user_id.isin(set(res.user_id))]
    return {"train": tr.row_id.tolist(), "val": va.row_id.tolist(), "test": te.row_id.tolist()}


BUILDERS = {
    "user_disjoint": user_disjoint,
    "random": random_split,
    "time": time_split,
    "domain": domain_split,
}


def load_split(name, seed=SEED, folder=OUT):
    suffix = "" if seed == SEED else f"_seed{seed}"
    return json.loads((Path(folder) / f"{name}{suffix}.json").read_text())


def check(df, name, sp):
    """Leak tests. Raises on the two leaks the report calls out."""
    idx = df.set_index("row_id")
    parts = {k: idx.loc[v] for k, v in sp.items()}
    sizes = {k: len(v) for k, v in sp.items()}
    rates = {k: round(float(v.label.mean()), 4) for k, v in parts.items()}

    all_ids = [i for v in sp.values() for i in v]
    assert len(all_ids) == len(set(all_ids)), f"{name}: a row_id is in two parts"

    if name == "user_disjoint":
        us = {k: set(v.user_id) for k, v in parts.items()}
        assert not (us["train"] & us["val"]), f"{name}: reviewer in train and val"
        assert not (us["train"] & us["test"]), f"{name}: reviewer in train and test"
        assert not (us["val"] & us["test"]), f"{name}: reviewer in val and test"
        assert len(all_ids) == len(df), f"{name}: covers {len(all_ids)} of {len(df)} rows"
    if name == "domain":
        tr_u = set(parts["train"].user_id) | set(parts["val"].user_id)
        assert not (tr_u & set(parts["test"].user_id)), f"{name}: reviewer seen in training"
        assert set(parts["test"].domain) == {"hotel"}, f"{name}: test is not hotel-only"

    print(f"  {name:<14} sizes={sizes}  fake_rate={rates}")
    return True


if __name__ == "__main__":
    df = load_processed()
    OUT.mkdir(exist_ok=True)
    print("splits (seed 42):")
    for name, fn in BUILDERS.items():
        sp = fn(df)
        check(df, name, sp)
        (OUT / f"{name}.json").write_text(json.dumps(sp))

    # extra seeds for the user-disjoint split, for the mean +- std table
    for seed in range(43, 47):
        sp = user_disjoint(df, seed=seed)
        check(df, "user_disjoint", sp)
        (OUT / f"user_disjoint_seed{seed}.json").write_text(json.dumps(sp))
    print("all leak tests passed")

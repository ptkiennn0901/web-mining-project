"""Checkpoint asserts: counts, no label leak across the user-disjoint split,
feature row order, and the graph edge counts.

    python -m pytest tests/ -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from features.behavior import behavior_features     # noqa: E402
from features.text import text_stats                # noqa: E402
from graph.build_edges import build as build_graph   # noqa: E402
from load_data import load_processed                 # noqa: E402
from splits import BUILDERS                          # noqa: E402

EXPECTED = dict(n_rows=67395, n_fake=8919, n_users=38063, n_products=201,
                hotel=5854, restaurant=61541)
EDGES = dict(RUR=110112, RTR=466352, RSR=8166840)


@pytest.fixture(scope="module")
def df():
    return load_processed()


def test_counts(df):
    assert len(df) == EXPECTED["n_rows"]
    assert int(df.label.sum()) == EXPECTED["n_fake"]
    assert df.user_id.nunique() == EXPECTED["n_users"]
    assert df.product_id.nunique() == EXPECTED["n_products"]
    assert df.domain.value_counts().to_dict() == {
        "restaurant": EXPECTED["restaurant"], "hotel": EXPECTED["hotel"]}


def test_row_id_and_dates(df):
    assert df.row_id.tolist() == list(range(len(df)))
    assert df.date.dt.year.between(2004, 2012).all()
    assert df.star.between(1, 5).all()
    assert df.text.notna().all() and (df.text.str.len() > 0).all()


def test_votes_leak_the_label(df):
    """The finding the whole design rests on: no flagged review has a vote."""
    assert df.loc[df.label == 1, ["useful", "funny", "cool"]].to_numpy().sum() == 0
    assert df.loc[df.label == 0].pipe(
        lambda d: (d[["useful", "funny", "cool"]].sum(axis=1) > 0).mean()) > 0.49


@pytest.mark.parametrize("name", ["user_disjoint", "random", "time", "domain"])
def test_split_file_exists_and_is_disjoint(df, name):
    sp = json.loads((ROOT_DIR / "splits" / f"{name}.json").read_text())
    ids = [i for part in sp.values() for i in part]
    assert len(ids) == len(set(ids)), f"{name}: a row_id appears twice"
    assert set(ids) <= set(df.row_id)


def test_user_disjoint_has_no_reviewer_leak(df):
    sp = json.loads((ROOT_DIR / "splits" / "user_disjoint.json").read_text())
    idx = df.set_index("row_id")
    users = {k: set(idx.loc[v].user_id) for k, v in sp.items()}
    assert not users["train"] & users["val"]
    assert not users["train"] & users["test"]
    assert not users["val"] & users["test"]
    assert sum(len(v) for v in sp.values()) == len(df), "split must cover every row"


def test_domain_split_has_no_reviewer_leak(df):
    sp = json.loads((ROOT_DIR / "splits" / "domain.json").read_text())
    idx = df.set_index("row_id")
    seen = set(idx.loc[sp["train"]].user_id) | set(idx.loc[sp["val"]].user_id)
    test = idx.loc[sp["test"]]
    assert not seen & set(test.user_id)
    assert set(test.domain) == {"hotel"}


def test_splits_are_reproducible(df):
    """Rebuilding with seed 42 must give back the committed file."""
    for name, fn in BUILDERS.items():
        built = fn(df)
        saved = json.loads((ROOT_DIR / "splits" / f"{name}.json").read_text())
        for part in ("train", "val", "test"):
            assert built[part] == saved[part], f"{name}/{part} drifted"


def test_feature_tables_keep_parquet_row_order(df):
    for fn in (behavior_features, text_stats):
        f = fn(df)
        assert len(f) == len(df)
        assert (f.row_id.to_numpy() == df.row_id.to_numpy()).all()
        cols = [c for c in f.columns if c != "row_id"]
        assert np.isfinite(f[cols].to_numpy().astype(float)).all()


def test_no_vote_column_in_features(df):
    for fn in (behavior_features, text_stats):
        assert not {"useful", "funny", "cool"} & set(fn(df).columns)


def test_edge_counts_and_homophily(df):
    rep = build_graph(df, save=False)
    assert {k: v["edges"] for k, v in rep.items()} == EDGES
    assert rep["RUR"]["homophily"] > 0.99          # labels are nearly per-reviewer
    assert 0.78 < rep["RTR"]["homophily"] < 0.81   # product relations are weak
    assert 0.78 < rep["RSR"]["homophily"] < 0.81

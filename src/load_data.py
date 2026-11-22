"""Load the 4 raw YelpChi files into one Parquet table.

Meta line = 9 space-separated fields:
    date  review_id  user_id  product_id  label(Y/N)  useful  funny  cool  star
The review file has one review per line, in the same order as the meta file.
We unpack from the right so a review ID containing a space cannot shift columns.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw"
OUT = ROOT / "data/processed/reviews.parquet"

FILES = {
    "hotel": ("output_meta_yelpHotelData_NRYRcleaned.txt",
              "output_review_yelpHotelData_NRYRcleaned.txt"),
    "restaurant": ("output_meta_yelpResData_NRYRcleaned.txt",
                   "output_review_yelpResData_NRYRcleaned.txt"),
}


def read_domain(meta_path, text_path, domain):
    rows = []
    with open(meta_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            date, *rest = line.split()
            *rid, uid, pid, lab, useful, funny, cool, star = rest
            rows.append(dict(
                date=date, review_id=" ".join(rid), user_id=uid, product_id=pid,
                label=1 if lab == "Y" else 0,
                useful=int(useful), funny=int(funny), cool=int(cool),
                star=int(star), domain=domain,
            ))

    # Do NOT drop blank lines here: text and meta are aligned by line number,
    # so a blank review would shift every following label.
    with open(text_path, encoding="utf-8", errors="replace") as f:
        texts = [line.rstrip("\n") for line in f]
    while texts and not texts[-1].strip():      # trailing newline at EOF only
        texts.pop()

    assert len(texts) == len(rows), f"{domain}: {len(texts)} texts vs {len(rows)} meta rows"
    df = pd.DataFrame(rows)
    df["text"] = texts
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    return df


def load_yelpchi(raw=RAW):
    raw = Path(raw)
    parts = [read_domain(raw / m, raw / t, d) for d, (m, t) in FILES.items()]
    df = pd.concat(parts, ignore_index=True)
    df["row_id"] = range(len(df))      # stable integer ID used everywhere
    return df


def load_processed(path=OUT):
    return pd.read_parquet(path)


if __name__ == "__main__":
    df = load_yelpchi()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT}  rows={len(df):,}")
    print(f"  fake      : {df.label.sum():,} ({df.label.mean():.1%})")
    print(f"  reviewers : {df.user_id.nunique():,}")
    print(f"  products  : {df.product_id.nunique():,}")
    print(f"  per domain: {df.domain.value_counts().to_dict()}")
    print(f"  date range: {df.date.min().date()} .. {df.date.max().date()}")

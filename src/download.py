"""Fetch the 4 raw YelpChi files into data/raw/.

The shared Drive folder is public:
    https://drive.google.com/drive/folders/1FkwjyhqX78mbxH80HQxvIcqfdzAziRv5
(inside https://drive.google.com/drive/folders/1B15rx0TZbfuJyE4hrZrp1HKB_xfXXA8M,
which also holds YelpNYC and YelpZip — we only need YelpChi.)

Uses `gdown` when it is installed, otherwise urllib against the public
download endpoint, so a fresh Colab needs no extra package.
"""
import sys
import urllib.request
from pathlib import Path

from load_data import RAW

FOLDER = "https://drive.google.com/drive/folders/1FkwjyhqX78mbxH80HQxvIcqfdzAziRv5"

# name -> (Drive file id, expected line count)
FILES = {
    "output_meta_yelpHotelData_NRYRcleaned.txt":   ("1BvFIly7G-awRpWwk0Dxls_4jIuMrzATy", 5854),
    "output_meta_yelpResData_NRYRcleaned.txt":     ("1u4kaHXKb1NfbrID6MFUVtCt_RPeqwwJA", 61541),
    "output_review_yelpHotelData_NRYRcleaned.txt": ("1KD_XiG3dKf5vDf6DzgBmROeqoO2Nm-d2", 5854),
    "output_review_yelpResData_NRYRcleaned.txt":   ("1G-KWHu3ZofOARHJHn0WdsyLVAor5fZle", 61541),
}
URL = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"


def count_lines(path):
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def main(raw=RAW):
    raw = Path(raw)
    raw.mkdir(parents=True, exist_ok=True)
    for name, (fid, n_lines) in FILES.items():
        dest = raw / name
        if dest.exists() and count_lines(dest) == n_lines:
            print(f"  have  {name}")
            continue
        print(f"  get   {name} ...", end=" ", flush=True)
        try:
            import gdown
            gdown.download(id=fid, output=str(dest), quiet=True)
        except ImportError:
            urllib.request.urlretrieve(URL.format(fid), dest)
        got = count_lines(dest)
        if got != n_lines:
            sys.exit(f"\n{name}: got {got} lines, expected {n_lines}. "
                     f"Download it by hand from {FOLDER}")
        print(f"ok ({got:,} lines)")
    print(f"raw data ready in {raw}")


if __name__ == "__main__":
    main()

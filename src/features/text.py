"""Text features: TF-IDF (word + char) fitted on the train split only, plus
cheap style statistics that can be computed on the whole dataset.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from load_data import ROOT, load_processed

OUT = ROOT / "data/processed/text_stats.parquet"
SEED = 42

FIRST_PERSON = re.compile(r"\b(i|me|my|mine|myself|we|us|our|ours)\b", re.I)
SENT_END = re.compile(r"[.!?]+")


def text_stats(df):
    """Style cues. Fitted on nothing, so safe to compute on all rows."""
    t = df.text.fillna("")
    words = t.str.split()
    n_words = words.str.len().clip(lower=1)
    n_chars = t.str.len().clip(lower=1)
    d = pd.DataFrame({"row_id": df.row_id.to_numpy()})
    d["t_n_words"] = words.str.len().to_numpy()
    d["t_n_chars"] = t.str.len().to_numpy()
    d["t_mean_word_len"] = (n_chars / n_words).to_numpy()
    d["t_first_person"] = (t.str.count(FIRST_PERSON) / n_words).to_numpy()
    d["t_excl"] = t.str.count("!").to_numpy()
    d["t_excl_ratio"] = (t.str.count("!") / n_words).to_numpy()
    d["t_caps_ratio"] = (t.str.count(r"[A-Z]") / n_chars).to_numpy()
    d["t_digit_ratio"] = (t.str.count(r"\d") / n_chars).to_numpy()
    d["t_n_sent"] = t.str.count(SENT_END).clip(lower=1).to_numpy()
    d["t_words_per_sent"] = (d.t_n_words / d.t_n_sent).to_numpy()
    d["t_uniq_ratio"] = words.apply(lambda w: len(set(w)) / max(1, len(w))).to_numpy()
    return d


def _vectorizers():
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000,
                           sublinear_tf=True, strip_accents="unicode")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                           max_features=200_000, sublinear_tf=True)
    return word, char


def tfidf_lr(train_text, y_train, other_texts, C=4.0, seed=SEED):
    """Fit word+char TF-IDF and LR on the train split, score the other texts."""
    word, char = _vectorizers()
    Xtr = hstack([word.fit_transform(train_text), char.fit_transform(train_text)]).tocsr()
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed)
    clf.fit(Xtr, y_train)
    tf = lambda t: hstack([word.transform(t), char.transform(t)]).tocsr()
    return [clf.predict_proba(tf(t))[:, 1] for t in other_texts], (word, char, clf)


def oof_text_scores(train_text, y_train, n_splits=5, C=4.0, seed=SEED):
    """Out-of-fold TF-IDF scores for the TRAIN split, so a stacked model never
    sees a text score that was fitted on its own label. The vectorizers are
    refitted inside each fold for the same reason."""
    train_text = np.asarray(train_text)
    y_train = np.asarray(y_train)
    oof = np.zeros(len(y_train))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(train_text, y_train):
        (p,), _ = tfidf_lr(train_text[tr], y_train[tr], [train_text[va]], C=C, seed=seed)
        oof[va] = p
    return oof


if __name__ == "__main__":
    df = load_processed()
    d = text_stats(df)
    d.to_parquet(OUT, index=False)
    cols = [c for c in d.columns if c != "row_id"]
    print(f"wrote {OUT}  rows={len(d):,}  features={len(cols)}")
    print("  " + ", ".join(cols))
    assert (d.row_id.to_numpy() == df.row_id.to_numpy()).all(), "row order changed"
    assert np.isfinite(d[cols].to_numpy()).all(), "non-finite text stat"
    print("  row order matches parquet, all finite")

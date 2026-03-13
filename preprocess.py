"""
preprocess.py
=============
Preprocesses raw Groww reviews with REAL distribution.
No oversampling — class weights handle imbalance in training.

Output:
  processed_reviews.csv
  X_train.npy, X_test.npy, y_train.npy, y_test.npy
  tokenizer.json
"""

import re
import json
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

MAX_WORDS  = 10000
MAX_LEN    = 150
TEST_SIZE  = 0.2
INPUT_CSV  = "raw_reviews.csv"
OUTPUT_CSV = "processed_reviews.csv"

def map_sentiment(score):
    """1-3 stars = Negative (0), 4-5 stars = Positive (1)"""
    return 0 if score <= 3 else 1

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def main():
    print("=" * 60)
    print("  Preprocessing — Groww Reviews (Real Distribution)")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(INPUT_CSV)
    df = df.dropna(subset=["review", "score"])
    df["review"] = df["review"].astype(str)
    print(f"\n  [1] Loaded {len(df)} reviews")

    # ── Labels ────────────────────────────────────────────────────────────────
    df["binary"] = df["score"].astype(int).apply(map_sentiment)
    df["sentiment"] = df["binary"].map({0: "Negative", 1: "Positive"})

    print(f"\n  [2] REAL sentiment distribution:")
    total = len(df)
    for label, count in df["sentiment"].value_counts().items():
        bar = "█" * int(count / total * 30)
        print(f"      {label:10s} {bar} {count} ({count/total*100:.1f}%)")

    # ── Clean ─────────────────────────────────────────────────────────────────
    print(f"\n  [3] Cleaning text...")
    df["clean_review"] = df["review"].apply(clean_text)
    df = df[df["clean_review"].str.len() > 5].reset_index(drop=True)
    print(f"      {len(df)} reviews after cleaning")

    # ── Tokenize + Pad ────────────────────────────────────────────────────────
    print(f"\n  [4] Tokenizing...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["clean_review"])
    seqs = tokenizer.texts_to_sequences(df["clean_review"])
    X    = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")
    y    = df["binary"].values
    print(f"      Vocab: {len(tokenizer.word_index)} unique words")
    print(f"      X shape: {X.shape}")

    # ── Split ─────────────────────────────────────────────────────────────────
    print(f"\n  [5] Train/test split (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    print(f"      Train: {len(X_train)}  |  Test: {len(X_test)}")
    print(f"      NO oversampling — class weights will handle imbalance in train.py")

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\n  [6] Saving...")
    df[["score","binary","sentiment","clean_review","date","thumbs_up"]].to_csv(
        OUTPUT_CSV, index=False)
    np.save("X_train.npy", X_train)
    np.save("X_test.npy",  X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy",  y_test)
    with open("tokenizer.json", "w") as f:
        json.dump(tokenizer.to_json(), f)

    print(f"\n{'=' * 60}")
    print(f"  Done! {len(df)} reviews ready for training")
    print(f"  Real class imbalance preserved — model will learn from it")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
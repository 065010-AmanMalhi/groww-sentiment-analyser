"""
train.py
========
Binary sentiment classifier on real imbalanced Groww data.
Manual class weights tuned for better negative recall.
Architecture: Embedding → GlobalAvgPool → BatchNorm → Dense → Sigmoid
"""

import json
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Dense, Dropout,
    GlobalAveragePooling1D, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

MAX_WORDS     = 10000
MAX_LEN       = 150
EMBED_DIM     = 32
EPOCHS        = 50
BATCH_SIZE    = 32
LEARNING_RATE = 0.001
LABEL_NAMES   = {0: "Negative", 1: "Positive"}


def load_data():
    print("\n  [1] Loading data...")
    X_train = np.load("X_train.npy")
    X_test  = np.load("X_test.npy")
    y_train = np.load("y_train.npy")
    y_test  = np.load("y_test.npy")

    print(f"      Train: {len(X_train)}  |  Test: {len(X_test)}")
    print(f"\n      Real class distribution (train):")
    counts = Counter(y_train)
    total  = len(y_train)
    for label, count in sorted(counts.items()):
        bar = "█" * int(count / total * 30)
        print(f"        {LABEL_NAMES[label]:10s} {bar} {count} ({count/total*100:.1f}%)")

    return X_train, X_test, y_train, y_test


def get_class_weights(y_train):
    """
    Manual class weights tuned for better negative recall.
    Auto 'balanced' weights weren't aggressive enough (negative recall = 0.47).
    We give Negative 4x penalty — model gets punished harder for missing negatives.
    """
    counts   = Counter(y_train)
    neg_n    = counts[0]
    pos_n    = counts[1]
    ratio    = pos_n / neg_n   # natural imbalance ratio

    # Multiply ratio by 1.5 to be more aggressive than balanced
    neg_weight = ratio * 1.5
    pos_weight = 1.0

    cw = {0: neg_weight, 1: pos_weight}
    print(f"\n  [2] Class weights (manually tuned for better negative recall):")
    print(f"        Negative : {neg_weight:.2f}x  (ratio {ratio:.1f}x × 1.5)")
    print(f"        Positive : {pos_weight:.2f}x")
    return cw


def build_model():
    model = Sequential([
        Embedding(MAX_WORDS, EMBED_DIM, input_length=MAX_LEN),
        GlobalAveragePooling1D(),
        BatchNormalization(),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    print("=" * 60)
    print("  Sentiment Model — Groww (Real Distribution)")
    print("  Negative (1-3★) vs Positive (4-5★)")
    print("=" * 60)

    tf.random.set_seed(42)
    np.random.seed(42)

    X_train, X_test, y_train, y_test = load_data()
    class_weights = get_class_weights(y_train)

    model = build_model()
    print(f"\n  [3] Model summary:")
    model.summary()

    print(f"\n  [4] Training...")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=3, min_lr=1e-6, verbose=1)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\n  [5] Evaluation:")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n      Test Loss    : {loss:.4f}")
    print(f"      Test Accuracy: {acc*100:.2f}%")

    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()

    print(f"\n      Classification Report:")
    print(classification_report(y_test, y_pred,
                                  target_names=["Negative", "Positive"]))

    print(f"      Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                   Pred Neg  Pred Pos")
    for i, row in enumerate(cm):
        print(f"      Actual {LABEL_NAMES[i]:8s}: {row}")

    model.save("sentiment_model.keras")
    history_dict = {k: [float(v) for v in vals]
                    for k, vals in history.history.items()}
    with open("training_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  Saved: sentiment_model.keras + training_history.json")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
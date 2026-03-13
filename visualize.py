"""
visualize.py
============
Interactive visualizations using Plotly Express.
All charts saved as standalone HTML files (interactive) + PNG (static).

Plots:
  1. Training accuracy & loss curves
  2. Confusion matrix heatmap
  3. Sentiment & score distribution
  4. Review length distribution
  5. Word clouds (Negative vs Positive)
  6. Sentiment over time

Output: HTML files (interactive) in /plots folder
"""

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
import os

os.makedirs("plots", exist_ok=True)
MAX_LEN = 150

TEMPLATE = "plotly_dark"
GREEN    = "#00d4aa"
RED      = "#ff4757"
BLUE     = "#5352ed"
YELLOW   = "#ffa502"
PURPLE   = "#a29bfe"


def save(fig, fname):
    path = f"plots/{fname}"
    fig.write_html(path)
    print(f"  Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. TRAINING CURVES
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curves():
    with open("training_history.json") as f:
        h = json.load(f)

    epochs = list(range(1, len(h["accuracy"]) + 1))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Accuracy", "Loss"),
        horizontal_spacing=0.12
    )

    # Accuracy
    fig.add_trace(go.Scatter(x=epochs, y=h["accuracy"], name="Train Accuracy",
                             line=dict(color=GREEN, width=2.5),
                             mode="lines+markers"), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=h["val_accuracy"], name="Val Accuracy",
                             line=dict(color=BLUE, width=2.5, dash="dash"),
                             mode="lines+markers"), row=1, col=1)

    # Loss
    fig.add_trace(go.Scatter(x=epochs, y=h["loss"], name="Train Loss",
                             line=dict(color=RED, width=2.5),
                             mode="lines+markers"), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=h["val_loss"], name="Val Loss",
                             line=dict(color=YELLOW, width=2.5, dash="dash"),
                             mode="lines+markers"), row=1, col=2)

    fig.update_layout(
        title=dict(text="Model Training — Groww Sentiment Analysis",
                   font=dict(size=20), x=0.5),
        template=TEMPLATE,
        height=500,
        legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="#444"),
    )
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)

    save(fig, "1_training_curves.html")


# ══════════════════════════════════════════════════════════════════════════════
# 2. CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix():
    model = load_model("sentiment_model.keras")
    df    = pd.read_csv("processed_reviews.csv")
    df["binary"] = (df["score"] >= 4).astype(int)
    df = df.dropna(subset=["clean_review"])

    with open("tokenizer.json") as f:
        tok_json = json.load(f)
    tokenizer = tokenizer_from_json(tok_json)

    seqs  = tokenizer.texts_to_sequences(df["clean_review"].fillna(""))
    X     = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")
    y     = df["binary"].values

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    cm     = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    labels = ["Negative", "Positive"]
    text   = [[f"{cm_norm[i][j]:.2f}<br>({cm[i][j]} reviews)"
               for j in range(2)] for i in range(2)]

    fig = go.Figure(go.Heatmap(
        z=cm_norm,
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=16, color="white"),
        colorscale=[[0, "#1a1a2e"], [1, GREEN]],
        showscale=True,
        colorbar=dict(title="Proportion"),
    ))

    fig.update_layout(
        title=dict(text="Confusion Matrix — Groww Sentiment Classifier",
                   font=dict(size=18), x=0.5),
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        template=TEMPLATE,
        height=500,
        width=550,
    )

    save(fig, "2_confusion_matrix.html")


# ══════════════════════════════════════════════════════════════════════════════
# 3. DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_distributions():
    df = pd.read_csv("processed_reviews.csv")
    df["binary"] = (df["score"] >= 4).astype(int)
    df["sentiment_label"] = df["binary"].map({0: "Negative", 1: "Positive"})

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Sentiment Split", "Score Distribution (1-5★)"),
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        horizontal_spacing=0.15,
    )

    # Pie
    counts = df["sentiment_label"].value_counts()
    fig.add_trace(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        marker=dict(colors=[RED, GREEN],
                    line=dict(color="#0f0f0f", width=2)),
        textfont=dict(size=14),
        hole=0.35,
    ), row=1, col=1)

    # Bar
    score_counts = df["score"].value_counts().sort_index()
    bar_colors   = [RED if s <= 2 else GREEN for s in score_counts.index]
    fig.add_trace(go.Bar(
        x=score_counts.index.astype(str),
        y=score_counts.values,
        marker=dict(color=bar_colors, line=dict(color="#0f0f0f", width=1.5)),
        text=score_counts.values,
        textposition="outside",
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        title=dict(text="Groww Review Distributions", font=dict(size=20), x=0.5),
        template=TEMPLATE,
        height=500,
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
    )
    fig.update_xaxes(title_text="Star Rating", row=1, col=2)
    fig.update_yaxes(title_text="Number of Reviews", row=1, col=2)

    save(fig, "3_distributions.html")


# ══════════════════════════════════════════════════════════════════════════════
# 4. REVIEW LENGTH DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def plot_review_lengths():
    df = pd.read_csv("processed_reviews.csv")
    df["binary"]     = (df["score"] >= 4).astype(int)
    df["word_count"] = df["clean_review"].fillna("").apply(lambda x: len(x.split()))
    df["sentiment"]  = df["binary"].map({0: "Negative", 1: "Positive"})

    fig = px.histogram(
        df, x="word_count", color="sentiment",
        barmode="overlay",
        nbins=50,
        color_discrete_map={"Negative": RED, "Positive": GREEN},
        opacity=0.75,
        title="Review Length Distribution by Sentiment",
        labels={"word_count": "Word Count", "count": "Number of Reviews"},
        template=TEMPLATE,
    )

    # Mean lines
    for label, color in [("Negative", RED), ("Positive", GREEN)]:
        mean_val = df[df["sentiment"] == label]["word_count"].mean()
        fig.add_vline(x=mean_val, line_dash="dash", line_color=color,
                      annotation_text=f"{label} mean: {mean_val:.0f}w",
                      annotation_position="top right",
                      annotation_font_color=color)

    fig.update_layout(
        height=500,
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
        title=dict(font=dict(size=20), x=0.5),
    )

    save(fig, "4_review_lengths.html")


# ══════════════════════════════════════════════════════════════════════════════
# 5. TOP WORDS BAR CHART (Plotly alternative to wordcloud)
# ══════════════════════════════════════════════════════════════════════════════

def plot_top_words():
    df = pd.read_csv("processed_reviews.csv")
    df["binary"] = (df["score"] >= 4).astype(int)

    stopwords = {"app", "groww", "the", "and", "is", "it", "to", "a",
                 "of", "in", "for", "i", "this", "that", "with", "my",
                 "on", "are", "was", "has", "have", "be", "an", "at",
                 "by", "from", "but", "not", "so", "we", "they", "or",
                 "as", "its", "me", "you", "can", "do", "if", "all",
                 "just", "very", "s", "t", "re", "ve", "2", "1", "use",
                 "using", "used", "good", "great", "nice", "best"}

    def top_words(text_series, n=20):
        words = " ".join(text_series.fillna("")).split()
        words = [w for w in words if w not in stopwords and len(w) > 2]
        return Counter(words).most_common(n)

    neg_words = top_words(df[df["binary"] == 0]["clean_review"])
    pos_words = top_words(df[df["binary"] == 1]["clean_review"])

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Top Words — Negative Reviews",
                        "Top Words — Positive Reviews"),
        horizontal_spacing=0.2,
    )

    # Negative
    neg_df = pd.DataFrame(neg_words, columns=["word", "count"]).sort_values("count")
    fig.add_trace(go.Bar(
        x=neg_df["count"], y=neg_df["word"],
        orientation="h",
        marker=dict(color=RED, line=dict(color="#0f0f0f", width=1)),
        showlegend=False,
    ), row=1, col=1)

    # Positive
    pos_df = pd.DataFrame(pos_words, columns=["word", "count"]).sort_values("count")
    fig.add_trace(go.Bar(
        x=pos_df["count"], y=pos_df["word"],
        orientation="h",
        marker=dict(color=GREEN, line=dict(color="#0f0f0f", width=1)),
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        title=dict(text="Most Frequent Words by Sentiment",
                   font=dict(size=20), x=0.5),
        template=TEMPLATE,
        height=600,
    )
    fig.update_xaxes(title_text="Frequency")

    save(fig, "5_top_words.html")


# ══════════════════════════════════════════════════════════════════════════════
# 6. SENTIMENT OVER TIME
# ══════════════════════════════════════════════════════════════════════════════

def plot_sentiment_over_time():
    df = pd.read_csv("raw_reviews.csv", parse_dates=["date"])
    df["binary"] = (df["score"] >= 4).astype(int)
    df = df.dropna(subset=["date"])

    df["month"] = df["date"].dt.to_period("M").astype(str)
    monthly = df.groupby("month")["binary"].agg(["mean", "count"]).reset_index()
    monthly.columns = ["month", "pos_ratio", "count"]
    monthly = monthly[monthly["count"] >= 5]

    if len(monthly) < 2:
        print("  [!] Not enough date range — skipping sentiment over time.")
        return

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Sentiment ratio
    fig.add_trace(go.Scatter(
        x=monthly["month"], y=monthly["pos_ratio"],
        name="Positive Ratio",
        line=dict(color=GREEN, width=2.5),
        mode="lines+markers",
        fill="tozeroy",
        fillcolor="rgba(0, 212, 170, 0.1)",
    ), secondary_y=False)

    # Volume bars
    fig.add_trace(go.Bar(
        x=monthly["month"], y=monthly["count"],
        name="Review Volume",
        marker=dict(color=BLUE, opacity=0.4),
    ), secondary_y=True)

    # 50% line
    fig.add_hline(y=0.5, line_dash="dot", line_color="white",
                  opacity=0.4, secondary_y=False)

    fig.update_layout(
        title=dict(text="Groww — Sentiment Trend Over Time",
                   font=dict(size=20), x=0.5),
        template=TEMPLATE,
        height=500,
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
        xaxis=dict(title="Month", tickangle=-45),
    )
    fig.update_yaxes(title_text="Positive Review Ratio", secondary_y=False,
                     range=[0, 1.1], tickformat=".0%")
    fig.update_yaxes(title_text="Review Count", secondary_y=True)

    save(fig, "6_sentiment_over_time.html")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Plotly Visualizations — Groww Sentiment Analysis")
    print("=" * 60)

    print("\n  [1] Training curves...")
    plot_training_curves()

    print("  [2] Confusion matrix...")
    plot_confusion_matrix()

    print("  [3] Distributions...")
    plot_distributions()

    print("  [4] Review lengths...")
    plot_review_lengths()

    print("  [5] Top words...")
    plot_top_words()

    print("  [6] Sentiment over time...")
    plot_sentiment_over_time()

    print(f"\n{'=' * 60}")
    print("  All plots saved to /plots folder!")
    print("  Open any .html file in your browser for interactive charts.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
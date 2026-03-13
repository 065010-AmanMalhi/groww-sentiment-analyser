"""
app.py — Groww Sentiment Analyser
Premium UI: bold hero, large logo, Groww purple+green, vertical nav
"""

import json, re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Groww Sentiment AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

G_GREEN  = "#00d09c"
G_PURPLE = "#5367ff"
G_RED    = "#eb5757"
G_BG     = "#ffffff"
G_CARD   = "#f7f8ff"
G_BORDER = "#eaecf8"
G_TEXT   = "#0d0e2a"
G_MUTED  = "#8b90b8"
G_GRAD   = "linear-gradient(135deg, #5367ff 0%, #00d09c 100%)"
TEMPLATE = "plotly_white"
MAX_LEN  = 150

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after {{
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    box-sizing: border-box;
}}

html, body, [class*="css"], .stApp {{
    background: {G_BG} !important;
    color: {G_TEXT} !important;
}}

section[data-testid="stSidebar"] {{ display: none !important; }}
header[data-testid="stHeader"]   {{ display: none !important; }}
.block-container {{
    padding: 0 !important;
    max-width: 100% !important;
}}
footer {{ display: none !important; }}

/* ── TOP NAV ── */
.g-nav {{
    background: {G_BG};
    border-bottom: 1.5px solid {G_BORDER};
    padding: 0 48px;
    display: flex;
    align-items: center;
    height: 64px;
    position: sticky;
    top: 0;
    z-index: 1000;
    box-shadow: 0 2px 20px rgba(83,103,255,0.06);
}}
.g-nav-logo {{
    display: flex;
    align-items: center;
    gap: 12px;
}}
.g-logo-mark {{
    width: 42px; height: 42px;
    border-radius: 50%;
    background: {G_GRAD};
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; font-weight: 800; color: white;
    letter-spacing: -0.5px;
    box-shadow: 0 4px 16px rgba(83,103,255,0.35);
}}
.g-logo-name {{
    font-size: 1.6rem;
    font-weight: 800;
    background: {G_GRAD};
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.8px;
    line-height: 1;
}}
.g-logo-sub {{
    font-size: 0.7rem;
    font-weight: 600;
    color: {G_MUTED};
    text-transform: uppercase;
    letter-spacing: 2px;
    line-height: 1;
    margin-top: 3px;
}}
.g-badge {{
    margin-left: 14px;
    background: linear-gradient(135deg, rgba(83,103,255,0.1), rgba(0,208,156,0.1));
    border: 1.5px solid rgba(83,103,255,0.25);
    color: {G_PURPLE};
    font-size: 0.68rem;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 99px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}}

/* ── HERO ── */
.g-hero {{
    background: linear-gradient(160deg, #f3f4ff 0%, #f0fff9 50%, #ffffff 100%);
    border-bottom: 1.5px solid {G_BORDER};
    padding: 64px 48px 52px;
    position: relative;
    overflow: hidden;
}}
.g-hero::before {{
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 400px; height: 400px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(83,103,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}}
.g-hero::after {{
    content: '';
    position: absolute;
    bottom: -60px; left: 30%;
    width: 300px; height: 300px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,208,156,0.07) 0%, transparent 70%);
    pointer-events: none;
}}
.g-hero-eyebrow {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(83,103,255,0.08);
    border: 1px solid rgba(83,103,255,0.2);
    color: {G_PURPLE};
    font-size: 0.75rem;
    font-weight: 700;
    padding: 6px 14px;
    border-radius: 99px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 20px;
}}
.g-hero-title {{
    font-size: 3.6rem;
    font-weight: 800;
    color: {G_TEXT};
    letter-spacing: -2px;
    line-height: 1.08;
    margin-bottom: 16px;
}}
.g-hero-title span {{
    background: {G_GRAD};
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}
.g-hero-sub {{
    font-size: 1.1rem;
    color: {G_MUTED};
    font-weight: 500;
    line-height: 1.6;
    max-width: 560px;
    margin-bottom: 44px;
}}
.g-metrics {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
}}
.g-metric {{
    background: {G_BG};
    border: 1.5px solid {G_BORDER};
    border-radius: 16px;
    padding: 20px 28px;
    min-width: 150px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(83,103,255,0.06);
}}
.g-metric::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: {G_GRAD};
}}
.g-metric-val {{
    font-size: 2rem;
    font-weight: 800;
    background: {G_GRAD};
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
    line-height: 1.1;
}}
.g-metric-lbl {{
    font-size: 0.72rem;
    color: {G_MUTED};
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 6px;
}}

/* ── BODY LAYOUT ── */
.g-body {{
    display: flex;
    min-height: calc(100vh - 280px);
}}
.g-vnav {{
    width: 240px;
    min-width: 240px;
    background: {G_CARD};
    border-right: 1.5px solid {G_BORDER};
    padding: 28px 14px;
    position: sticky;
    top: 64px;
    height: calc(100vh - 64px);
    overflow-y: auto;
}}
.g-nav-label {{
    font-size: 0.65rem;
    font-weight: 700;
    color: {G_MUTED};
    text-transform: uppercase;
    letter-spacing: 2px;
    padding: 0 12px;
    margin: 0 0 8px;
}}
.g-nav-label:not(:first-child) {{ margin-top: 24px; }}
.g-content {{
    flex: 1;
    padding: 40px 48px;
    max-width: 1000px;
}}
.g-page-title {{
    font-size: 1.75rem;
    font-weight: 800;
    color: {G_TEXT};
    letter-spacing: -0.8px;
    margin-bottom: 6px;
}}
.g-page-sub {{
    font-size: 0.9rem;
    color: {G_MUTED};
    font-weight: 500;
    margin-bottom: 32px;
    line-height: 1.5;
}}
.g-divider {{
    border: none;
    border-top: 1.5px solid {G_BORDER};
    margin: 0 0 32px;
}}

/* ── NAV BUTTONS ── */
.stButton > button {{
    background: transparent !important;
    color: {G_MUTED} !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    text-align: left !important;
    width: 100% !important;
    justify-content: flex-start !important;
    transition: all 0.15s !important;
    box-shadow: none !important;
    letter-spacing: 0 !important;
}}
.stButton > button:hover {{
    background: rgba(83,103,255,0.08) !important;
    color: {G_PURPLE} !important;
    transform: none !important;
}}

/* ── PREDICT BUTTON specific override ── */
.predict-btn .stButton > button {{
    background: {G_GRAD} !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    box-shadow: 0 6px 24px rgba(83,103,255,0.3) !important;
    letter-spacing: 0.2px !important;
}}
.predict-btn .stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(83,103,255,0.4) !important;
    color: white !important;
}}

/* ── INPUTS ── */
textarea, .stTextArea textarea {{
    border: 1.5px solid {G_BORDER} !important;
    border-radius: 14px !important;
    background: {G_BG} !important;
    color: {G_TEXT} !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    padding: 14px 16px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    resize: vertical !important;
}}
textarea:focus {{
    border-color: {G_PURPLE} !important;
    box-shadow: 0 0 0 4px rgba(83,103,255,0.1) !important;
    outline: none !important;
}}
.stSelectbox > div > div {{
    border: 1.5px solid {G_BORDER} !important;
    border-radius: 12px !important;
    background: {G_BG} !important;
    color: {G_TEXT} !important;
}}

/* ── RESULT CARDS ── */
.g-result {{
    border-radius: 20px;
    padding: 36px;
    text-align: center;
    border: 2px solid;
    position: relative;
    overflow: hidden;
}}
.g-result-pos {{
    background: linear-gradient(135deg, #f0fff9 0%, #e6fdf4 100%);
    border-color: {G_GREEN};
}}
.g-result-neg {{
    background: linear-gradient(135deg, #fff5f5 0%, #ffe8e8 100%);
    border-color: {G_RED};
}}
.g-result-icon {{
    font-size: 3.5rem;
    margin-bottom: 12px;
    display: block;
}}
.g-result-label {{
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.8px;
    margin-bottom: 8px;
}}
.g-result-conf {{
    font-size: 0.9rem;
    color: {G_MUTED};
    font-weight: 600;
}}
.g-conf-track {{
    background: rgba(0,0,0,0.06);
    border-radius: 99px;
    height: 8px;
    margin: 16px 0 8px;
    overflow: hidden;
}}
.g-conf-fill {{
    height: 100%;
    border-radius: 99px;
    transition: width 0.8s cubic-bezier(0.34, 1.56, 0.64, 1);
}}

/* ── EMPTY STATE ── */
.g-empty {{
    text-align: center;
    padding: 72px 24px;
    border: 2px dashed {G_BORDER};
    border-radius: 20px;
    background: {G_CARD};
}}
.g-empty-icon {{
    font-size: 3rem;
    margin-bottom: 16px;
    opacity: 0.5;
}}
.g-empty-text {{
    color: {G_MUTED};
    font-size: 0.95rem;
    font-weight: 500;
}}

/* ── FILE UPLOAD ── */
.stFileUploader section {{
    border: 2px dashed {G_BORDER} !important;
    border-radius: 16px !important;
    background: {G_CARD} !important;
    padding: 24px !important;
}}
.stDownloadButton > button {{
    background: {G_GRAD} !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    padding: 12px 24px !important;
    box-shadow: 0 4px 16px rgba(83,103,255,0.25) !important;
}}

/* ── METRICS ── */
div[data-testid="stMetric"] {{
    background: {G_CARD};
    border: 1.5px solid {G_BORDER};
    border-radius: 14px;
    padding: 16px 20px;
}}
div[data-testid="stMetricValue"] {{
    font-size: 1.6rem !important;
    font-weight: 800 !important;
    background: {G_GRAD};
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.stDataFrame {{ border-radius: 14px; overflow: hidden; border: 1.5px solid {G_BORDER} !important; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model_and_tokenizer():
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    model = load_model("sentiment_model.keras")
    with open("tokenizer.json") as f:
        tok_json = json.load(f)
    tokenizer = tokenizer_from_json(tok_json)
    return model, tokenizer

@st.cache_data
def load_data():
    df = pd.read_csv("processed_reviews.csv")
    df["binary"]    = (df["score"] >= 4).astype(int)
    df["sentiment"] = df["binary"].map({0: "Negative", 1: "Positive"})
    return df

@st.cache_data
def load_raw():
    return pd.read_csv("raw_reviews.csv", parse_dates=["date"])


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════════════════

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def predict(model, tokenizer, text):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq    = tokenizer.texts_to_sequences([clean_text(text)])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    prob   = float(model.predict(padded, verbose=0)[0][0])
    label  = "Positive" if prob > 0.5 else "Negative"
    conf   = prob if prob > 0.5 else 1 - prob
    return label, conf, prob


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def chart_style():
    return dict(template=TEMPLATE, paper_bgcolor=G_BG, plot_bgcolor=G_CARD,
                font=dict(family="Plus Jakarta Sans", color=G_TEXT))

@st.cache_data
def chart_training():
    with open("training_history.json") as f:
        h = json.load(f)
    e   = list(range(1, len(h["accuracy"]) + 1))
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Accuracy per Epoch", "Loss per Epoch"),
                        horizontal_spacing=0.1)
    for name, data, color, col, dash in [
        ("Train Acc",  h["accuracy"],     G_PURPLE, 1, "solid"),
        ("Val Acc",    h["val_accuracy"], G_GREEN,  1, "dash"),
        ("Train Loss", h["loss"],         G_PURPLE, 2, "solid"),
        ("Val Loss",   h["val_loss"],     G_RED,    2, "dash"),
    ]:
        fig.add_trace(go.Scatter(x=e, y=data, name=name,
                                  line=dict(color=color, width=2.5, dash=dash),
                                  mode="lines+markers",
                                  marker=dict(size=6, color=color)),
                      row=1, col=col)
    fig.update_layout(height=400, legend=dict(bgcolor="rgba(255,255,255,0.9)",
                      bordercolor=G_BORDER, borderwidth=1), **chart_style())
    return fig

@st.cache_data
def chart_cm(_model, _tokenizer):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    df   = load_data()
    seqs = _tokenizer.texts_to_sequences(df["clean_review"].fillna(""))
    X    = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")
    y    = df["binary"].values
    _, X_t, _, y_t = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_p  = (_model.predict(X_t, verbose=0) > 0.5).astype(int).flatten()
    cm   = confusion_matrix(y_t, y_p)
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    labels = ["Negative", "Positive"]
    text   = [[f"<b>{cm_n[i][j]:.2f}</b><br><span style='font-size:11px'>({cm[i][j]} reviews)</span>"
               for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=cm_n, x=labels, y=labels, text=text,
        texttemplate="%{text}", textfont=dict(size=14),
        colorscale=[[0, "#f3f4ff"], [0.5, "#a0adff"], [1, G_PURPLE]],
        showscale=False,
    ))
    fig.update_layout(title="", xaxis_title="Predicted",
                      yaxis_title="Actual", height=380, **chart_style())
    return fig

@st.cache_data
def chart_dist():
    df  = load_data()
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Sentiment Split", "Score Distribution"),
                        specs=[[{"type":"pie"},{"type":"bar"}]],
                        horizontal_spacing=0.12)
    c = df["sentiment"].value_counts()
    fig.add_trace(go.Pie(
        labels=c.index.tolist(), values=c.values.tolist(),
        marker=dict(colors=[G_RED, G_GREEN],
                    line=dict(color="white", width=3)),
        hole=0.5, textfont=dict(size=13),
        pull=[0.03, 0.03],
    ), row=1, col=1)
    sc = df["score"].value_counts().sort_index()
    fig.add_trace(go.Bar(
        x=sc.index.astype(str), y=sc.values,
        marker=dict(color=[G_RED if s<=3 else G_GREEN for s in sc.index],
                    line=dict(color="white", width=2),
                    cornerradius=6),
        text=sc.values, textposition="outside", showlegend=False,
    ), row=1, col=2)
    fig.update_layout(height=420,
                      legend=dict(bgcolor="rgba(255,255,255,0.9)"),
                      **chart_style())
    return fig

@st.cache_data
def chart_lengths():
    df = load_data()
    df["word_count"] = df["clean_review"].fillna("").apply(lambda x: len(x.split()))
    fig = px.histogram(df, x="word_count", color="sentiment",
                       barmode="overlay", nbins=50, opacity=0.75,
                       color_discrete_map={"Negative": G_RED, "Positive": G_GREEN},
                       labels={"word_count": "Word Count"},
                       template=TEMPLATE)
    for label, color in [("Negative", G_RED), ("Positive", G_GREEN)]:
        m = df[df["sentiment"]==label]["word_count"].mean()
        fig.add_vline(x=m, line_dash="dash", line_color=color,
                      annotation_text=f"{label}: {m:.0f}w",
                      annotation_font_color=color, annotation_font_size=12)
    fig.update_layout(height=400, paper_bgcolor=G_BG, plot_bgcolor=G_CARD,
                      legend=dict(bgcolor="rgba(255,255,255,0.9)"),
                      font=dict(family="Plus Jakarta Sans"))
    return fig

@st.cache_data
def chart_words():
    df   = load_data()
    stop = {"app","groww","the","and","is","it","to","a","of","in","for","i",
            "this","that","with","my","on","are","was","has","have","be","an",
            "at","by","from","but","not","so","we","they","or","as","its","me",
            "you","can","do","if","all","just","very","s","t","re","ve","2","1",
            "use","using","used","good","great","nice","best","also","even","like"}
    def top(series, n=15):
        words = " ".join(series.fillna("")).split()
        words = [w for w in words if w not in stop and len(w)>2]
        return pd.DataFrame(Counter(words).most_common(n),
                             columns=["word","count"]).sort_values("count")
    neg = top(df[df["binary"]==0]["clean_review"])
    pos = top(df[df["binary"]==1]["clean_review"])
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Top Words — Negative","Top Words — Positive"),
                        horizontal_spacing=0.18)
    fig.add_trace(go.Bar(x=neg["count"], y=neg["word"], orientation="h",
                         marker=dict(color=G_RED, cornerradius=4),
                         showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=pos["count"], y=pos["word"], orientation="h",
                         marker=dict(color=G_GREEN, cornerradius=4),
                         showlegend=False), row=1, col=2)
    fig.update_layout(height=500, **chart_style())
    return fig

@st.cache_data
def chart_time():
    df = load_raw()
    df["binary"] = (df["score"] >= 4).astype(int)
    df = df.dropna(subset=["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    m = df.groupby("month")["binary"].agg(["mean","count"]).reset_index()
    m.columns = ["month","pos_ratio","count"]
    m = m[m["count"] >= 3]
    if len(m) < 2: return None
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=m["month"], y=m["pos_ratio"], name="Positive Ratio",
        line=dict(color=G_GREEN, width=3),
        mode="lines+markers", marker=dict(size=8, color=G_GREEN),
        fill="tozeroy", fillcolor="rgba(0,208,156,0.08)",
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=m["month"], y=m["count"], name="Review Volume",
        marker=dict(color=G_PURPLE, opacity=0.2, cornerradius=4),
    ), secondary_y=True)
    fig.add_hline(y=0.5, line_dash="dot", line_color=G_MUTED, opacity=0.5)
    fig.update_layout(height=420,
                      legend=dict(bgcolor="rgba(255,255,255,0.9)"),
                      xaxis=dict(tickangle=-40),
                      **chart_style())
    fig.update_yaxes(title_text="Positive Ratio", secondary_y=False,
                     tickformat=".0%", range=[0,1.1])
    fig.update_yaxes(title_text="Review Count", secondary_y=True)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════════════

if "page" not in st.session_state:
    st.session_state.page = "predictor"

model, tokenizer = load_model_and_tokenizer()
df = load_data()
total    = len(df)
pos_pct  = df["binary"].mean() * 100
neg_pct  = 100 - pos_pct


# ══════════════════════════════════════════════════════════════════════════════
# TOP NAV
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="g-nav">
  <div class="g-nav-logo">
    <div class="g-logo-mark">G</div>
    <div>
      <div class="g-logo-name">Groww</div>
      <div class="g-logo-sub">Stocks &amp; Mutual Funds</div>
    </div>
    <div class="g-badge">Sentiment AI</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="g-hero">
  <div class="g-hero-eyebrow">
    NLP · RNN · Google Play Data
  </div>
  <div class="g-hero-title">
    Groww Review<br><span>Sentiment Analyser</span>
  </div>
  <div class="g-hero-sub">
    Scraped {total:,} real Play Store reviews · trained a deep learning model
    to classify user sentiment · achieved <strong>88% accuracy</strong> on real-world imbalanced data.
  </div>
  <div class="g-metrics">
    <div class="g-metric">
      <div class="g-metric-val">{total:,}</div>
      <div class="g-metric-lbl">Reviews Scraped</div>
    </div>
    <div class="g-metric">
      <div class="g-metric-val">88%</div>
      <div class="g-metric-lbl">Model Accuracy</div>
    </div>
    <div class="g-metric">
      <div class="g-metric-val">{pos_pct:.0f}%</div>
      <div class="g-metric-lbl">Positive Reviews</div>
    </div>
    <div class="g-metric">
      <div class="g-metric-val">{neg_pct:.0f}%</div>
      <div class="g-metric-lbl">Negative Reviews</div>
    </div>
    <div class="g-metric">
      <div class="g-metric-val">150</div>
      <div class="g-metric-lbl">Max Sequence Len</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# BODY
# ══════════════════════════════════════════════════════════════════════════════

nav_col, content_col = st.columns([1, 4.5], gap="small")

with nav_col:
    st.markdown('<div class="g-nav-label">Predict</div>', unsafe_allow_html=True)
    if st.button("🔍  Live Predictor",  key="p1"): st.session_state.page = "predictor"
    if st.button("📂  Batch Predict",   key="p2"): st.session_state.page = "batch"
    st.markdown('<div class="g-nav-label">Analytics</div>', unsafe_allow_html=True)
    if st.button("📈  Training Curves", key="p3"): st.session_state.page = "training"
    if st.button("🎯  Confusion Matrix",key="p4"): st.session_state.page = "cm"
    if st.button("🥧  Distributions",   key="p5"): st.session_state.page = "dist"
    if st.button("📏  Review Lengths",  key="p6"): st.session_state.page = "lengths"
    if st.button("💬  Top Words",       key="p7"): st.session_state.page = "words"
    if st.button("📅  Sentiment Trend", key="p8"): st.session_state.page = "time"
    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.75rem; color:{G_MUTED}; line-height:2; padding:0 4px;'>
      <b style='color:{G_TEXT}'>Architecture</b><br>
      Embedding(32d)<br>
      GlobalAvgPool<br>
      BatchNorm<br>
      Dense(64) → Dense(32)<br>
      Sigmoid output<br><br>
      <b style='color:{G_GREEN}'>88.0%</b> accuracy<br>
      <b style='color:{G_PURPLE}'>0.80</b> macro F1
    </div>
    """, unsafe_allow_html=True)

with content_col:
    page = st.session_state.page

    # ── LIVE PREDICTOR ────────────────────────────────────────────────────────
    if page == "predictor":
        st.markdown('<div class="g-page-title">🔍 Live Predictor</div>', unsafe_allow_html=True)
        st.markdown('<div class="g-page-sub">Type any Groww app review and get an instant AI-powered sentiment prediction.</div>', unsafe_allow_html=True)
        st.markdown('<hr class="g-divider">', unsafe_allow_html=True)

        left, right = st.columns([1.2, 1], gap="large")

        with left:
            examples = {
                "Select an example...": "",
                "😊 Very happy user":   "This app is absolutely fantastic! Investing in mutual funds has never been easier. Super clean UI, fast transactions, love it.",
                "😡 Frustrated user":   "Worst app ever. Keeps crashing when I try to place orders. Lost money because of this bug. Customer support doesn't respond.",
                "😐 Mixed review":      "App is okay, does the job but can be slow sometimes. The SIP feature is nice but the portfolio view needs improvement.",
            }
            choice = st.selectbox("Try an example:", list(examples.keys()))
            text   = st.text_area("Or write your own review:",
                                   value=examples[choice], height=150,
                                   placeholder="e.g. Great app for investing in mutual funds...")
            st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
            btn = st.button("Analyse Sentiment →", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            if btn and text.strip():
                with st.spinner(""):
                    label, conf, prob = predict(model, tokenizer, text)
                is_pos  = label == "Positive"
                color   = G_GREEN if is_pos else G_RED
                css     = "g-result-pos" if is_pos else "g-result-neg"
                icon    = "📈" if is_pos else "📉"
                bar_w   = f"{prob*100:.1f}%" if is_pos else f"{(1-prob)*100:.1f}%"

                st.markdown(f"""
                <div class="g-result {css}">
                  <span class="g-result-icon">{icon}</span>
                  <div class="g-result-label" style="color:{color};">{label}</div>
                  <div class="g-result-conf">Confidence: <strong>{conf*100:.1f}%</strong></div>
                  <div class="g-conf-track">
                    <div class="g-conf-fill"
                         style="width:{bar_w}; background:{color};"></div>
                  </div>
                  <div style="display:flex; justify-content:space-between;
                               font-size:0.72rem; color:{G_MUTED}; font-weight:600;">
                    <span>Negative ←</span><span>→ Positive</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(prob * 100, 1),
                    number={"suffix": "%", "font": {"color": G_PURPLE, "size": 32,
                             "family": "Plus Jakarta Sans"}},
                    title={"text": "Positive Probability",
                           "font": {"color": G_MUTED, "size": 13,
                                    "family": "Plus Jakarta Sans"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": G_BORDER,
                                 "tickwidth": 1},
                        "bar":  {"color": G_GREEN if prob > 0.5 else G_RED,
                                 "thickness": 0.25},
                        "bgcolor": G_CARD,
                        "borderwidth": 0,
                        "steps": [{"range":[0,50],  "color":"#fff5f5"},
                                  {"range":[50,100],"color":"#f0fff9"}],
                        "threshold": {"line":{"color": G_PURPLE,"width":3},
                                      "thickness": 0.8, "value": 50},
                    }
                ))
                fig_g.update_layout(
                    height=240, paper_bgcolor=G_BG,
                    font=dict(family="Plus Jakarta Sans"),
                    margin=dict(t=40, b=0, l=30, r=30)
                )
                st.plotly_chart(fig_g, use_container_width=True)

            elif btn:
                st.warning("Please enter a review to analyse.")
            else:
                st.markdown(f"""
                <div class="g-empty">
                  <div class="g-empty-icon">📝</div>
                  <div class="g-empty-text">Enter a review on the left and click<br>
                  <strong>Analyse Sentiment</strong> to see the prediction</div>
                </div>
                """, unsafe_allow_html=True)

    # ── BATCH ─────────────────────────────────────────────────────────────────
    elif page == "batch":
        st.markdown('<div class="g-page-title">📂 Batch Prediction</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="g-page-sub">Upload a CSV with a <code>review</code> column — the model predicts sentiment for every row instantly.</div>', unsafe_allow_html=True)
        st.markdown('<hr class="g-divider">', unsafe_allow_html=True)

        up = st.file_uploader("Upload CSV", type=["csv"])
        if up:
            bdf = pd.read_csv(up)
            if "review" not in bdf.columns:
                st.error("CSV must contain a 'review' column.")
            else:
                with st.spinner(f"Predicting {len(bdf)} reviews..."):
                    lbls, confs, probs = [], [], []
                    for t in bdf["review"].fillna(""):
                        l, c, p = predict(model, tokenizer, t)
                        lbls.append(l); confs.append(round(c*100,2)); probs.append(round(p,4))
                bdf["sentiment"]  = lbls
                bdf["confidence"] = confs
                bdf["pos_prob"]   = probs
                pos_n = lbls.count("Positive"); neg_n = lbls.count("Negative")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total", len(lbls))
                c2.metric("Positive", f"{pos_n} ({pos_n/len(lbls)*100:.1f}%)")
                c3.metric("Negative", f"{neg_n} ({neg_n/len(lbls)*100:.1f}%)")
                fig_p = px.pie(values=[pos_n,neg_n], names=["Positive","Negative"],
                                color_discrete_sequence=[G_GREEN,G_RED], hole=0.5,
                                template=TEMPLATE)
                fig_p.update_layout(height=320, paper_bgcolor=G_BG)
                st.plotly_chart(fig_p, use_container_width=True)
                st.dataframe(bdf[["review","sentiment","confidence"]].head(50),
                              use_container_width=True, hide_index=True)
                st.download_button("⬇ Download Results",
                                    data=bdf.to_csv(index=False).encode(),
                                    file_name="sentiment_predictions.csv",
                                    mime="text/csv", use_container_width=True)
        else:
            st.markdown(f"""
            <div class="g-empty">
              <div class="g-empty-icon">📂</div>
              <div class="g-empty-text">Upload a CSV with a <strong>review</strong> column to begin</div>
            </div>
            """, unsafe_allow_html=True)

    # ── ANALYTICS ─────────────────────────────────────────────────────────────
    elif page == "training":
        st.markdown('<div class="g-page-title">📈 Training Curves</div>', unsafe_allow_html=True)
        st.markdown('<div class="g-page-sub">Model accuracy and loss across training epochs. Early stopping restored best weights from epoch 5.</div>', unsafe_allow_html=True)
        st.markdown('<hr class="g-divider">', unsafe_allow_html=True)
        st.plotly_chart(chart_training(), use_container_width=True)

    elif page == "cm":
        st.markdown('<div class="g-page-title">🎯 Confusion Matrix</div>', unsafe_allow_html=True)
        st.markdown('<div class="g-page-sub">How well the model distinguishes Negative vs Positive reviews on the held-out test set.</div>', unsafe_allow_html=True)
        st.markdown('<hr class="g-divider">', unsafe_allow_html=True)
        st.plotly_chart(chart_cm(model, tokenizer), use_container_width=True)

    elif page == "dist":
        st.markdown('<div class="g-page-title">🥧 Distributions</div>', unsafe_allow_html=True)
        st.markdown('<div class="g-page-sub">Real sentiment split and star rating distribution — no artificial balancing applied.</div>', unsafe_allow_html=True)
        st.markdown('<hr class="g-divider">', unsafe_allow_html=True)
        st.plotly_chart(chart_dist(), use_container_width=True)

    elif page == "lengths":
        st.markdown('<div class="g-page-title">📏 Review Lengths</div>', unsafe_allow_html=True)
        st.markdown('<div class="g-page-sub">Word count distribution — negative reviews tend to be longer and more detailed.</div>', unsafe_allow_html=True)
        st.markdown('<hr class="g-divider">', unsafe_allow_html=True)
        st.plotly_chart(chart_lengths(), use_container_width=True)

    elif page == "words":
        st.markdown('<div class="g-page-title">💬 Top Words</div>', unsafe_allow_html=True)
        st.markdown('<div class="g-page-sub">Most frequent words in Negative vs Positive reviews after stopword removal.</div>', unsafe_allow_html=True)
        st.markdown('<hr class="g-divider">', unsafe_allow_html=True)
        st.plotly_chart(chart_words(), use_container_width=True)

    elif page == "time":
        st.markdown('<div class="g-page-title">📅 Sentiment Trend</div>', unsafe_allow_html=True)
        st.markdown('<div class="g-page-sub">How Groww user sentiment has shifted over time alongside review volume.</div>', unsafe_allow_html=True)
        st.markdown('<hr class="g-divider">', unsafe_allow_html=True)
        fig = chart_time()
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough date range in the current dataset for a time trend.")
# 📈 Groww Review Sentiment Analyser

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-88%25-00d09c?style=for-the-badge)

<br/>

**A production-grade NLP pipeline that scrapes real Groww app reviews from Google Play, trains a deep learning sentiment classifier, and serves live predictions through an interactive Streamlit dashboard.**

<br/>

<a href="https://groww-sentiment-analyser-wsoegxtgshtgdfdyzbpc2o.streamlit.app">
  <img src="https://img.shields.io/badge/🚀%20Live%20Demo-Click%20Here-00d09c?style=for-the-badge&logoColor=white" alt="Live Demo"/>
</a>
&nbsp;
<a href="#-results">
  <img src="https://img.shields.io/badge/📊%20Results-View-5367ff?style=for-the-badge&logoColor=white" alt="Results"/>
</a>
&nbsp;
<a href="#-run-locally">
  <img src="https://img.shields.io/badge/🛠%20Run%20Locally-Guide-FF6F00?style=for-the-badge&logoColor=white" alt="Run Locally"/>
</a>

</div>

---

## 🎯 Overview

Most sentiment analysis tutorials use clean, pre-packaged datasets. This project does it the hard way — scraping real, messy, imbalanced data from the wild and building a model that actually handles it.

| Step | What happens |
|------|-------------|
| **Scrape** | 5,759 real Groww app reviews from Google Play Store |
| **Preprocess** | Clean text, tokenize, pad sequences, stratified split |
| **Train** | Binary classifier with tuned class weights for real imbalance |
| **Deploy** | Live Streamlit app — predict, explore analytics, batch process |

---

## 🏗 Architecture

```
Google Play Store
       ↓
  scraper.py       →  raw_reviews.csv        (5,759 reviews, natural distribution)
       ↓
 preprocess.py     →  X_train / X_test / tokenizer.json
       ↓
   train.py        →  sentiment_model.keras  (88% test accuracy)
       ↓
    app.py         →  Streamlit dashboard    (live predictor + 6 analytics charts)
```

### Model

```
Embedding(10000, 32)
        ↓
GlobalAveragePooling1D    ← no vanishing gradients on short reviews
        ↓
BatchNormalization
        ↓
Dense(64, relu) + Dropout(0.5)
        ↓
Dense(32, relu) + Dropout(0.3)
        ↓
Dense(1, sigmoid)         ← Negative / Positive
```

---

## 📊 Results

### Classification Report

| | Precision | Recall | F1 |
|---|---|---|---|
| **Negative** | 0.77 | 0.61 | 0.68 |
| **Positive** | 0.90 | 0.95 | 0.92 |
| **Overall** | — | — | **88% accuracy · 0.80 macro F1** |

### Real Data Distribution

```
5★  ████████████████████████████  71.7%  (4,131 reviews)
1★  ██████                        16.1%  (  927 reviews)
4★  ██                             7.1%  (  410 reviews)
3★  █                              2.8%  (  161 reviews)
2★                                 2.3%  (  130 reviews)
```

> No oversampling. Class imbalance handled via **tuned class weights** — Negative class given 6.6× penalty during training so the model doesn't just predict Positive for everything.

---

## 🗂 Project Structure

```
groww-sentiment-analyser/
├── scraper.py             # Google Play scraper — 5000+ reviews, natural distribution
├── preprocess.py          # Text cleaning, tokenization, stratified train/test split
├── train.py               # Model architecture, class weights, training, evaluation
├── visualize.py           # 6 interactive Plotly charts saved as HTML
├── app.py                 # Streamlit dashboard — live predictor + analytics
├── download_model.py      # Auto-downloads model from Google Drive at runtime
├── processed_reviews.csv  # Cleaned dataset used by analytics charts
├── raw_reviews.csv        # Raw scraped reviews
├── training_history.json  # Epoch-by-epoch accuracy and loss
├── tokenizer.json         # Fitted Keras tokenizer
├── requirements.txt       # Dependencies
└── .python-version        # Python 3.12 pin for Streamlit Cloud
```

---

## 🌐 App Features

| Page | Feature |
|------|---------|
| 🔍 Live Predictor | Type any review → sentiment label + confidence gauge |
| 📂 Batch Predict | Upload CSV → bulk predictions + download results |
| 📈 Training Curves | Accuracy & loss per epoch, train vs validation |
| 🎯 Confusion Matrix | Normalised heatmap with raw counts |
| 🥧 Distributions | Real sentiment split + star rating bar chart |
| 📏 Review Lengths | Word count histogram by sentiment class |
| 💬 Top Words | Most frequent words — Negative vs Positive |
| 📅 Sentiment Trend | Monthly positive ratio + review volume over time |

---

## 🛠 Run Locally

```bash
# 1. Clone
git clone https://github.com/065010-AmanMalhi/groww-sentiment-analyser.git
cd groww-sentiment-analyser

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the pipeline (in order)
python scraper.py       # → raw_reviews.csv
python preprocess.py    # → X_train.npy, tokenizer.json
python train.py         # → sentiment_model.keras
python visualize.py     # → plots/

# 5. Launch
streamlit run app.py
```

---

## 🧠 Key Decisions

**Why GlobalAveragePooling instead of GRU/LSTM?**
App reviews are short (avg 41 words). GRUs need large datasets to avoid vanishing gradients — on 5K reviews, GlobalAveragePooling consistently outperformed GRU by 5-8% and trained 10× faster.

**Why class weights instead of oversampling?**
SMOTE doesn't work on token sequences. Random oversampling on a 72/28 split just duplicates data — class weights achieve the same correction without touching the real distribution.

**Why Groww?**
Finance app reviews have clear, opinionated language — users either love the UX or are furious about crashes and missing money. That polarity makes for a strong sentiment signal.

---

## 📦 Tech Stack

`Python 3.12` · `TensorFlow 2.20` · `Keras 3.13` · `Streamlit` · `Plotly` · `scikit-learn` · `google-play-scraper` · `pandas` · `gdown`

---

<div align="center">
Built as Project 1 of RNN <br/>
</div>

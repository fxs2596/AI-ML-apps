**AI/ML Sentiment Analysis Portfolio Projects**

A collection of two end-to-end Python and Flask applications for text-based sentiment analysis, demonstrating the full ML pipeline, handling real-world data challenges (e.g., class imbalance), and integrating models into web interfaces.

---

## 🚀 Table of Contents

1. [Multi-Class Movie Overview Sentiment Analyzer](#multi-class-movie-overview-sentiment-analyzer)
2. [Binary Movie Review Sentiment Analyzer](#binary-movie-review-sentiment-analyzer)
3. [Getting Started](#getting-started)
4. [Future Improvements](#future-improvements)

---

## 1. Multi-Class Movie Overview Sentiment Analyzer

Classifies IMDb Top 1000 movie overviews into six sentiment categories based on rating:

- **Excellent** (≥ 9.0)
- **Great** (8.5–8.9)
- **Very Good** (8.0–8.4)
- **Good** (7.0–7.9)
- **One-Time Watch!** (6.0–6.9)
- **Needs Improvement** (< 6.0)

### 🔗 Demo

Run locally after setup and visit `http://127.0.0.1:5000/` in your browser.

### 📂 Project Structure

```
multi_class_sentiment_app/
├── senti.py              # Data processing, training, evaluation, saving
├── app.py                # Flask web server
├── templates/
│   └── index.html        # UI for user input and predictions
├── imdb_top_1000.csv     # Raw dataset
└── models/               # Generated after running senti.py
    ├── svc_model.joblib
    └── tfidf_vectorizer.joblib
```

> **Note:** `.joblib` files are created when you run `senti.py`.

### 📊 Data Source & Distribution

- **IMDb Top 1000 Dataset** (download [`imdb_top_1000.csv`](imdb_top_1000.csv)).
- Sentiment labels derived from the `IMDB_Rating` column.
- Class imbalance overview:

![Multi-Class Sentiment Distribution](multi_class_sentiment_distribution.png)

### 🛠️ Features

- **Preprocessing:** Cleaning (HTML tags, non-alphabetic chars), tokenization, stop-word removal
- **Vectorization:** TF–IDF
- **Model:** SVC (linear kernel) with `class_weight='balanced'`
- **Evaluation:** Accuracy, weighted precision/recall/F1 for imbalanced data
- **Persistence:** Saved model & vectorizer via `joblib`
- **Web App:** Predicts sentiment for new overviews

### 📈 Results

| Metric                  | Score  |
|-------------------------|-------:|
| Accuracy                | 0.4950 |
| Precision (weighted)    | 0.5085 |
| Recall (weighted)       | 0.4950 |
| F1‑Score (weighted)     | 0.4813 |

<details>
<summary>Confusion Matrix & Classification Report</summary>

![Confusion Matrix](multi_class_confusion_matrix.png)

```text
              precision  recall  f1-score  support

Excellent       0.00      0.00      0.00       1
Good            0.54      0.63      0.58     107
Great           1.00      0.10      0.18      10
Very Good       0.41      0.38      0.39      82

accuracy        0.49      200
macro avg       0.49      0.28      0.29     200
weighted avg    0.51      0.49      0.48     200
```
</details>

### 🔍 Lessons Learned

- Real-world imbalance drastically affects minority-class performance.
- Weighted metrics are essential beyond simple accuracy.
- Tried different models (Logistic Regression, Naive Bayes, SMOTE), SVC performed best.

---

## 2. Binary Movie Review Sentiment Analyzer

Binary classification (Positive vs. Negative) on custom reviews dataset.

### 🔗 Demo

Run locally after setup and visit `http://127.0.0.1:5001/`.

### 📂 Project Structure

```
binary_sentiment_app/
├── senti2.py             # Data pipeline & model training
├── app2.py               # Flask application
├── templates2/
│   └── index2.html       # UI for binary sentiment
├── movie.csv             # Reviews dataset
└── models/               # Saved artifacts after running senti2.py
    ├── lr_model.joblib
    └── tfidf_vectorizer_v2.joblib
```

> **Note:** `.joblib` files are created by `senti2.py`.

### 📊 Data Source & Distribution

- **Dataset:** `movie.csv` with `text` and `label` (0=Negative, 1=Positive)

![Binary Sentiment Distribution](binary_sentiment_distribution.png)

### 🛠️ Features

- **Preprocessing:** Cleaning, tokenization, stop-word removal
- **Vectorization:** TF–IDF
- **Model:** Logistic Regression with `class_weight='balanced'`
- **Evaluation:** Accuracy, precision/recall/F1, ROC AUC
- **Persistence:** Saved model & vectorizer via `joblib`
- **Web App:** Predicts positive/negative for new reviews

### 📈 Results

| Metric                | Score    |
|-----------------------|---------:|
| Accuracy              | 0.8866   |
| Precision (Positive)  | 0.8826   |
| Recall (Positive)     | 0.8916   |
| F1-Score (Positive)   | 0.8871   |
| ROC AUC               | 0.9557   |

<details>
<summary>Confusion Matrix & Classification Report</summary>

![Confusion Matrix](binary_confusion_matrix.png)

```text
             precision  recall  f1-score  support

Negative      0.89      0.88      0.89    4004
Positive      0.88      0.89      0.89    3996

accuracy      0.89      8000
macro avg     0.89      0.89      0.89    8000
weighted avg  0.89      0.89      0.89    8000
```
</details>

---

## Getting Started

1. **Clone repository:**
   ```bash
   git clone https://github.com/[YourUsername]/AI-ML-apps.git
   ```
2. **Create & activate** a Python virtual environment.
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Download NLTK data:**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```
5. **Run projects:**
   - Multi-class:
     ```bash
     cd multi_class_sentiment_app
     python senti.py
     python app.py
     ```
   - Binary:
     ```bash
     cd binary_sentiment_app
     python senti2.py
     python app2.py
     ```

Access each web app at:  
- Multi-class: `http://127.0.0.1:5000/`  
- Binary:      `http://127.0.0.1:5001/`

---

## Future Improvements

- Advanced imbalance handling: SMOTE variants, ensemble resampling
- Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- Explore deep learning / sentence embeddings (e.g., BERT)
- Enhance web UI: prediction probabilities, user feedback loop


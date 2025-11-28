# Fake News Detection: Evaluating Model Generalization Across ISOT and WELFake Datasets

## Overview

This repository contains the code, analysis, and documentation for **Capstone Three**, a machine learning project focused on fake news detection. The core objective is to build and evaluate classifiers that detect fake news articles, with a particular emphasis on assessing model generalization when trained on one dataset (ISOT or WELFake) and tested on the other. This addresses the challenge of domain adaptation in misinformation detection, where models often overfit to specific sources.

Key goals:
- Perform data wrangling, exploratory data analysis (EDA), and feature engineering on two benchmark datasets.
- Train and compare multiple ML models (e.g., Logistic Regression, SVM, Random Forest, BERT-based models).
- Quantify performance drops in cross-dataset scenarios to highlight generalization issues.
- Provide insights for improving robust fake news detectors.

The project demonstrates skills in NLP, scikit-learn, transformers (Hugging Face), pandas, and visualization tools. It was developed as part of a data science capstone program.

## Datasets

- **ISOT Dataset**: ~45,000 articles (fake/real) from credible sources like Reuters and fake news sites. Features: title, text, subject, date.
- **WELFake Dataset**: ~72,000 articles labeled as fake/real, sourced from diverse web scrapes. Focuses on textual content for binary classification.

Both datasets are loaded via CSV files (available in `data/raw/`). Preprocessing handles text cleaning, tokenization, and imbalance.

## Technologies Used

| Category          | Tools/Libraries                          |
|-------------------|------------------------------------------|
| **Language**      | Python 3.9+                             |
| **Data Manipulation** | pandas, numpy                          |
| **NLP/ML**        | scikit-learn, transformers (BERT), XGBoost |
| **Visualization** | Matplotlib, Seaborn, WordCloud          |
| **Environment**   | Jupyter Notebook, Google Colab          |
| **Other**         | NLTK for tokenization, imbalanced-learn for sampling |

## Project Structure

## Installation

1. **Clone the Repository**
2. **Set Up Environment**
3. **Download Datasets**
- ISOT: Download from [Kaggle ISOT](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) and place in `data/raw/`.
- WELFake: Download from [Kaggle WELFake](https://www.kaggle.com/datasets/andreaisb/welfake-dataset) and place in `data/raw/`.
- Or use notebook code to fetch via URLs.

## Usage

### Run the Analysis Pipeline
- Launch Jupyter: `jupyter notebook`
- Execute notebooks sequentially:
1. `01_data_wrangling_EDA.ipynb`: Load data, handle missing values, generate EDA plots (e.g., word clouds for fake vs. real, length distributions).
2. Feature engineering notebooks: Apply TF-IDF, n-grams, or embeddings; handle class imbalance with SMOTE.
3. Modeling notebooks: Train baselines (LR, SVM) and advanced (BERT fine-tuned); evaluate with cross-validation.
4. Comparison notebook: Test ISOT-trained models on WELFake (and vice versa); analyze generalization gaps.
5. Presentation: Render slides for overview.

### Key Code Example (from Modeling Notebook)
```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

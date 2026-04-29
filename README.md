# Hate Speech Detector

An NLP-based web application that classifies tweets into three categories: **Hate Speech**, **Offensive Language**, and **None of the Above**. The project explores multiple machine learning approaches — from traditional Naive Bayes to transformer-based RoBERTa — and ships a Flask web app for real-time predictions.

## Project Overview

This project follows the CRISP-DM methodology to build a hate speech classification pipeline. Tweets were collected, manually labeled, preprocessed, and used to train several models. The best-performing traditional model (Naive Bayes with TF-IDF) is deployed through a lightweight Flask interface where users can input a tweet and receive classification probabilities.

## Repository Structure

```
hatespeechdetector/
│
├── app.py                         # Flask web application
├── index.html                     # Input form (frontend)
├── result.html                    # Classification results page (frontend)
│
├── modelling.ipynb                # Core modelling notebook (preprocessing, training, evaluation)
├── bayesnaiveclass.ipynb          # Naive Bayes classifier experiments
├── Roberta_class.ipynb            # RoBERTa transformer-based classification
├── palestine_hatespeech.ipynb     # Topic-specific analysis (Palestine/Israel tweets)
├── notebook1c20a1fb3f.ipynb       # Additional exploratory notebook
│
├── naive_bayes_model.pkl          # Trained Naive Bayes model (serialized)
├── one_vs_rest_classifier.pkl     # One-vs-Rest classifier (serialized)
├── vectorizer.pkl                 # TF-IDF vectorizer (serialized)
│
├── 25k_dataset.csv                # Full 25k tweet dataset
├── twits_25k_balanced.csv         # Balanced version of the 25k dataset
├── twits_25k_preprocessed.csv     # Preprocessed tweets
├── hatespeech_train.csv           # Training dataset
├── prepared_hatespeech_train.csv  # Cleaned training dataset
├── train_dataset.csv              # Train split
├── test_dataset.csv               # Test split
│
└── README.md
```

## Models

### Naive Bayes (deployed)
A Multinomial Naive Bayes classifier trained on TF-IDF features. This is the model served by the Flask app. It outputs class probabilities for all three categories.

### One-vs-Rest Classifier
An alternative approach that trains a separate binary classifier per class, also included as a serialized `.pkl` file.

### RoBERTa (experimental)
A fine-tuned RoBERTa transformer model explored in `Roberta_class.ipynb` for improved contextual understanding of tweet content.

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/Aylin1/hatespeechdetector.git
cd hatespeechdetector
pip install flask joblib scikit-learn
```

### Running the App

```bash
python app.py
```

Then open your browser and navigate to `http://127.0.0.1:5000`. Enter a tweet in the text field, hit **Submit**, and the app will return the predicted probabilities for each class.

### Running the Notebooks

To explore the data analysis and model training:

```bash
pip install jupyter pandas numpy scikit-learn transformers torch
jupyter notebook
```

Open any of the `.ipynb` files to step through the experiments.

## Data

The project uses multiple CSV datasets derived from ~25,000 tweets. The data pipeline includes:

1. **Collection** — Tweets gathered via the Twitter API (using Tweepy), with a focus on the Palestine/Israel topic.
2. **Labeling** — Manual classification into Hate Speech, Offensive Language, or Neither.
3. **Preprocessing** — Text cleaning, tokenization, and balancing of class distributions.
4. **Splitting** — Standard train/test split for model evaluation.

## Tech Stack

- **Backend:** Flask, joblib, scikit-learn
- **NLP/ML:** Naive Bayes, One-vs-Rest, TF-IDF, RoBERTa (Hugging Face Transformers)
- **Data:** pandas, NumPy
- **Frontend:** HTML (Jinja2 templates)

## Acknowledgments

Forked from [pffaundez/hatespeechdetector](https://github.com/pffaundez/hatespeechdetector).

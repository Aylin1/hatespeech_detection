# Hate Speech Detection on Twitter

An NLP-based classification pipeline and web application that classifies tweets into three categories: **Hate Speech**, **Offensive Language**, and **None of the Above**. The project compares multiple machine learning approaches (Naive Bayes, RoBERTa, DistilBERT) and deploys the best-performing model via a Flask web app and Hugging Face Hub.

> University project (team of 6) | M.Sc. Project Management & Data Science, HTW Berlin | CRISP-DM methodology

## Project Overview

This project follows the CRISP-DM methodology to build a hate speech classification pipeline for automatic content moderation in online forums. Tweets were collected via the Twitter API, manually labeled, preprocessed, and used to train several models. Three model architectures were compared, with **DistilBERT achieving the best results** across all metrics.

The best-performing traditional model (Naive Bayes with TF-IDF) is deployed through a lightweight Flask interface for local real-time predictions. The fine-tuned DistilBERT model was additionally deployed on the **Hugging Face Hub** for public access via the Inference API.

## Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Naive Bayes (baseline) | 75% | 77% | 75% | 76% |
| RoBERTa | 79% | 80% | 79% | 80% |
| **DistilBERT** | **84%** | **84%** | **84%** | **84%** |

DistilBERT was selected as the primary model due to its superior and balanced performance across all metrics, showing a significant improvement over the Naive Bayes baseline while maintaining efficiency.

## Repository Structure

```
hatespeech_detection/
|
├── app.py                         # Flask web application for real-time predictions
├── index.html                     # Input form (frontend)
├── result.html                    # Classification results page (frontend)
|
├── modelling.ipynb                # Core notebook: preprocessing, DistilBERT fine-tuning,
|                                  # feature engineering, grid search, evaluation
├── bayesnaiveclass.ipynb          # Naive Bayes classifier experiments with TF-IDF
├── Roberta_class.ipynb            # RoBERTa transformer-based classification
├── palestine_hatespeech.ipynb     # Topic-specific analysis (Palestine/Israel tweets)
|
├── naive_bayes_model.pkl          # Trained Naive Bayes model (serialized)
├── one_vs_rest_classifier.pkl     # One-vs-Rest classifier (serialized)
|
├── 25k_dataset.csv                # Full 25k tweet dataset
├── train_dataset.csv              # Train split
├── test_dataset.csv               # Test split
|
└── README.md
```

## Models

### Naive Bayes (Flask app)

A Multinomial Naive Bayes classifier trained on TF-IDF features. Serves as the baseline model and is deployed via the Flask web app. Outputs class probabilities for all three categories.

### DistilBERT (best performing, Hugging Face Hub)

A fine-tuned DistilBERT model explored in `modelling.ipynb`. Chosen for deployment on the Hugging Face Hub due to its efficiency and accuracy in understanding textual context. Preprocessing includes lowercasing, URL removal, spelling correction via TextBlob, emoji-to-text conversion, and mention removal.

### RoBERTa (experimental)

A fine-tuned RoBERTa transformer model explored in `Roberta_class.ipynb` using the Hugging Face Transformers library and TensorFlow/Keras. Tokenization with `RobertaTokenizer`, sequence padding to 128 tokens, and training with one-hot encoded labels across 3 classes.

## Deployment

- **Flask web app:** Local deployment serving the Naive Bayes model for real-time tweet classification with probability output.
- **Hugging Face Hub:** The fine-tuned DistilBERT model was hosted on the Hugging Face Hub, making it publicly accessible via the Inference API. The deployment supports both real-time and batch inference strategies.

## Data

The project uses multiple CSV datasets derived from ~25,000 tweets. The data pipeline includes:

1. **Collection:** Tweets gathered via the Twitter API (using Tweepy), with a focus on the Palestine/Israel topic.
2. **Labeling:** Classification into Hate Speech (0), Offensive Language (1), or Neither (2).
3. **Preprocessing:** Lowercasing, URL/mention removal, spelling correction, emoji-to-text conversion, hashtag cleaning, and class balancing via oversampling.
4. **Splitting:** Standard train/test split (80/20) for model evaluation.

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/Aylin1/hatespeech_detection.git
cd hatespeech_detection
pip install flask joblib scikit-learn pandas numpy transformers tensorflow torch textblob emoji imblearn seaborn matplotlib nltk
```

### Running the Web App

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser. Enter a tweet and get classification probabilities for each category.

### Running the Notebooks

```bash
jupyter notebook modelling.ipynb        # Core pipeline with DistilBERT
jupyter notebook Roberta_class.ipynb    # RoBERTa experiments
jupyter notebook bayesnaiveclass.ipynb  # Naive Bayes baseline
```

## Tech Stack

- **Backend:** Flask, joblib, scikit-learn
- **NLP/ML:** Naive Bayes, One-vs-Rest, TF-IDF, DistilBERT, RoBERTa (Hugging Face Transformers)
- **Deep Learning:** TensorFlow, Keras, PyTorch
- **Data:** pandas, NumPy, TextBlob, emoji, NLTK, imbalanced-learn
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Hugging Face Hub, Flask
- **Frontend:** HTML (Jinja2 templates)

## License

This project is for academic and research purposes.

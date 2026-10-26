# IMDB sentiment analysis

Individual coursework project. It compares text representations and models for
binary sentiment classification on the IMDB movie-review dataset, from sparse
baselines to a fine-tuned transformer, and includes error analysis.

All work is in [`notebooks/sentiment_analysis.ipynb`](notebooks/sentiment_analysis.ipynb),
with saved outputs. Figures are in [`outputs/figures/`](outputs/figures).

## What it covers

1. Exploratory analysis and an NLTK preprocessing pipeline (cleaning, stopword
   removal, lemmatisation)
2. Representations: bag-of-words, TF-IDF and Word2Vec
3. Classical models: logistic regression, naive Bayes and linear SVM
4. Neural models: bidirectional LSTM, and an LSTM with attention (with
   attention visualisation)
5. Fine-tuned `distilbert-base-uncased` (3 epochs, max length 256)
6. Error analysis and confidence-based thresholding

## Results

Evaluated on the standard IMDB test split (25,000 reviews, balanced).

| Model | Accuracy | F1 |
|---|---|---|
| Naive Bayes + BoW | 0.846 | 0.843 |
| Linear SVM + TF-IDF | 0.874 | 0.873 |
| Logistic regression + TF-IDF | 0.889 | 0.889 |
| Bidirectional LSTM | 0.860 | 0.857 |
| LSTM + attention | 0.877 | 0.877 |
| DistilBERT (fine-tuned) | 0.913 | 0.914 |

**Thresholding trade-off.** If DistilBERT makes a prediction only when its
confidence is at least 0.95, F1 on the reviews it does classify rises to 0.939.
It abstains on 6.4% of reviews (coverage 93.6%).

**Evaluation caveat.** There is no separate validation split. The test split
was used to monitor training and to choose the threshold, so the thresholded
figure, and to a lesser extent the neural-model results, are optimistic.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/sentiment_analysis.ipynb
```

The dataset downloads automatically through Hugging Face `datasets`
(`load_dataset("imdb")`). Seeds are fixed, but GPU training is not fully
deterministic. A GPU is recommended for the LSTM and DistilBERT sections.

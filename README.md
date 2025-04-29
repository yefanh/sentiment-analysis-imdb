# Sentiment Analysis of IMDb Movie Reviews using Traditional and Deep Learning Approaches
## Introduction
This is our final project for CS 6140 at Northeastern University. We aim to classify the IMDb reviews as positive and negative reviews using a multitude of models, such as SVM, Random Forest, XGBoost, LSTM, and BERT.
## Dataset
We utilized the dataset found at this link: https://www.kaggle.com/datasets/yasserh/imdb-movie-ratings-sentiment-analysis. It consists of 40,000 unique reviews, labeled as Positive (1) or Negative (0) sentiment, with an even 50/50 distribution of data.
## Preprocess
We preprocessed the data with multiple methods:
* HTML Tag Removal
* Lowercasing
* Removal of special characters and digits
* Stopword Removal
* Lemmatization
* Tokenization
* Joining Tokens

## Vectorization
Vectorization was done using the following methods:
* TF-IDF
* CountVectorizer
* GloVe Embeddings
* Deep Learning Tokenization (Keras' Tokenizer and bert-base-uncased tokenizer)
## Models
The models we utilized are:
* Support Vector Machine (SVM)
* Random Forest Classifier
* XG-Boost Classifier
* Long Short-Term Memory (LSTM)
* BERT
* 
## Result

<figure>
  <img 
  src="Results.png" 
  alt="Results of models" 
  width="8000" height="300">
</figure>

* Linear SVM is still the strongest traditional classifier on sparse vectors (0.888 accuracy).
* Tree ensembles (RF, XGBoost) trail SVM by about 3–4 pp6, but are less brittle to high-order n-grams and class imbalance.
* LSTM overtakes all classical baselines, reaching 0.888 accuracy with only 100-dim embeddings, confirming the benefit of sequence modelling.
* Fine-tuned BERT sets the new high-water mark (0.902 accuracy, 0.902 F1), showing that
contextual transformer features capture sentiment nuances missed by bag-of-words representations.

Overall, deep contextual models clearly surpass both sparse and static-embedding baselines,
but at the cost of ∼10× training time and GPU resources.

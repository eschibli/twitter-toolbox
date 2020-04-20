from ..tweet_sentiment_classifier import Classifier, tokenizer_filter

import pickle as pkl
import numpy as np

import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


class BoW_Model(Classifier):

    def __init__(self, vocab_size=100000, max_iter=10000, validation_split=0.2, accuracy=0, bootstrap=1,
                 remove_stopwords=True, remove_punctuation=True, lemmatize=True, **kwargs):
        """
        Constructor for BoW_Model
        Be sure to add additional parameters to export()
        :param vocab_size: (int) Maximum vocabulary size. Default 1E6
        :param max_iter: (int) Maximum number of fit iterations
        :param remove_punctuation: (Bool) Remove punctuation. Recommended.
        :param remove_stopwords: (Bool) Remove stopwords. Recommended.
        :param lemmatize: (Bool) Lemmatize words. Recommended.
        """
        self.package = 'twitter_nlp_toolkit.tweet_sentiment_classifier.models.bow_models'
        self.type = 'BoW_Model'
        self.vectorizer = None
        self.classifier = None
        self.vocab_size = vocab_size
        self.max_iter = max_iter
        self.validation_split = validation_split
        self.accuracy = accuracy
        self.bootstrap = bootstrap
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize

    def fit(self, train_data, y, weights=None, custom_vocabulary=None):
        """
        Fit the model (from scratch)
        :param train_data: (List-like) List of strings to train on
        :param y: (vector) Targets
        :param weights: (vector) Training weights. Optional
        :param custom_vocabulary: (List of Strings) Custom vocabulary. Not recommended
        """

        if weights is not None:
            try:
                y = np.hstack(y, weights)
            except:
                print('Weights not accepted')

        if 1 < self.bootstrap < len(y):
            train_data, y = resample(train_data, y, n_samples=self.bootstrap, stratify=y, replace=False)
        elif self.bootstrap < 1:
            n_samples = int(self.bootstrap * len(y))
            train_data, y = resample(train_data, y, n_samples=n_samples, stratify=y, replace=False)

        filtered_data = tokenizer_filter(train_data, remove_punctuation=self.remove_punctuation,
                                         remove_stopwords=self.remove_stopwords, lemmatize=self.lemmatize)

        self.vectorizer = TfidfVectorizer(analyzer=str.split, max_features=self.vocab_size)
        cleaned_data = [' '.join(tweet) for tweet in filtered_data]
        X = self.vectorizer.fit_transform(cleaned_data)

        trainX, testX, trainY, testY = train_test_split(X, y, test_size=self.validation_split, stratify=y)

        print('Fitting BoW model')
        self.classifier = LogisticRegression(max_iter=self.max_iter).fit(trainX, trainY)
        self.accuracy = accuracy_score(testY, self.classifier.predict(testX))

    def refine(self, train_data, y, bootstrap=True, weights=None, max_iter=500, preprocess=True):
        """
        Train the models further on new data. Note that it is not possible to increase the vocabulary
        :param train_data: (List-like of Strings) List of strings to train on
        :param y: (vector) Targets
        :param max_iter: (int) Maximum number of fit iterations. Default: 500
        """

        if weights is not None:
            try:
                y = np.hstack(y, weights)
            except:
                print('Weights not accepted')

        if bootstrap and 1 < self.bootstrap < len(y):
            train_data, y = resample(train_data, y, n_samples=self.bootstrap, stratify=y, replace=False)
        elif bootstrap and self.bootstrap < 1:
            n_samples = int(self.bootstrap * len(y))
            train_data, y = resample(train_data, y, n_samples=n_samples, stratify=y, replace=False)
        if preprocess:
            filtered_data = tokenizer_filter(train_data, remove_punctuation=self.remove_punctuation,
                                             remove_stopwords=self.remove_stopwords, lemmatize=self.lemmatize)
            print('\n Filtered data')
        else:
            filtered_data = train_data

        cleaned_data = [' '.join(tweet) for tweet in filtered_data]
        X = self.vectorizer.transform(cleaned_data)
        self.classifier = LogisticRegression(random_state=0, max_iter=max_iter).fit(X, y)

        self.classifier.fit(X, y)

    def predict(self, data, **kwargs):
        """
        Predict the binary sentiment of a list of tweets
        :param data: (list of Strings) Input tweets
        :param kwargs: Keywords for predict_proba
        :return: (list of bool) Predictions
        """
        return np.round(self.predict_proba(data, **kwargs))

    def predict_proba(self, data):
        """
        Makes predictions
        :param data: (List-like) List of strings to predict sentiment
        :return: (vector) Un-binarized Predictions
        """
        if self.classifier is None:
            raise ValueError('Model has not been trained!')

        filtered_data = tokenizer_filter(data, remove_punctuation=self.remove_punctuation,
                                         remove_stopwords=self.remove_stopwords, lemmatize=self.lemmatize,
                                         verbose=False)

        cleaned_data = [' '.join(tweet) for tweet in filtered_data]
        X = self.vectorizer.transform(cleaned_data)
        return self.classifier.predict(X)

    def export(self, filename):
        """
        Saves the model to disk
        :param filename: (String) Path to file
        """
        parameters = {'Classifier': self.type,
                      'package': self.package,
                      'vocab_size': int(self.vocab_size),
                      'max_iter': int(self.max_iter),
                      'validation_split': float(self.validation_split),
                      'accuracy': float(self.accuracy),
                      'remove_punctuation': self.remove_punctuation,
                      'remove_stopwords': self.remove_stopwords,
                      'lemmatize': self.lemmatize,
                      'bootstrap': self.bootstrap
                      }

        if parameters['bootstrap'] < 1:
            parameters['bootstrap'] = float(parameters['bootstrap'])
        else:
            parameters['bootstrap'] = int(parameters['bootstrap'])

        os.makedirs(filename, exist_ok=True)
        with open(filename + '/param.json', 'w+') as outfile:
            json.dump(parameters, outfile)
        with open(filename + '/bow_vectorizer.pkl', 'wb+') as outfile:
            pkl.dump(self.vectorizer, outfile)
        with open(filename + '/bow_classifier.pkl', 'wb+') as outfile:
            pkl.dump(self.classifier, outfile)

    def load_model(self, filename):
        """
        # TODO revise to properly close pkl files
        :param filename: (String) Path to file
        """

        self.vectorizer = pkl.load(open(filename + '/bow_vectorizer.pkl', 'rb'))
        self.classifier = pkl.load(open(filename + '/bow_classifier.pkl', 'rb'))

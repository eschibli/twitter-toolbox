import numpy as np
import pandas as pd
import spacy
import pickle as pkl
import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# TODO consider renaming glove to pre-trained as it in principle should work with any pre-trained index
# TODO find way to limit learning rate when refining BoW model

def tokenizer_filter(text, remove_punctuation=True, remove_stopwords=True, lemmatize=True, lemmatize_pronouns=False):
    """
    :param text: (series) Text to process
    :param remove_punctuation: (bool) Strip all punctuation
    :param remove_stopwords: (bool) Remove all stopwords TODO enable custom stopword lists
    :param lemmatize: (bool) Lemmatize all words
    :param lemmatize_pronouns: (bool) lemmatize pronouns to -PRON-
    :return: (list) tokenized and processed text
    """
    import spacy
    nlp = spacy.load("en_core_web_sm")

    """
    Define filter
    """
    nlp = spacy.load("en_core_web_sm", disable=['textcat', "parser", 'ner', 'entity_linker'])
    docs = list(text)
    filtered_tokens = []
    if remove_stopwords and remove_punctuation:
        def token_filter(token):
            return not (token.is_punct | token.is_space | token.is_stop)
    elif remove_punctuation:
        def token_filter(token):
            return not (token.is_punct | token.is_space)
    elif remove_stopwords:
        def token_filter(token):
            return not (token.is_stop | token.is_space)
    else:
        def token_filter(token):
            return not token.is_space

    """
    Do filtering
    """
    count = 0
    if lemmatize and lemmatize_pronouns:
        for doc in nlp.pipe(docs, n_threads=-1, batch_size=10000):
            tokens = [token.lemma_.lower() for token in doc if token_filter(token)]
            filtered_tokens.append(tokens)
            count = count + 1
            # Starting with a carriage return rather than ending with one keeps the text visible in some IDEs.
            print('\r Preprocessed %d tweets' % count, end=' ')
        return filtered_tokens
    elif lemmatize:
        for doc in nlp.pipe(docs):
            # pronouns lemmatize to -PRON- which is undesirable when using pre-trained embeddings
            tokens = [token.lemma_.lower() if token.lemma_ != '-PRON-'
                      else token.lower_ for token in doc if token_filter(token)]
            count = count + 1
            # Starting with a carriage return rather than ending with one keeps the text visible in some IDEs.
            print('\r Preprocessed %d tweets'% count, end=' ')
            filtered_tokens.append(tokens)
        return filtered_tokens
    else:
        # lemmatizing pronouns to -PRON- is desirable when not using pre-trained embeddings
        for doc in nlp.pipe(docs):
            tokens = [token.lower_ for token in doc if token_filter(token)]
            filtered_tokens.append(tokens)
            count = count + 1
            # Starting with a carriage return rather than ending with one keeps the text visible in some IDEs.
            print('\r Preprocessed %d tweets' % count, end=' ')
        return filtered_tokens


class SentimentAnalyzer:
    def __init__(self, models=[], model_path='Models'):
        """
        Constructor for SentimentAnalyzer module
        :param models: (list) Models to initialize. Should be a list of tuples formatted like (name, type, params)
        type can be 'bow', 'lstm', or 'glove', and params is a dictionary of parameters
        """
        self.models = {}
        self.model_path = model_path

        for name, type, params in models:
            if type == 'bow':
                self.add_bow_model(name, **params)
            elif type == 'lstm':
                self.add_lstm_model(name, **params)
            elif type == 'glove':
                self.add_glove_model(name, **params)
            else:
                print('Model type %s not recognized' % type)


    def set_model_path(self, model_path):
        self.model_path = model_path

    def add_bow_model(self, name, **kwargs):
        """
        Add another BoW model to the classifier
        :param name: (String) A name to refer to the model
        :param kwargs: Model keywords
        :return:
        """
        self.models[name] = self.BoW_Model(**kwargs)

    def add_lstm_model(self, name, **kwargs):
        """
        Add another LSTM model to the classifier
        :param name: (String) A name to refer to the model
        :param kwargs: Model keywords
        :return:
        """
        self.models[name] = self.LSTM_Model(**kwargs)

    def add_glove_model(self, name, glove_index, **kwargs):
        """
        Add another lstm model with pre-trained embeddings to the classifier
        :param name: (String) A name to refer to the model
        :param glove_index: (dict) The embedding dictionary
        :param kwargs: Model keywords
        :return:
        """
        self.models[name] = self.GloVE_Model(glove_index, **kwargs)

    def delete_models(self, models):
        for model in models:
            self.delete_model(model)

    def delete_model(self, model):
        del self.models[model]

    def trim_models(self, testX, testY, threshold=0.7, metric=accuracy_score, models=[]):
        if len(models) == 0:
            models = self.models.copy().keys()
        for name in models:
            score = metric(testY, self.models[name].predict(testX))
            print('Model %s score: %0.3f' % (name, score))
            if score < threshold:
                print('Deleting model %s' % name)
                del self.models[name]

    def fit(self, X, y, models=[], weights=None):
        """
        Fits the enabled models onto X. Note that this rebuilds the models, as it is not currently possible to update
        the tokenizers
        :param X: (array) Feature matrix
        :param y: (vector) Targets
        :param models: (list of Strings) Names of models to fit. Default: all. Note that default behavior will likely
        cause an error if additional models have been added after fitting a pre-trained embedding model
        :return:
        """
        if len(models) == 0:
            models = self.models.keys()

        for name in models:
            try:
                print('Fitting %s' % name)
                self.models[name].fit(X, y, weights=weights)
            except KeyError:
                print('Model %s not found!' % name)

    def refine(self, X, y, bootstrap=True, weights=None, **kwargs):
        """
        Refits the trained models onto additional data. Not that this does NOT retrain the tokenizers, so it will not
        retrain the vocabulary
        :param X: (array) Feature matrix
        :param y: (vector) Targets
        :param boostrap: (bool) Bootstrap sample the refining data. Default True.
        """

        for model in self.models.values():
            model.refine(X, y, bootstrap=bootstrap, **kwargs)

    def predict(self, X):
        """
        Predicts the sentiment of some
        :param X:
        :return:
        """
        predictions = self.predict_proba(X)

        return np.round(predictions)

    def predict_proba(self, X):
        predictions = []

        for name, model in self.models.items():
            try:
                predictions.append(model.predict_proba(X).reshape(-1))
            except ValueError:
                print('Error using model %s - probably has not been trained' % name)
            except:
                print('Error using model %s' % name)
        return np.mean(predictions, axis=0)

    def save_models(self, names=None):
        """
        Write models to disk for re-use
        """
        if names is None:
            for name, model in self.models.items():
                model.export(self.model_path + '/' + name)
                print('Model %s saved' % name)
        else:
            for name in names:
                try:
                    self.models[name].export(self.model_path + '/' + name)
                    print('Model %s saved' % name)
                except KeyError:
                    print('Model %s not found!' % name)

    def load_models(self, filenames=None, path=None):
        """
        Reloads a saved model from the disk
        :param filenames: (list) List of model names to import
        :param path: (String) Directory to load from
        """
        if path is None:
            path = self.model_path

        if filenames is not None:
            for filename in filenames:
                self.load_model(path + '/' + filename)

        if filenames is None:
            filenames = [folder.path for folder in os.scandir(path) if folder.is_dir()]
            for filename in filenames:
                print(filename)
                self.load_model(filename)

    def load_model(self, filename):
        """
        # TODO this is ugly, consider reworking
        :param filename:
        :return:
        """
        if os.path.exists(filename + '/bow_param.json'):
            try:
                print('Loading BoW model %s' % filename)
                with open(filename + '/bow_param.json', 'r') as infile:
                    bow_param = json.load(infile)
                self.models[filename] = self.BoW_Model(**bow_param)
                self.models[filename].load_model(filename)
                print('BoW model %s loaded successfully' % filename)
            except FileNotFoundError:
                print('Model %s not found' % filename)
            except (IOError, EOFError):
                print('Problem reading %s files' % filename)

        elif os.path.exists(filename + '/lstm_param.json'):
            try:
                print('Loading LSTM model %s' % filename)
                with open(filename + '/lstm_param.json', 'r') as infile:
                    lstm_param = json.load(infile)
                self.models[filename] = self.LSTM_Model(**lstm_param)
                self.models[filename].load_model(filename)
                print('LSTM model %s loaded successfully' % filename)
                self.models[filename].classifier.compile(loss='binary_crossentropy',
                                                         optimizer=self.models[filename].optimizer, metrics=['acc'])
            except FileNotFoundError:
                print('Model %s not found' % filename)
            except (IOError, EOFError):
                print('Problem reading %s files' % filename)

        elif os.path.exists(filename + '/glove_param.json'):
            try:
                print('Loading GloVE model %s' % filename)
                glove_param = json.load(open(filename + '/glove_param.json', 'r'))
                self.models[filename] = self.GloVE_Model(None, **glove_param)
                self.models[filename].load_model(filename)
                self.models[filename].classifier.compile(loss='binary_crossentropy',
                                                         optimizer=self.models[filename].optimizer,
                                                         metrics=['acc'])
                print('Pre-trained embedding model %s loaded successfully' % filename)
            except FileNotFoundError:
                print('Model %s not found' % filename)
            except (IOError, EOFError):
                print('Problem reading %s files' % filename)
        else:
            print('Model %s not found' % filename)

    def evaluate(self, X, y, metric=accuracy_score):
        """
        Evaluate each model
        :param X: (array) Feature matrix
        :param y: (vector) targets
        :param metric: (method) Metric to use
        """

        scores = {'ensembled': metric(y, self.predict(X))}
        for name, model in self.models.items():
            try:
                scores[name] = metric(y, model.predict(X))
                print('Model %s score: %0.3f' % (name, scores[name]))
            except ValueError:
                print('Error, %s probably has not been trained' % name)
        return scores

    class BoW_Model:

        def __init__(self, vocab_size=100000, max_iter=10000, validation_split=0.2, accuracy=0, bootstrap=1, **kwargs):
            """
            Constructor for BoW_Model
            Be sure to add additional parameters to export()
            :param vocab_size: (int) Maximum vocabulary size. Default 1E6
            :param max_iter: (int) Maximum number of fit iterations
            """
            # TODO test effect of vocab_size

            self.vectorizer = None
            self.classifier = None
            self.vocab_size = vocab_size
            self.max_iter = max_iter
            self.type = 'bow'
            self.validation_split = validation_split
            self.accuracy = accuracy
            self.bootstrap = bootstrap

        def fit(self, train_data, y, weights=None):
            """
            Fit the model (from scratch)
            :param train_data: (List-like) List of strings to train on
            :param y: (vector) Targets
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

            filtered_data = tokenizer_filter(train_data, remove_punctuation=True, remove_stopwords=True, lemmatize=True)

            self.vectorizer = TfidfVectorizer(analyzer=str.split, max_features=self.vocab_size)
            cleaned_data = [' '.join(tweet) for tweet in filtered_data]
            X = self.vectorizer.fit_transform(cleaned_data)

            trainX, testX, trainY, testY = train_test_split(X, y, test_size=self.validation_split, stratify=y)

            print('Fitting BoW model')
            self.classifier = LogisticRegression(random_state=0, max_iter=self.max_iter).fit(trainX, trainY)
            self.accuracy = accuracy_score(testY, self.classifier.predict(testX))

        def refine(self, train_data, y, bootstrap=True, weights=None, max_iter=500):
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

            filtered_data = tokenizer_filter(train_data, remove_punctuation=True, remove_stopwords=True, lemmatize=True)
            print('\n Filtered data')

            cleaned_data = [' '.join(tweet) for tweet in filtered_data]
            X = self.vectorizer.transform(cleaned_data)
            self.classifier = LogisticRegression(random_state=0, max_iter=max_iter).fit(X, y)

            self.classifier.fit(X, y)

        def predict(self, data):
            return np.round(self.predict_proba(data))

        def predict_proba(self, data):
            """
            Makes predictions
            :param data: (List-like) List of strings to predict sentiment
            :return: (vector) Un-binarized Predictions
            """
            if self.classifier is None:
                raise ValueError('Model has not been trained!')
            filtered_data = tokenizer_filter(data, remove_punctuation=True, remove_stopwords=True, lemmatize=True)
            cleaned_data = [' '.join(tweet) for tweet in filtered_data]
            X = self.vectorizer.transform(cleaned_data)
            return self.classifier.predict(X)

        def export(self, filename):
            """
            Saves the model to disk
            :param filename: (String) Path to file
            """
            parameters = {'type': self.type,
                          'vocab_size': self.vocab_size,
                          'max_iter': self.max_iter,
                          'validation_split': self.validation_split,
                          'accuracy': self.accuracy}

            os.makedirs(filename, exist_ok=True)
            with open(filename + '/bow_param.json', 'w+') as outfile:
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

    class GloVE_Model:

        def __init__(self, embedding_dict, max_length=25, vocab_size=1000000, batch_size=10000, neurons=100,
                     dropout=0.2, bootstrap=1, early_stopping=True, validation_split=0.2, patience=50, max_iter=250,
                     rec_dropout=0.2, activ='hard_sigmoid', optimizer='adam', accuracy=0, **kwargs):
            """
            Constructor for LSTM classifier using pre-trained embeddings
            Be sure to add extra parameters to export()
            :param embedding_dict: (dict) Embedding dictionary
            :param max_length: (int) Maximum text length, ie, number of temporal nodes. Default 25
            :param vocab_size: (int) Maximum vocabulary size. Default 1E7
            :param max_iter: (int) Number of training epochs. Default 100
            :param neurons: (int) Depth (NOT LENGTH) of LSTM network. Default 100
            :param dropout: (float) Dropout
            :param activ: (String) Activation function (for visible layer). Default 'hard_sigmoid'
            :param optimizer: (String) Optimizer. Default 'adam'
            :param early_stopping: (bool) Train with early stopping
            :param validation_split: (float) Fraction of training data to withold for validation
            :param patience: (int) Number of epochs to wait before early stopping
            """
            self.bootstrap = bootstrap
            self.early_stopping = early_stopping
            self.validation_split = validation_split
            self.patience = patience
            self.max_iter = max_iter

            self.max_length = max_length
            self.embedding_dict = embedding_dict
            self.max_iter = max_iter
            self.vocab_size = vocab_size
            self.neurons = neurons
            self.dropout = dropout
            self.rec_dropout = rec_dropout
            self.activ = activ
            self.optimizer = optimizer
            self.batch_size = batch_size

            self.type = 'glove'
            self.embed_vec_len = None
            self.tokenizer = None
            self.classifier = None
            self.word_index = None
            self.embedding_matrix = None
            self.accuracy = accuracy

            if self.embedding_dict is not None:
                self.embed_vec_len = len(list(self.embedding_dict.values())[0])

        def fit(self, train_data, y, clear_embedding_dict=True, weights=None, **kwargs):
            """
            :param train_data: (Dataframe) Training data
            :param y: (vector) Targets
            :param clear_embedding_dict: (Bool) Clear the embedding dictionary after building the neural network to save
            memory. Recommended, but note that the model will need to be build from scratch to refit (not refine)
            :return:
            """

            """
            # Preprocess and tokenize text
            """

            if 1 < self.bootstrap < len(y):
                train_data, y = resample(train_data, y, n_samples=self.bootstrap, stratify=y, replace=False)
            elif self.bootstrap < 1:
                n_samples = int(self.bootstrap * len(y))
                train_data, y = resample(train_data, y, n_samples=n_samples, stratify=y, replace=False)
            print('Sampled %d training points' % len(y))
            filtered_data = tokenizer_filter(train_data, remove_punctuation=False, remove_stopwords=False,
                                             lemmatize=True)
            print('Filtered data')
            cleaned_data = [' '.join(tweet) for tweet in filtered_data]

            self.tokenizer = Tokenizer(num_words=self.vocab_size)
            self.tokenizer.fit_on_texts(cleaned_data)
            train_sequences = self.tokenizer.texts_to_sequences(cleaned_data)

            self.word_index = self.tokenizer.word_index
            print('Tokenized data')
            print('Found %s unique tokens.' % len(self.word_index))

            X = pad_sequences(train_sequences, maxlen=self.max_length, padding='pre')

            self.embedding_matrix = np.zeros((len(self.word_index) + 1, self.embed_vec_len))
            for word, i in self.word_index.items():
                embedding_vector = self.embedding_dict.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    self.embedding_matrix[i] = embedding_vector

            neurons = self.neurons  # Depth (NOT LENGTH) of LSTM network
            dropout = self.dropout  # Dropout - around 0.25 is probably best
            rec_dropout = self.rec_dropout
            activ = self.activ
            costfunction = 'binary_crossentropy'

            """
            Create LSTM model
            """

            print("Creating LSTM model")
            init = keras.initializers.glorot_uniform(seed=1)
            optimizer = self.optimizer

            # TODO input_dim is kludged, MUST FIX - should be able to trim embedding matrix in embed_glove.py

            self.classifier = keras.models.Sequential()

            self.classifier.add(keras.layers.embeddings.Embedding(input_dim=len(self.word_index) + 1,
                                                                  output_dim=self.embed_vec_len,
                                                                  input_length=self.max_length,
                                                                  mask_zero=True, trainable=False,
                                                                  embeddings_initializer=keras.initializers.Constant(
                                                                      self.embedding_matrix)))
            self.classifier.add(keras.layers.SpatialDropout1D(dropout))
            self.classifier.add(keras.layers.LSTM(units=neurons, input_shape=(self.max_length, self.embed_vec_len),
                                                  kernel_initializer=init, dropout=dropout,
                                                  recurrent_dropout=rec_dropout))
            self.classifier.add(keras.layers.Dense(units=1, kernel_initializer=init, activation=activ))
            self.classifier.compile(loss=costfunction, optimizer=optimizer, metrics=['acc'])
            print(self.classifier.summary())

            if clear_embedding_dict:
                self.embedding_matrix = None
                self.embedding_dict = None

            es = []
            if self.early_stopping:
                es.append(
                    keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=self.patience))
            print('Fitting GloVE model')
            history = self.classifier.fit(X, y, validation_split=self.validation_split, batch_size=self.batch_size,
                                          epochs=self.max_iter, sample_weight=weights,
                                          callbacks=es, verbose=1)

            self.accuracy = np.max(history.history['val_acc'])
            return history

        def refine(self, train_data, y, bootstrap=True, weights=None):
            """
            Train model further
            :param train_data:
            :param y:
            :param bootstrap: (bool) Bootstrap sample the refining data. Default True
            :return:
            """

            """
            # Preprocess and tokenize text
            """
            if bootstrap and 1 < self.bootstrap < len(y):
                train_data, y = resample(train_data, y, n_samples=self.bootstrap, stratify=y, replace=False)
            elif bootstrap and self.bootstrap < 1:
                n_samples = int(self.bootstrap * len(y))
                train_data, y = resample(train_data, y, n_samples=n_samples, stratify=y, replace=False)

            filtered_data = tokenizer_filter(train_data, remove_punctuation=False, remove_stopwords=False,
                                             lemmatize=True)

            print('Filtered data')

            cleaned_data = [' '.join(tweet) for tweet in filtered_data]
            train_sequences = self.tokenizer.texts_to_sequences(cleaned_data)

            X = pad_sequences(train_sequences, maxlen=self.max_length, padding='pre')

            es = []
            if self.early_stopping:
                es.append(
                    keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=self.patience))

            history = self.classifier.fit(X, y, validation_split=self.validation_split, callbacks=es,
                                          batch_size=self.batch_size, sample_weight=weights,
                                          epochs=self.max_iter, verbose=1)
            self.accuracy = np.max(history.history['val_acc'])
            return history

        def predict(self, data):
            return np.round(self.predict_proba(data))

        def predict_proba(self, data):
            from keras.preprocessing.sequence import pad_sequences
            if self.tokenizer is None:
                raise ValueError('Model has not been trained!')
            filtered_data = tokenizer_filter(data, remove_punctuation=True, remove_stopwords=True, lemmatize=True)
            cleaned_data = [' '.join(tweet) for tweet in filtered_data]
            X = pad_sequences(self.tokenizer.texts_to_sequences(cleaned_data), maxlen=self.max_length)
            return self.classifier.predict(X)

        def export(self, filename):
            """
            Saves the model to disk
            :param filename: (String) Path to file
            """

            parameters = {'type': self.type,
                          'max_length': self.max_length,
                          'neurons': self.neurons,
                          'dropout': self.dropout,
                          'rec_dropout': self.rec_dropout,
                          'activ': self.activ,
                          'optimizer': self.optimizer,
                          'vocab_size': self.vocab_size,
                          'max_iter': self.max_iter,
                          'batch_size': self.batch_size,
                          'early_stopping': self.early_stopping,
                          'patience': self.patience,
                          'bootstrop': self.bootstrap,
                          'validation_split': self.validation_split,
                          'accuracy': self.accuracy}

            os.makedirs(filename, exist_ok=True)
            with open(filename + '/glove_param.json', 'w+') as outfile:
                json.dump(parameters, outfile)
            with open(filename + '/glove_tokenizer.pkl', 'wb+') as outfile:
                pkl.dump(self.tokenizer, outfile)
            model_json = self.classifier.to_json()
            with open(filename + "/glove_model.json", "w+") as json_file:
                json_file.write(model_json)
            self.classifier.save_weights(filename + "/glove_model.h5")

        def load_model(self, filename):
            """
            :param filename: (String) Path to file
            """
            self.tokenizer = pkl.load(open(filename + '/glove_tokenizer.pkl', 'rb'))
            with open(filename + '/glove_model.json', 'r') as infile:
                model_json = infile.read()
            self.classifier = keras.models.model_from_json(model_json)
            self.classifier.load_weights(filename + '/glove_model.h5')

    class LSTM_Model:

        def __init__(self, max_length=25, vocab_size=1000000, neurons=50,
                     dropout=0.25, rec_dropout=0.25, embed_vec_len=200, activ='hard_sigmoid', optimizer='adam',
                     bootstrap=1, early_stopping=True, patience=50, validation_split=0.2, max_iter=250,
                     batch_size=10000,
                     accuracy=0, **kwargs):
            """
            Constructor for LSTM classifier using pre-trained embeddings
            Be sure to add additional parametesr to export()
            :param max_length: (int) Maximum text length, ie, number of temporal nodes. Default 25
            :param vocab_size: (int) Maximum vocabulary size. Default 1E7
            :param max_iter: (int) Number of training epochs. Default 100
            :param neurons: (int) Depth (NOT LENGTH) of LSTM network. Default 100
            :param dropout: (float) Dropout
            :param activ: (String) Activation function (for visible layer). Default 'hard_sigmoid'
            :param optimizer: (String) Optimizer. Default 'adam'
            """

            self.bootstrap = bootstrap
            self.early_stopping = early_stopping
            self.validation_split = validation_split
            self.patience = patience
            self.max_iter = max_iter

            self.max_length = max_length
            self.max_iter = max_iter
            self.batch_size = batch_size
            self.vocab_size = vocab_size
            self.neurons = neurons
            self.dropout = dropout
            self.rec_dropout = rec_dropout
            self.activ = activ
            self.optimizer = optimizer
            self.embed_vec_len = embed_vec_len

            self.type = 'lstm'
            self.tokenizer = None
            self.classifier = None
            self.word_index = None
            self.embedding_matrix = None
            self.accuracy = accuracy

        def fit(self, train_data, y, weights=None, **kwargs):
            """
            :param train_data:
            :param y:
            :param early_stopping: (bool) train with early stopping
            :param validation_split: (float) Fraction of training data to withold for validation
            :param patience: (int) Number of epochs to wait before early stopping
            :return:
            """

            """
            # Preprocess and tokenize text
            """

            if 1 < self.bootstrap < len(y):
                train_data, y = resample(train_data, y, n_samples=self.bootstrap, stratify=y, replace=False)
            elif self.bootstrap < 1:
                n_samples = int(self.bootstrap * len(y))
                train_data, y = resample(train_data, y, n_samples=n_samples, stratify=y, replace=False)

            filtered_data = tokenizer_filter(train_data, remove_punctuation=False, remove_stopwords=False,
                                             lemmatize=True)

            print('Filtered data')
            cleaned_data = [' '.join(tweet) for tweet in filtered_data]
            self.tokenizer = Tokenizer(num_words=self.vocab_size, filters='"#$%&()*+-/:;<=>?@[\\]^_`{|}~\t\n')

            self.tokenizer.fit_on_texts(cleaned_data)
            train_sequences = self.tokenizer.texts_to_sequences(cleaned_data)

            self.word_index = self.tokenizer.word_index
            print('Found %s unique tokens.' % len(self.word_index))

            X = pad_sequences(train_sequences, maxlen=self.max_length, padding='pre')

            neurons = self.neurons  # Depth (NOT LENGTH) of LSTM network
            dropout = self.dropout  # Dropout - around 0.25 is probably best
            rec_dropout = self.rec_dropout
            activ = self.activ
            costfunction = 'binary_crossentropy'

            """
            Create LSTM model
            """

            print("Creating LSTM model")
            init = keras.initializers.glorot_uniform(seed=1)
            optimizer = self.optimizer

            self.classifier = keras.models.Sequential()

            self.classifier.add(keras.layers.embeddings.Embedding(input_dim=len(self.word_index) + 1,
                                                                  output_dim=self.embed_vec_len,
                                                                  input_length=self.max_length,
                                                                  mask_zero=True,
                                                                  embeddings_initializer=keras.initializers.glorot_normal(
                                                                      seed=None)))
            self.classifier.add(keras.layers.SpatialDropout1D(dropout))
            self.classifier.add(keras.layers.LSTM(units=neurons, input_shape=(self.max_length, self.embed_vec_len),
                                                  kernel_initializer=init, dropout=dropout,
                                                  recurrent_dropout=rec_dropout))
            self.classifier.add(keras.layers.Dense(units=1, kernel_initializer=init, activation=activ))
            self.classifier.compile(loss=costfunction, optimizer=optimizer, metrics=['acc'])
            print(self.classifier.summary())
            es = []
            if self.early_stopping:
                es.append(
                    keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=self.patience))

            print('Fitting LSTM model')
            history = self.classifier.fit(X, y, validation_split=self.validation_split, callbacks=es,
                                          batch_size=self.batch_size, sample_weight=weights,
                                          epochs=self.max_iter, verbose=1)
            self.accuracy = np.max(history.history['val_acc'])
            return history

        def refine(self, train_data, y, bootstrap=True, weights=None):
            """
            Train model further
            :param train_data:1
            :param y:
            :param max_iter:
            :param vocab_size:
            :return:
            """

            """
            # Preprocess and tokenize text
            """

            if bootstrap and 1 < self.bootstrap < len(y):
                train_data, y = resample(train_data, y, n_samples=self.bootstrap, stratify=y, replace=False)
            elif bootstrap and self.bootstrap < 1:
                n_samples = int(self.bootstrap * len(y))
                train_data, y = resample(train_data, y, n_samples=n_samples, stratify=y, replace=False)

            filtered_data = tokenizer_filter(train_data, remove_punctuation=False, remove_stopwords=False,
                                             lemmatize=True)
            cleaned_data = [' '.join(tweet) for tweet in filtered_data]
            train_sequences = self.tokenizer.texts_to_sequences(cleaned_data)

            X = pad_sequences(train_sequences, maxlen=self.max_length, padding='pre')

            es = []
            if self.early_stopping:
                es.append(keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=self.patience))

            history = self.classifier.fit(X, y, validation_split=validation_split, callbacks=es,
                                          batch_size=self.batch_size, sample_weight=weights,
                                          epochs=max_iter, verbose=1)
            self.accuracy = np.max(history.history['val_acc'])
            return history

        def predict(self, data):
            return np.round(self.predict_proba(data))

        def predict_proba(self, data):
            from keras.preprocessing.sequence import pad_sequences
            if self.tokenizer is None:
                raise ValueError('Model has not been trained!')
            filtered_data = tokenizer_filter(data, remove_punctuation=True, remove_stopwords=True, lemmatize=True)
            cleaned_data = [' '.join(tweet) for tweet in filtered_data]
            X = pad_sequences(self.tokenizer.texts_to_sequences(cleaned_data), maxlen=self.max_length)
            return self.classifier.predict(X)

        def export(self, filename):
            """
            Saves the model to disk
            :param filename: (String) Path to file
            """

            parameters = {'type': self.type,
                          'bootstrap': self.bootstrap,
                          'early_stopping': self.early_stopping,
                          'validation_split': self.validation_split,
                          'patience': self.patience,
                          'max_iter': self.max_iter,
                          'max_length': self.max_length,
                          'neurons': self.neurons,
                          'dropout': self.dropout,
                          'rec_dropout': self.rec_dropout,
                          'activ': self.activ,
                          'optimizer': self.optimizer,
                          'vocab_size': self.vocab_size,
                          'batch_size': self.batch_size,
                          'accuracy': self.accuracy}
            os.makedirs(filename, exist_ok=True)
            with open(filename + '/lstm_param.json', 'w+') as outfile:
                json.dump(parameters, outfile)

            with open(filename + '/lstm_tokenizer.pkl', 'wb+') as outfile:
                pkl.dump(self.tokenizer, outfile)
            model_json = self.classifier.to_json()
            with open(filename + "/lstm_model.json", "w+") as json_file:
                json_file.write(model_json)
            self.classifier.save_weights(filename + "/lstm_model.h5")

        def load_model(self, filename):
            """
            :param filename: (String) Path to file
            """
            self.tokenizer = pkl.load(open(filename + '/lstm_tokenizer.pkl', 'rb'))
            with open(filename + '/lstm_model.json', 'r') as infile:
                model_json = infile.read()
            self.classifier = keras.models.model_from_json(model_json)
            self.classifier.load_weights(filename + '/lstm_model.h5')

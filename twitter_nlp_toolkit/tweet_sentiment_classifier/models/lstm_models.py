from zipfile import ZipFile

from twitter_nlp_toolkit.file_fetcher import file_fetcher
from ..tweet_sentiment_classifier import Classifier, tokenizer_filter

import os
import json
import pickle as pkl
import numpy as np

import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.utils import resample


class LSTM_Model(Classifier):
    """
     LSTM model with trainable embedding layer"
    """

    def __init__(self, max_length=25, vocab_size=1000000, neurons=50,
                 dropout=0.25, rec_dropout=0.25, embed_vec_len=200, activ='hard_sigmoid',
                 learning_rate=0.001, bootstrap=1, early_stopping=True, patience=50, validation_split=0.2, max_iter=250,
                 batch_size=10000, accuracy=0, remove_punctuation=False, remove_stopwords=False, lemmatize=True,
                 hidden_neurons=0, **kwargs):
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
        self.type = 'LSTM_Model'
        self.package = 'twitter_nlp_toolkit.tweet_sentiment_classifier.models.lstm_models'
        self.bootstrap = bootstrap
        self.early_stopping = early_stopping
        self.validation_split = validation_split
        self.patience = patience
        self.max_iter = max_iter
        self.learning_rate = learning_rate

        self.max_length = max_length
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.neurons = neurons
        self.hidden_neurons = hidden_neurons
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.activ = activ
        self.optimizer = 'adam'
        self.embed_vec_len = embed_vec_len

        self.embedding_initializer = tf.keras.initializers.glorot_normal(seed=None)
        self.finetune_embeddings = True

        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize

        self.tokenizer = None
        self.classifier = None
        self.word_index = None
        self.embedding_matrix = None
        self.es = []
        self.accuracy = accuracy

    def preprocess(self, train_data, y, weights=None):
        if weights is None:
            weights = np.ones(len(y))

        """
        # Preprocess and tokenize text
        """

        if 1 < self.bootstrap < len(y):
            train_data, y, weights = resample(train_data, y, weights, n_samples=self.bootstrap, stratify=y,
                                              replace=False)
        elif self.bootstrap < 1:
            n_samples = int(self.bootstrap * len(y))
            train_data, y, weights = resample(train_data, y, weights, n_samples=n_samples, stratify=y,
                                              replace=False)

        filtered_data = tokenizer_filter(train_data, remove_punctuation=self.remove_punctuation,
                                         remove_stopwords=self.remove_punctuation, lemmatize_pronouns=True,
                                         lemmatize=self.lemmatize, verbose=True)
        print('Filtered data')

        cleaned_data = [' '.join(tweet) for tweet in filtered_data]
        return cleaned_data, y, weights

    def fit(self, train_data, y, weights=None, custom_vocabulary=None):
        """
        :param train_data: (List-like of Strings) Tweets to fit on
        :param y: (Vector) Targets
        :param weights: (Vector) Weights for fitting data
        :param custom_vocabulary: (List of String) Custom vocabulary to use for tokenizer. Not recommended.
        :return: Fit history

        """

        cleaned_data, y, weights = self.preprocess(train_data, y, weights)

        self.tokenizer = Tokenizer(num_words=self.vocab_size, filters='"#$%&()*+-/:;<=>?@[\\]^_`{|}~\t\n')
        self.tokenizer.fit_on_texts(cleaned_data)

        train_sequences = self.tokenizer.texts_to_sequences(cleaned_data)

        self.word_index = self.tokenizer.word_index
        print('Found %s unique tokens.' % len(self.word_index))

        X = pad_sequences(train_sequences, maxlen=self.max_length, padding='pre')

        self.build_LSTM_network()

        print('Fitting LSTM model')

        history = self.classifier.fit(X, y, validation_split=self.validation_split, callbacks=self.es,
                                      batch_size=self.batch_size, sample_weight=weights,
                                      epochs=self.max_iter, verbose=2)

        self.accuracy = np.max(history.history['val_acc'])
        return history

    def build_LSTM_network(self):
        print("Creating LSTM model")
        if self.optimizer == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        init = tf.keras.initializers.glorot_uniform(seed=1)

        self.classifier = tf.keras.models.Sequential()

        self.classifier.add(tf.keras.layers.Embedding(input_dim=len(self.word_index) + 1,
                                                      output_dim=self.embed_vec_len,
                                                      input_length=self.max_length,
                                                      mask_zero=True,
                                                      embeddings_initializer=self.embedding_initializer,
                                                      trainable=self.finetune_embeddings))

        self.classifier.add(tf.keras.layers.LSTM(units=self.neurons, input_shape=(self.max_length, self.embed_vec_len),
                                                 kernel_initializer=init, dropout=self.dropout,
                                                 recurrent_dropout=self.rec_dropout))

        if self.hidden_neurons > 0:
            self.classifier.add(
                tf.keras.layers.Dense(units=self.hidden_neurons, kernel_initializer=init, activation='elu'))
        self.classifier.add(tf.keras.layers.Dense(units=1, kernel_initializer=init, activation=self.activ))
        self.classifier.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['acc'])
        print(self.classifier.summary())
        if self.early_stopping:
            self.es.append(
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=self.patience))

    def refine(self, train_data, y, bootstrap=True, weights=None):
        """
        Train model further

        :param train_data: (list of Strings) Training tweets
        :param y: (vector) Targets
        :param weights: (vector) Training data weights
        :param bootstrap: (bool) Resample training data
        :returns: Fit history
        """

        """
        # Preprocess and tokenize text
        """

        if bootstrap and 1 < self.bootstrap < len(y):
            train_data, y, weights = resample(train_data, y, weights, n_samples=self.bootstrap, stratify=y,
                                              replace=False)
        elif bootstrap and self.bootstrap < 1:
            n_samples = int(self.bootstrap * len(y))
            train_data, y, weights = resample(train_data, y, weights, n_samples=n_samples, stratify=y,
                                              replace=False)

        filtered_data = tokenizer_filter(train_data, remove_punctuation=False, remove_stopwords=False,
                                         lemmatize=True)

        cleaned_data = [' '.join(tweet) for tweet in filtered_data]
        train_sequences = self.tokenizer.texts_to_sequences(cleaned_data)

        X = pad_sequences(train_sequences, maxlen=self.max_length, padding='pre')

        es = []
        if self.early_stopping:
            es.append(
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=self.patience))

        history = self.classifier.fit(X, y, validation_split=self.validation_split, callbacks=self.es,
                                      batch_size=self.batch_size, sample_weight=weights,
                                      epochs=self.max_iter, verbose=2)
        self.accuracy = np.max(history.history['val_acc'])
        return history

    def predict(self, data, **kwargs):
        """
        Make binary predictions
        :param data: (list of Strings) Tweets
        :return: (vector of Bool) Predictions
        """
        return np.round(self.predict_proba(data, **kwargs))

    def predict_proba(self, data, preprocess=True):
        """
        Make continuous predictions
        :param data:  (list of Strings) Tweets
        :return: (vector) Predictions
        """
        if self.tokenizer is None:
            raise ValueError('Model has not been trained!')

        filtered_data = tokenizer_filter(data, remove_punctuation=self.remove_punctuation,
                                         remove_stopwords=self.remove_stopwords, lemmatize=self.lemmatize,
                                         verbose=False)

        cleaned_data = [' '.join(tweet) for tweet in filtered_data]
        X = pad_sequences(self.tokenizer.texts_to_sequences(cleaned_data), maxlen=self.max_length)
        return self.classifier.predict(X)

    def export(self, filename):
        """
        Saves the model to disk
        :param filename: (String) Path to file
        """

        parameters = {'Classifier': self.type,
                      'package': self.package,
                      'bootstrap': self.bootstrap,
                      'early_stopping': self.early_stopping,
                      'validation_split': float(self.validation_split),
                      'patience': int(self.patience),
                      'max_iter': int(self.max_iter),
                      'max_length': int(self.max_length),
                      'neurons': int(self.neurons),
                      'hidden_neruons': int(self.hidden_neurons),
                      'dropout': float(self.dropout),
                      'rec_dropout': float(self.rec_dropout),
                      'activ': self.activ,
                      'vocab_size': self.vocab_size,
                      'batch_size': self.batch_size,
                      'accuracy': float(self.accuracy),
                      'remove_punctuation': self.remove_punctuation,
                      'remove_stopwords': self.remove_stopwords,
                      'lemmatize': self.lemmatize,
                      'learning_rate': self.learning_rate
                      }

        if parameters['bootstrap'] < 1:
            parameters['bootstrap'] = float(parameters['bootstrap'])
        else:
            parameters['bootstrap'] = int(parameters['bootstrap'])

        os.makedirs(filename, exist_ok=True)
        with open(filename + '/param.json', 'w+') as outfile:
            json.dump(parameters, outfile)

        with open(filename + '/lstm_tokenizer.pkl', 'wb+') as outfile:
            pkl.dump(self.tokenizer, outfile)
        model_json = self.classifier.to_json()
        with open(filename + "/lstm_model.json", "w+") as json_file:
            json_file.write(model_json)
        self.classifier.save_weights(filename + "/lstm_model.h5")

    def load_model(self, filename):
        """
        Load a model from the disc
        :param filename: (String) Path to file
        """
        self.tokenizer = pkl.load(open(filename + '/lstm_tokenizer.pkl', 'rb'))
        with open(filename + '/lstm_model.json', 'r') as infile:
            model_json = infile.read()
        self.classifier = tf.keras.models.model_from_json(model_json)
        self.classifier.load_weights(filename + '/lstm_model.h5')
        self.classifier.compile(loss='binary_crossentropy',
                                optimizer=self.optimizer,
                                metrics=['acc'])


class GloVE_Model(LSTM_Model):
    """
    LSTM model that uses GloVE pre-trained embeddings
    # TODO add automatic embedding downloading and unzipping
    """

    def __init__(self, embedding_dict=None, embed_vec_len=200, max_length=25, vocab_size=1000000, batch_size=10000,
                 neurons=100,
                 hidden_neurons=0, dropout=0.2, bootstrap=1, early_stopping=True, validation_split=0.2, patience=50,
                 max_iter=250,
                 rec_dropout=0.2, activ='hard_sigmoid', accuracy=0, remove_punctuation=False, learning_rate=0.001,
                 remove_stopwords=False, lemmatize=True, finetune_embeddings=False, **kwargs):
        """
        Constructor for LSTM classifier using pre-trained embeddings
        Be sure to add extra parameters to export()
        :param glove_index: (Dict) Embedding index to use. IF not provided, a standard one will be downloaded
        :param name: (String) Name of model
        :param embed_vec_len: (int) Embedding depth. Inferred from dictionary if provided. Otherwise 25, 50, 100, and
        are acceptible values. 200
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
        self.type = 'GloVE_Model'
        self.package = 'twitter_nlp_toolkit.tweet_sentiment_classifier.models.lstm_models'
        self.bootstrap = bootstrap
        self.early_stopping = early_stopping
        self.validation_split = validation_split
        self.patience = patience
        self.max_iter = max_iter
        self.embed_vec_len = embed_vec_len
        self.learning_rate = learning_rate

        self.embedding_initializer = None

        self.finetune_embeddings = finetune_embeddings
        self.max_length = max_length
        self.embedding_dict = embedding_dict
        self.max_iter = max_iter
        self.vocab_size = vocab_size
        self.neurons = neurons
        self.hidden_neurons = hidden_neurons
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.activ = activ
        self.optimizer = 'adam'
        self.batch_size = batch_size

        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.es = []

        self.tokenizer = None
        self.classifier = None
        self.word_index = None
        self.embedding_matrix = None
        self.accuracy = accuracy

        if self.embedding_dict is not None:
            self.embed_vec_len = len(list(self.embedding_dict.values())[0])
            print('Setting embedding depth to {}'.format(self.embed_vec_len))

    def preprocess(self, train_data, y, weights=None):
        if self.embedding_dict is None:
            print('Reloading embedding index')
            try:
                self.embedding_dict = {}
                with open('.glove_dicts/glove.twitter.27B.' + str(self.embed_vec_len) + 'd.txt', encoding="utf8") as f:
                    for line in f:
                        word, representation = line.split(maxsplit=1)
                        representation = np.fromstring(representation, 'f', sep=' ')
                        self.embedding_dict[word] = representation

                print('Dictionary loaded')

            except FileNotFoundError:
                file_fetcher.download_file("http://nlp.stanford.edu/data/glove.twitter.27B.zip",
                                           "glove_dicts.zip")
                with ZipFile('glove_dicts.zip', 'r') as zipObj:
                    zipObj.extractall('glove_dicts')
                self.embedding_dict = {}
                with open('/.glove_dicts/glove.twitter.27B.' + str(self.embed_vec_len) + 'd.txt', encoding="utf8") as f:
                    for line in f:
                        word, representation = line.split(maxsplit=1)
                        representation = np.fromstring(representation, 'f', sep=' ')
                        self.embedding_dict[word] = representation

                print('Dictionary loaded')

        if weights is None:
            weights = np.ones(len(y))

        if 1 < self.bootstrap < len(y):
            train_data, y, weights = resample(train_data, y, weights, n_samples=self.bootstrap, stratify=y,
                                              replace=False)
        elif self.bootstrap < 1:
            n_samples = int(self.bootstrap * len(y))
            train_data, y, weights = resample(train_data, y, weights, n_samples=n_samples, stratify=y,
                                              replace=False)

        print('Sampled %d training points' % len(train_data))

        filtered_data = tokenizer_filter(train_data, remove_punctuation=self.remove_punctuation,
                                         remove_stopwords=self.remove_stopwords, lemmatize=self.lemmatize,
                                         lemmatize_pronouns=False)
        print('Filtered data')

        cleaned_data = [' '.join(tweet) for tweet in filtered_data]
        return cleaned_data, y, weights

    def fit(self, train_data, y, weights=None, custom_vocabulary=None, clear_embedding_dictionary=True):
        """
        :param train_data: (Dataframe) Training data
        :param y: (vector) Targets
        :param weights: (vector) Weights for fitting data
        :param custom_vocabulary: Custom vocabulary for the tokenizer. Not recommended.
        :param clear_embedding_dictionary: Delete the embedding dictionary after loading the embedding layer.
        Recommended, but will prevent the model from being re-fit (not refined)
        :returns Fit history
        """

        """
        # Preprocess and tokenize text
        """
        cleaned_data, y, weights = self.preprocess(train_data, y, weights)

        if custom_vocabulary is not None:
            print('Applying custom vocabulary')
            self.tokenizer = Tokenizer(num_words=len(custom_vocabulary))
            self.tokenizer.fit_on_texts(custom_vocabulary)
        else:
            print('Fitting tokenizer')
            self.tokenizer = Tokenizer(num_words=self.vocab_size, char_level=False)
            self.tokenizer.fit_on_texts(cleaned_data)

        train_sequences = self.tokenizer.texts_to_sequences(cleaned_data)

        self.word_index = self.tokenizer.word_index

        X = pad_sequences(train_sequences, maxlen=self.max_length, padding='pre')

        self.embedding_matrix = np.zeros((len(self.word_index) + 1, self.embed_vec_len))
        for word, i in self.word_index.items():
            embedding_vector = self.embedding_dict.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros. # TODO consider optimizing
                self.embedding_matrix[i] = embedding_vector

        self.build_LSTM_network()

        if clear_embedding_dictionary:
            self.embedding_matrix = None
            self.embedding_dict = None

        print('Fitting GloVE model')

        history = self.classifier.fit(X, y, validation_split=self.validation_split, batch_size=self.batch_size,
                                      epochs=self.max_iter, sample_weight=weights,
                                      callbacks=self.es, verbose=2)

        self.accuracy = np.max(history.history['val_acc'])
        return history

    def export(self, filename):
        """
        Saves the model to disk
        :param filename: (String) Path to file
        """

        parameters = {'Classifier': self.type,
                      'package': self.package,
                      'max_length': int(self.max_length),
                      'neurons': int(self.neurons),
                      'hidden_neruons': int(self.hidden_neurons),
                      'dropout': float(self.dropout),
                      'rec_dropout': float(self.rec_dropout),
                      'activ': self.activ,
                      'vocab_size': int(self.vocab_size),
                      'max_iter': int(self.max_iter),
                      'batch_size': self.batch_size,
                      'early_stopping': self.early_stopping,
                      'patience': int(self.patience),
                      'bootstrap': self.bootstrap,
                      'validation_split': float(self.validation_split),
                      'accuracy': float(self.accuracy),
                      'remove_punctuation': self.remove_punctuation,
                      'remove_stopwords': self.remove_stopwords,
                      'lemmatize': self.lemmatize,
                      'finetune_embeddings': self.finetune_embeddings,
                      'learning_rate': self.learning_rate
                      }

        if parameters['bootstrap'] < 1:
            parameters['bootstrap'] = float(parameters['bootstrap'])
        else:
            parameters['bootstrap'] = int(parameters['bootstrap'])

        os.makedirs(filename, exist_ok=True)
        with open(filename + '/param.json', 'w+') as outfile:
            json.dump(parameters, outfile)
        with open(filename + '/glove_tokenizer.pkl', 'wb+') as outfile:
            pkl.dump(self.tokenizer, outfile)
        # model_json = self.classifier.to_json()
        with open(filename + "/glove_model.json", "w+") as json_file:
            json_file.write(self.classifier.to_json())
        self.classifier.save_weights(filename + "/glove_model.h5")

    def load_model(self, filename):
        """
        :param filename: (String) Path to file
        """
        self.tokenizer = pkl.load(open(filename + '/glove_tokenizer.pkl', 'rb'))
        with open(filename + '/glove_model.json', 'r') as infile:
            model_json = infile.read()
        self.classifier = tf.keras.models.model_from_json(model_json)
        self.classifier.load_weights(filename + '/glove_model.h5')
        self.classifier.compile(loss='binary_crossentropy',
                                optimizer=self.optimizer,
                                metrics=['acc'])


class NGRAM_Model(LSTM_Model):
    """
    LSTM model that uses GloVE pre-trained embeddings
    # TODO add automatic embedding downloading and unzipping
    """

    def __init__(self, embedding_dict=None, embed_vec_len=200, max_length=25, vocab_size=1000000, batch_size=10000,
                 neurons=100,
                 hidden_neurons=0, dropout=0.2, bootstrap=1, early_stopping=True, validation_split=0.2, patience=50,
                 max_iter=250,
                 rec_dropout=0.2, activ='hard_sigmoid', accuracy=0, remove_punctuation=False, learning_rate=0.001,
                 remove_stopwords=False, lemmatize=True, finetune_embeddings=True, n_gram=3, feature_maps=10, **kwargs):
        """
        Constructor for NGRAM LSTM classifier using pre-trained embeddings
        Be sure to add extra parameters to export()
        :param glove_index: (Dict) Embedding index to use. IF not provided, a standard one will be downloaded
        :param name: (String) Name of model
        :param embed_vec_len: (int) Embedding depth. Inferred from dictionary if provided. Otherwise 25, 50, 100, and
        are acceptible values. 200
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
        self.type = 'NGRAM_Model'
        self.package = 'twitter_nlp_toolkit.tweet_sentiment_classifier.models.lstm_models'
        self.bootstrap = bootstrap
        self.early_stopping = early_stopping
        self.validation_split = validation_split
        self.patience = patience
        self.max_iter = max_iter
        self.embed_vec_len = embed_vec_len
        self.learning_rate = learning_rate

        self.embedding_initializer = None

        self.finetune_embeddings = finetune_embeddings
        self.max_length = max_length
        self.embedding_dict = embedding_dict
        self.max_iter = max_iter
        self.vocab_size = vocab_size
        self.neurons = neurons
        self.hidden_neurons = hidden_neurons
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.activ = activ
        self.optimizer = 'adam'
        self.batch_size = batch_size

        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.es = []

        self.tokenizer = None
        self.classifier = None
        self.word_index = None
        self.embedding_matrix = None
        self.accuracy = accuracy

        self.n_gram = n_gram
        self.feature_maps = feature_maps

        if self.embedding_dict is not None:
            self.embed_vec_len = len(list(self.embedding_dict.values())[0])
            print('Setting embedding depth to {}'.format(self.embed_vec_len))

    def build_NGRAM_network(self):
        print("Creating LSTM model")
        if self.optimizer == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        init = tf.keras.initializers.glorot_uniform(seed=1)

        self.classifier = tf.keras.models.Sequential()

        self.classifier.add(tf.keras.layers.Embedding(input_dim=len(self.word_index) + 1,
                                                      output_dim=self.embed_vec_len,
                                                      input_length=self.max_length,
                                                      mask_zero=True,
                                                      embeddings_initializer=self.embedding_initializer,
                                                      trainable=self.finetune_embeddings))

        self.classifier.add(tf.keras.layers.Conv1D(self.feature_maps, self.n_gram, self.embed_vec_len, data_format='channels_first'))
        self.classifier.add(tf.keras.layers.Dropout(self.dropout))

        # self.classifier.add(tf.keras.layers.MaxPooling2D(poolsize=(self.max_length - self.n_gram + 1, 1)))

        self.classifier.add(tf.keras.layers.LSTM(units=self.neurons, input_shape=(self.max_length, self.embed_vec_len),
                                                 kernel_initializer=init, dropout=self.dropout,
                                                 recurrent_dropout=self.rec_dropout))

        self.classifier.summary()

        if self.hidden_neurons > 0:
            self.classifier.add(
                tf.keras.layers.Dense(units=self.hidden_neurons, kernel_initializer=init, activation='elu'))
        self.classifier.add(tf.keras.layers.Dense(units=1, kernel_initializer=init, activation=self.activ))
        self.classifier.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['acc'])
        print(self.classifier.summary())
        if self.early_stopping:
            self.es.append(
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=self.patience))

    def preprocess(self, train_data, y, weights=None):
        if self.embedding_dict is None:
            print('Reloading embedding index')
            try:
                self.embedding_dict = {}
                with open('/.glove_dicts/glove.twitter.27B.' + str(self.embed_vec_len) + 'd.txt', encoding="utf8") as f:
                    for line in f:
                        word, representation = line.split(maxsplit=1)
                        representation = np.fromstring(representation, 'f', sep=' ')
                        self.embedding_dict[word] = representation

                print('Dictionary loaded')

            except FileNotFoundError:
                file_fetcher.download_file("http://nlp.stanford.edu/data/glove.twitter.27B.zip",
                                           "glove_dicts.zip")
                with ZipFile('glove_dicts.zip', 'r') as zipObj:
                    zipObj.extractall('glove_dicts')
                self.embedding_dict = {}
                with open('.glove_dicts/glove.twitter.27B.' + str(self.embed_vec_len) + 'd.txt', encoding="utf8") as f:
                    for line in f:
                        word, representation = line.split(maxsplit=1)
                        representation = np.fromstring(representation, 'f', sep=' ')
                        self.embedding_dict[word] = representation

                print('Dictionary loaded')

        if weights is None:
            weights = np.ones(len(y))

        if 1 < self.bootstrap < len(y):
            train_data, y, weights = resample(train_data, y, weights, n_samples=self.bootstrap, stratify=y,
                                              replace=False)
        elif self.bootstrap < 1:
            n_samples = int(self.bootstrap * len(y))
            train_data, y, weights = resample(train_data, y, weights, n_samples=n_samples, stratify=y,
                                              replace=False)

        print('Sampled %d training points' % len(train_data))

        filtered_data = tokenizer_filter(train_data, remove_punctuation=self.remove_punctuation,
                                         remove_stopwords=self.remove_stopwords, lemmatize=self.lemmatize,
                                         lemmatize_pronouns=False)
        print('Filtered data')

        cleaned_data = [' '.join(tweet) for tweet in filtered_data]
        return cleaned_data, y, weights

    def fit(self, train_data, y, weights=None, custom_vocabulary=None, clear_embedding_dictionary=True):
        """
        :param train_data: (Dataframe) Training data
        :param y: (vector) Targets
        :param weights: (vector) Weights for fitting data
        :param custom_vocabulary: Custom vocabulary for the tokenizer. Not recommended.
        :param clear_embedding_dictionary: Delete the embedding dictionary after loading the embedding layer.
        Recommended, but will prevent the model from being re-fit (not refined)
        :returns Fit history
        """

        """
        # Preprocess and tokenize text
        """
        cleaned_data, y, weights = self.preprocess(train_data, y, weights)

        if custom_vocabulary is not None:
            print('Applying custom vocabulary')
            self.tokenizer = Tokenizer(num_words=len(custom_vocabulary))
            self.tokenizer.fit_on_texts(custom_vocabulary)
        else:
            print('Fitting tokenizer')
            self.tokenizer = Tokenizer(num_words=self.vocab_size, char_level=False)
            self.tokenizer.fit_on_texts(cleaned_data)

        train_sequences = self.tokenizer.texts_to_sequences(cleaned_data)

        self.word_index = self.tokenizer.word_index

        X = pad_sequences(train_sequences, maxlen=self.max_length, padding='pre')

        self.embedding_matrix = np.zeros((len(self.word_index) + 1, self.embed_vec_len))
        for word, i in self.word_index.items():
            embedding_vector = self.embedding_dict.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros. # TODO consider optimizing
                self.embedding_matrix[i] = embedding_vector

        self.build_NGRAM_network()

        if clear_embedding_dictionary:
            self.embedding_matrix = None
            self.embedding_dict = None

        print('Fitting GloVE model')

        history = self.classifier.fit(X, y, validation_split=self.validation_split, batch_size=self.batch_size,
                                      epochs=self.max_iter, sample_weight=weights,
                                      callbacks=self.es, verbose=2)

        self.accuracy = np.max(history.history['val_acc'])
        return history

    def export(self, filename):
        """
        Saves the model to disk
        :param filename: (String) Path to file
        """

        parameters = {'Classifier': self.type,
                      'package': self.package,
                      'max_length': int(self.max_length),
                      'neurons': int(self.neurons),
                      'hidden_neruons': int(self.hidden_neurons),
                      'dropout': float(self.dropout),
                      'rec_dropout': float(self.rec_dropout),
                      'activ': self.activ,
                      'vocab_size': int(self.vocab_size),
                      'max_iter': int(self.max_iter),
                      'batch_size': self.batch_size,
                      'early_stopping': self.early_stopping,
                      'patience': int(self.patience),
                      'bootstrap': self.bootstrap,
                      'validation_split': float(self.validation_split),
                      'accuracy': float(self.accuracy),
                      'remove_punctuation': self.remove_punctuation,
                      'remove_stopwords': self.remove_stopwords,
                      'lemmatize': self.lemmatize,
                      'finetune_embeddings': self.finetune_embeddings,
                      'learning_rate': self.learning_rate,
                      'n_gram': self.n_gram,
                      'feature_maps': self.feature_maps

                      }

        if parameters['bootstrap'] < 1:
            parameters['bootstrap'] = float(parameters['bootstrap'])
        else:
            parameters['bootstrap'] = int(parameters['bootstrap'])

        os.makedirs(filename, exist_ok=True)
        with open(filename + '/param.json', 'w+') as outfile:
            json.dump(parameters, outfile)
        with open(filename + '/ngram_tokenizer.pkl', 'wb+') as outfile:
            pkl.dump(self.tokenizer, outfile)
        # model_json = self.classifier.to_json()
        with open(filename + "/ngram_model.json", "w+") as json_file:
            json_file.write(self.classifier.to_json())
        self.classifier.save_weights(filename + "/ngram_model.h5")

    def load_model(self, filename):
        """
        :param filename: (String) Path to file
        """
        self.tokenizer = pkl.load(open(filename + '/glove_tokenizer.pkl', 'rb'))
        with open(filename + '/glove_model.json', 'r') as infile:
            model_json = infile.read()
        self.classifier = tf.keras.models.model_from_json(model_json)
        self.classifier.load_weights(filename + '/glove_model.h5')
        self.classifier.compile(loss='binary_crossentropy',
                                optimizer=self.optimizer,
                                metrics=['acc'])

class Charlevel_Model(LSTM_Model):
    """
    LSTM model that uses GloVE pre-trained embeddings
    # TODO add automatic embedding downloading and unzipping
    """

    def __init__(self, embedding_dict=None, embed_vec_len=128, max_length=140, vocab_size=128, batch_size=10000,
                 neurons=100,
                 hidden_neurons=0, dropout=0.2, bootstrap=1, early_stopping=True, validation_split=0.2, patience=50,
                 max_iter=250,
                 rec_dropout=0.2, activ='hard_sigmoid', accuracy=0, remove_punctuation=False, learning_rate=0.001,
                 remove_stopwords=False, lemmatize=False, finetune_embeddings=True, n_grams=[3,4,5], feature_maps=10, bidirectional=False, **kwargs):
        """
        Constructor for NGRAM LSTM classifier using pre-trained embeddings
        Be sure to add extra parameters to export()
        :param glove_index: (Dict) Embedding index to use. IF not provided, a standard one will be downloaded
        :param name: (String) Name of model
        :param embed_vec_len: (int) Embedding depth. Inferred from dictionary if provided. Otherwise 25, 50, 100, and
        are acceptible values. 200
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
        self.type = 'Charlevel_Model'
        self.package = 'twitter_nlp_toolkit.tweet_sentiment_classifier.models.lstm_models'
        self.bootstrap = bootstrap
        self.early_stopping = early_stopping
        self.validation_split = validation_split
        self.patience = patience
        self.max_iter = max_iter
        self.embed_vec_len = embed_vec_len
        self.learning_rate = learning_rate

        self.embedding_initializer = None

        self.finetune_embeddings = finetune_embeddings
        self.max_length = max_length
        self.embedding_dict = embedding_dict
        self.max_iter = max_iter
        self.vocab_size = vocab_size
        self.neurons = neurons
        self.hidden_neurons = hidden_neurons
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.activ = activ
        self.optimizer = 'adam'
        self.batch_size = batch_size

        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.es = []

        self.tokenizer = None
        self.classifier = None
        self.word_index = None
        self.embedding_matrix = None
        self.accuracy = accuracy

        self.n_grams = n_grams
        self.feature_maps = feature_maps
        self.bidirectional = bidirectional

        if self.embedding_dict is not None:
            self.embed_vec_len = len(list(self.embedding_dict.values())[0])
            print('Setting embedding depth to {}'.format(self.embed_vec_len))

    def build_charlevel_network(self):
        # TODO consider bidirectional
        print("Creating character-level model")
        if self.optimizer == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        init = tf.keras.initializers.glorot_uniform(seed=1)

        inputs = tf.keras.Input(shape=(self.max_length,))
        embeddings = tf.keras.layers.Embedding(input_dim=len(self.word_index) + 1,
                                                      output_dim=self.embed_vec_len,
                                                      input_length=self.max_length,
                                                      mask_zero=False,
                                                      embeddings_initializer=self.embedding_initializer,
                                                      trainable=self.finetune_embeddings)(inputs)
        reshape = tf.keras.layers.Reshape((self.max_length, self.embed_vec_len, 1))(embeddings)
        outputs = []
        for ngram in self.n_grams:
            conv_layer = tf.keras.layers.Conv2D(self.feature_maps, kernel_size=ngram)(reshape)
            #reshape_layer = tf.keras.layers.Reshape((self.max_length, self.feature_maps, 1))(conv_layer)
            pooling_layer = tf.keras.layers.MaxPooling2D(pool_size=(1, self.embed_vec_len-ngram), data_format='channels_last')(conv_layer)
            reshape_layer = tf.keras.layers.Reshape((self.max_length-ngram+1,self.feature_maps))(pooling_layer)
            if self.bidirectional:
                outputs.append(tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(units=self.neurons, input_shape=(self.max_length, self.embed_vec_len),
                                         kernel_initializer=init, dropout=self.dropout,
                                         recurrent_dropout=self.rec_dropout))(reshape_layer))
            else:
                outputs.append(tf.keras.layers.LSTM(units=self.neurons, input_shape=(self.max_length, self.embed_vec_len),
                                                 kernel_initializer=init, dropout=self.dropout,
                                                 recurrent_dropout=self.rec_dropout)(reshape_layer))
        output = tf.keras.layers.concatenate(outputs)

        flattened = tf.keras.layers.Flatten()(output)
        if self.hidden_neurons > 0:
            hidden_layer = tf.keras.layers.Dense(units=self.hidden_neurons, kernel_initializer=init, activation='relu')(flattened)
            dropout = tf.keras.layers.Dropout(self.dropout)(hidden_layer)
        else:
            dropout = tf.keras.layers.Dropout(self.dropout)(flattened)

        output = tf.keras.layers.Dense(units=1, kernel_initializer=init, activation=self.activ)(dropout)
        self.classifier = tf.keras.Model(inputs=inputs, outputs=output)
        self.classifier.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['acc'])

        self.classifier.summary()

        if self.early_stopping:
            self.es.append(
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=self.patience))

    def preprocess(self, train_data, y, weights=None):

        if weights is None:
            weights = np.ones(len(y))

        if 1 < self.bootstrap < len(y):
            train_data, y, weights = resample(train_data, y, weights, n_samples=self.bootstrap, stratify=y,
                                              replace=False)
        elif self.bootstrap < 1:
            n_samples = int(self.bootstrap * len(y))
            train_data, y, weights = resample(train_data, y, weights, n_samples=n_samples, stratify=y,
                                              replace=False)

        print('Sampled %d training points' % len(train_data))

        filtered_data = tokenizer_filter(train_data, remove_punctuation=self.remove_punctuation,
                                         remove_stopwords=self.remove_stopwords, lemmatize=self.lemmatize,
                                         lemmatize_pronouns=False)
        print('Filtered data')

        cleaned_data = [' '.join(tweet) for tweet in filtered_data]
        print(cleaned_data[0])
        return cleaned_data, y, weights

    def fit(self, train_data, y, weights=None, custom_vocabulary=None):
        """
        :param train_data: (List-like of Strings) Tweets to fit on
        :param y: (Vector) Targets
        :param weights: (Vector) Weights for fitting data
        :param custom_vocabulary: (List of String) Custom vocabulary to use for tokenizer. Not recommended.
        :return: Fit history

        """

        cleaned_data, y, weights = self.preprocess(train_data, y, weights)

        self.tokenizer = Tokenizer(num_words=self.vocab_size, filters='"#$%&()*+-/:;<=>?@[\\]^_`{|}~\t\n',
                                   char_level=True, oov_token=0)
        self.tokenizer.fit_on_texts(cleaned_data)

        train_sequences = self.tokenizer.texts_to_sequences(cleaned_data)

        self.word_index = self.tokenizer.word_index
        print('Found %s unique tokens.' % len(self.word_index))

        X = pad_sequences(train_sequences, maxlen=self.max_length, padding='pre')

        self.build_charlevel_network()

        print('Fitting LSTM model')

        history = self.classifier.fit(X, y, validation_split=self.validation_split, callbacks=self.es,
                                      batch_size=self.batch_size, sample_weight=weights,
                                      epochs=self.max_iter, verbose=1, steps_per_epoch=100)

        self.accuracy = np.max(history.history['val_acc'])
        return history

    def export(self, filename):
        """
        Saves the model to disk
        :param filename: (String) Path to file
        """

        parameters = {'Classifier': self.type,
                      'package': self.package,
                      'max_length': int(self.max_length),
                      'neurons': int(self.neurons),
                      'hidden_neruons': int(self.hidden_neurons),
                      'dropout': float(self.dropout),
                      'rec_dropout': float(self.rec_dropout),
                      'activ': self.activ,
                      'vocab_size': int(self.vocab_size),
                      'max_iter': int(self.max_iter),
                      'batch_size': self.batch_size,
                      'early_stopping': self.early_stopping,
                      'patience': int(self.patience),
                      'bootstrap': self.bootstrap,
                      'validation_split': float(self.validation_split),
                      'accuracy': float(self.accuracy),
                      'remove_punctuation': self.remove_punctuation,
                      'remove_stopwords': self.remove_stopwords,
                      'lemmatize': self.lemmatize,
                      'finetune_embeddings': self.finetune_embeddings,
                      'learning_rate': self.learning_rate,
                      'n_gram': self.n_gram,
                      'feature_maps': self.feature_maps,
                      'bidirectional': self.bidirectional
                      }

        if parameters['bootstrap'] < 1:
            parameters['bootstrap'] = float(parameters['bootstrap'])
        else:
            parameters['bootstrap'] = int(parameters['bootstrap'])

        os.makedirs(filename, exist_ok=True)
        with open(filename + '/param.json', 'w+') as outfile:
            json.dump(parameters, outfile)
        with open(filename + '/charlevel_tokenizer.pkl', 'wb+') as outfile:
            pkl.dump(self.tokenizer, outfile)
        # model_json = self.classifier.to_json()
        with open(filename + "/charlevel_model.json", "w+") as json_file:
            json_file.write(self.classifier.to_json())
        self.classifier.save_weights(filename + "/charlevel_model.h5")

        tf.keras.utils.plot_model(
            self.classifier,
            to_file=(filename + "/model_topology.png"),
            show_shapes=True,
            show_layer_names=True)

    def load_model(self, filename):
        """
        :param filename: (String) Path to file
        """
        self.tokenizer = pkl.load(open(filename + '/glove_tokenizer.pkl', 'rb'))
        with open(filename + '/glove_model.json', 'r') as infile:
            model_json = infile.read()
        self.classifier = tf.keras.models.model_from_json(model_json)
        self.classifier.load_weights(filename + '/glove_model.h5')
        self.classifier.compile(loss='binary_crossentropy',
                                optimizer=self.optimizer,
                                metrics=['acc'])

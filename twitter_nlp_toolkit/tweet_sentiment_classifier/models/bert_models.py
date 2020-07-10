from ..tweet_sentiment_classifier import Classifier

import numpy as np
import pickle as pkl

import json
import os

import tensorflow_hub as hub
import tensorflow as tf

try:
    import bert

    FullTokenizer = bert.bert_tokenization.FullTokenizer
except ImportError:
    raise ImportError('Issue loading bert-for-tf, please ensure it is installed')

try:
    import tensorflow_hub
except ImportError:
    raise ImportError('Issue loading bert-for-tf, please ensure it is installed')

from tensorflow.keras.models import Model
import math
from random import choice

import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split


def get_masks(tokens, max_seq_len):
    """Mask for padding"""
    if len(tokens) > max_seq_len:
        raise IndexError("Token length more than max seq length!")
    return [1] * len(tokens) + [0] * (max_seq_len - len(tokens))


def get_segments(tokens, max_seq_len):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens) > max_seq_len:
        raise IndexError("Token length more than max seq length!")
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_len - len(tokens))


def get_ids(tokens, tokenizer, max_seq_len):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_len - len(token_ids))
    return input_ids


class BERT_Model(Classifier):
    def __init__(self, model="bert_en_uncased_L-12_H-768_A-12/1", max_length=48, patience=10, early_stopping=True,
                 validation_split=0.2, max_iter=500, bootstrap=1,
                 batch_size=32, accuracy=0, activ='sigmoid', optimizer=tf.keras.optimizers.Adam,
                 learning_rate=1E-4, finetune_embeddings=True, hidden_neurons=0,  **kwargs):
        self.type = 'BERT_Model'
        self.package = 'twitter_nlp_toolkit.tweet_sentiment_classifier.models.bert_models'
        self.model = model
        self.max_length = max_length
        self.bootstrap = bootstrap
        self.early_stopping = early_stopping
        self.patience = patience
        self.validation_split = validation_split
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.accuracy = accuracy
        self.activ = activ
        self.optimzier = optimizer
        self.learning_rate = learning_rate
        self.finetune_embeddings = finetune_embeddings
        self.hidden_neurons = hidden_neurons

        self.loss = 'binary_crossentropy'

        self.tokenizer, self.vocab_file, self.classifier = None, None, None

        input_word_ids = tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int32,
                                               name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int32,
                                           name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int32,
                                            name="segment_ids")
        bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                                    trainable=self.finetune_embeddings)
        pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

        bert_model = tf.keras.Model(inputs=[input_word_ids, input_mask, segment_ids],
                                    outputs=[pooled_output, sequence_output])
        if self.hidden_neurons is not 0:
            hidden_layer = tf.keras.layers.Dense(units=self.hidden_neurons, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=1),
                                  activation='relu')(bert_model.outputs[0])
            dropout_layer = tf.keras.layers.Dropout(0.25)(hidden_layer)
            output_layer = tf.keras.layers.Dense(units=1,
                                                 kernel_initializer=tf.keras.initializers.glorot_uniform(seed=1),
                                                 activation=self.activ)(dropout_layer)
        else:
            output_layer = tf.keras.layers.Dense(units=1,
                                                 kernel_initializer=tf.keras.initializers.glorot_uniform(seed=1),
                                                 activation=self.activ)(bert_model.outputs[0])

        self.classifier = tf.keras.Model(inputs=bert_model.inputs, outputs=output_layer)
        self.classifier.compile(loss=self.loss, optimizer=self.optimzier(learning_rate=self.learning_rate, clipnorm=1), metrics=['acc'])
        self.vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
        self.tokenizer = FullTokenizer(self.vocab_file, do_lower_case)

    def preprocess(self, tweets, verbose=True):
        sequences = ([["[CLS]"] + self.tokenizer.tokenize(tweet) + ["[SEP]"] for tweet in tweets])
        input_ids = []
        input_masks = []
        input_segments = []
        i = 0

        """
        Tokenize data
        """

        for sequence in sequences:
            # TODO this is very slow, see if it can be sped up
            if verbose and i % 100 == 0:
                print('\r Processing tweet {}'.format(i), end=' ')

            if len(sequence) > self.max_length:
                sequence = sequence[-self.max_length:-1]
            input_ids.append(get_ids(sequence, self.tokenizer, self.max_length))
            input_masks.append(get_masks(sequence, self.max_length))
            input_segments.append(get_segments(sequence, self.max_length))

            i = i + 1
        print('Preprocessed {} tweets'.format(i))
        return [np.array(input_ids), np.array(input_masks), np.array(input_segments)]

    def fit(self, train_data, y, weights=None, **kwargs):

        if 1 < self.bootstrap < len(y):
            train_data, y, weights = resample(train_data, y, weights, n_samples=self.bootstrap, stratify=y,
                                              replace=False)
        elif self.bootstrap < 1:
            n_samples = int(self.bootstrap * len(y))
            train_data, y, weights = resample(train_data, y, weights, n_samples=n_samples, stratify=y,
                                              replace=False)

        trainX = self.preprocess(train_data)
        trainy = np.array(y)
        if weights is None:
            weights = np.ones(len(y))

        es = []
        if self.early_stopping:
            es.append(
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=self.patience))

        print('Fitting BERT classifier')
        history = self.classifier.fit(trainX, trainy, sample_weight=weights, epochs=self.max_iter, batch_size=self.batch_size,
                                      verbose=1, validation_split=self.validation_split, callbacks=es, steps_per_epoch=10000)

        self.accuracy = np.max(history.history['val_acc'])
        return history

    def predict_proba(self, data, **kwargs):
        X = self.preprocess(data, verbose=False)
        predictions = self.classifier.predict(X, **kwargs)
        return predictions

    def refine(self, train_data, y, weights=None, bootstrap=True, **kwargs):
        if (bootstrap and 1 < self.bootstrap < len(y)):
            train_data, y, weights = resample(train_data, y, weights, n_samples=self.bootstrap, stratify=y,
                                              replace=False)
        elif (bootstrap and self.bootstrap < 1):
            n_samples = int(self.bootstrap * len(y))
            train_data, y, weights = resample(train_data, y, weights, n_samples=n_samples, stratify=y,
                                              replace=False)

        trainX = self.preprocess(train_data)
        trainy = np.array(y)
        if weights is None:
            weights = np.ones(len(y))

        es = []
        if self.early_stopping:
            es.append(
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=self.patience))

        print('Fitting BERT classifier')
        history = self.classifier.fit(trainX, trainy, sample_weight=weights, epochs=self.max_iter,
                                      batch_size=self.batch_size,
                                      verbose=1, validation_split=self.validation_split, callbacks=es)

        self.accuracy = np.max(history.history['val_acc'])
        return history

    def predict(self, data, **kwargs):
        predictions = self.predict_proba(data)
        return np.round(predictions, **kwargs)

    def export(self, filename):
        """
        Saves the classifier to disk
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
                      'activ': self.activ,
                      'batch_size': self.batch_size,
                      'accuracy': float(self.accuracy),
                      }

        if parameters['bootstrap'] < 1:
            parameters['bootstrap'] = float(parameters['bootstrap'])
        else:
            parameters['bootstrap'] = int(parameters['bootstrap'])

        os.makedirs(filename, exist_ok=True)
        with open(filename + '/param.json', 'w+') as outfile:
            json.dump(parameters, outfile)

        model_json = self.classifier.to_json()
        self.classifier.save_weights(filename + "/bert_model.h5")

    def load_model(self, filename):
        """
        Load a model from the disc
        :param filename: (String) Path to file
        """

        self.classifier.load_weights(filename + '/bert_model.h5')
        self.classifier.compile(loss='binary_crossentropy',
                                optimizer=self.optimzier(learning_rate=self.learning_rate),
                                metrics=['acc'])

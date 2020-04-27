import json
import os
import numpy as np
import spacy
import dload

from sklearn.metrics import accuracy_score
from zipfile import ZipFile

from twitter_nlp_toolkit.file_fetcher import file_fetcher

# Urgent TODO tokenizer_filter appears to hang when a tweet contains a a single unprocessable word - confirm and fix
# Urgent TODO fitting tokenizer appears to be bugged when preprocess=False. Confirm and apply fix of casting train_data
#  to list
# Possible TODO consider renaming glove to pre-trained as it in principle should work with any pre-trained index
# Possible TODO find way to limit learning rate when refining BoW model
# Possible TODO set up Model/submodel inheritance

def tokenizer_filter(text, remove_punctuation=True, remove_stopwords=True, lemmatize=True, lemmatize_pronouns=False,
                     verbose=True):
    """
    :param text: (series) Text to process
    :param remove_punctuation: (bool) Strip all punctuation
    :param remove_stopwords: (bool) Remove all stopwords TODO enable custom stopword lists
    :param lemmatize: (bool) Lemmatize all words
    :param lemmatize_pronouns: (bool) lemmatize pronouns to -PRON-
    :return: (list) tokenized and processed text
    """

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
            if verbose and count % 1000 == 0:
                print('\r Preprocessed %d tweets' % count, end=' ')
        if verbose: print('\r Preprocessed %d tweets' % count)
        return filtered_tokens
    elif lemmatize:
        for doc in nlp.pipe(docs, n_threads=-1, batch_size=10000):
            # pronouns lemmatize to -PRON- which is undesirable when using pre-trained embeddings
            tokens = [token.lemma_.lower() if token.lemma_ != '-PRON-'
                      else token.lower_ for token in doc if token_filter(token)]
            count = count + 1
            # Starting with a carriage return rather than ending with one keeps the text visible in some IDEs.
            if verbose and count % 1000 == 0:
                print('\r Preprocessed %d tweets' % count, end=' ')
            filtered_tokens.append(tokens)
        if verbose: print('\r Preprocessed %d tweets' % count)
        return filtered_tokens
    else:
        # lemmatizing pronouns to -PRON- is desirable when not using pre-trained embeddings
        for doc in nlp.pipe(docs, n_threads=-1, batch_size=10000):
            tokens = [token.lower_ for token in doc if token_filter(token)]
            filtered_tokens.append(tokens)
            count = count + 1
            # Starting with a carriage return rather than ending with one keeps the text visible in some IDEs.
            if verbose and count % 1000 == 0:
                print('\r Preprocessed %d tweets' % count, end=' ')
        if verbose: print('\r Preprocessed %d tweets' % count)
        return filtered_tokens


class SentimentAnalyzer:
    def __init__(self, models=[], model_path=None):
        """
        Constructor for SentimentAnalyzer module
        :param models: (list) Models to initialize. Should be a list of tuples formatted like (name, type, params)
        type can be 'bow', 'lstm', or 'glove', and params is a dictionary of parameters
        """
        self.models = {}
        self.model_path = model_path

        if self.model_path is None:
            self.model_path = '.models'

        for name, type, params in models:
            if type == 'bow':
                self.add_bow_model(name, **params)
            elif type == 'lstm':
                self.add_lstm_model(name, **params)
            elif type == 'glove':
                self.add_glove_model(name, **params)
            else:
                print('Model type %s not recognized' % type)

    def load_small_ensemble(self, **kwargs):
        # TODO improve model choice
        # small_ensemble = ['glove_25N_1M_weighted_50d_1', 'glove_25N_1M_weighted_50d_2', 'glove_25N_1M_weighted_50d_3']
        try:
            self.load_models(path=self.model_path + '/small_ensemble', **kwargs)
            assert (len(self.models) > 2)
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            print('Downloading ensemble')
            # Compressed model
            self.download_small_ensemble()
            self.load_models(path=self.model_path + '/small_ensemble', **kwargs)

    def download_small_ensemble(self, **kwargs):
        file_fetcher.download_file("https://www.dropbox.com/s/ave5cmw6imhi74q/small_ensemble.zip?dl=1", "small_ensemble.zip")
        with ZipFile('small_ensemble.zip', 'r') as zipObj:
            zipObj.extractall(path=self.model_path + '/small_ensemble')

    def add_model(self, model, name, **kwargs):
        """
        Add a model.
        :param model: Constructor function for model
        :param name: Model name
        :param kwargs: Keyword arguments
        """
        self.models[name] = model(**kwargs)

    def set_model_path(self, model_path):
        """
        Set model path
        """
        self.model_path = model_path

    def add_bow_model(self, name, **kwargs):
        """
        Add another BoW model to the classifier
        :param vocab_size: (int) Maximum vocabulary size. Default 1E6
        :param max_iter: (int) Maximum number of fit iterations
        :param remove_punctuation: (Bool) Remove punctuation. Recommended.
        :param remove_stopwords: (Bool) Remove stopwords. Recommended.
        :param lemmatize: (Bool) Lemmatize words. Recommended.
        """
        from .models.bow_models import BoW_Model

        self.models[name] = BoW_Model(**kwargs)

    def add_lstm_model(self, name, **kwargs):
        """
        Add another LSTM model to the classifier
        :param max_length: (int) Maximum text length, ie, number of temporal nodes. Default 25
        :param vocab_size: (int) Maximum vocabulary size. Default 1E7
        :param max_iter: (int) Number of training epochs. Default 100
        :param neurons: (int) Depth (NOT LENGTH) of LSTM network. Default 100
        :param dropout: (float) Dropout
        :param activ: (String) Activation function (for visible layer). Default 'hard_sigmoid'
        :param optimizer: (String) Optimizer. Default 'adam'
        """
        from .models.lstm_models import LSTM_Model

        self.models[name] = LSTM_Model(**kwargs)

    def add_glove_model(self, name, glove_index=None, **kwargs):
        """
        Add another lstm model with pre-trained embeddings to the classifier.
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
        from .models.lstm_models import GloVE_Model

        self.models[name] = GloVE_Model(glove_index=glove_index, **kwargs)

    def delete_models(self, models):
        """
        Delete models
        :param models: (list of Strings) List of names of models to delete
        """
        for model in models:
            self.delete_model(model)

    def delete_model(self, model):
        """
        Delete a single model
        :param model: (String) name of model to delete
        """
        del self.models[model]

    def trim_models(self, testX, testY, threshold=0.7, metric=accuracy_score, models=[]):
        """
        Delete models that score below a threshold
        :param testX: Test data
        :param testY: Test labels
        :param threshold: Score threshold
        :param metric: Metric function to use
        :param models: Models to fest. Default: all
        :return:
        """
        if len(models) == 0:
            models = self.models.copy().keys()
        for name in models:
            score = metric(testY, self.models[name].predict(testX))
            print('Model %s score: %0.3f' % (name, score))
            if score < threshold:
                print('Deleting model %s' % name)
                self.delete_model(name)

    def fit(self, X, y, models=None, weights=None, custom_vocabulary=None, preprocess=True, **kwargs):
        """
        Fits the enabled models onto X. Note that this rebuilds the models, as it is not currently possible to update
        the tokenizers
        :param X: (array) Feature matrix
        :param y: (vector) Targets
        :param models: (list of Strings) Names of models to fit. Default: all. Note that default behavior will likely
        cause an error if additional models have been added after fitting a pre-trained embedding model
        :return:
        """

        if weights is None:
            weights = np.ones(len(y))
        else:
            weights = np.array(weights)

        if models is None:
            models = self.models.keys()

        for name in models:
            try:
                print('Fitting %s' % name)
                self.models[name].fit(X, y, weights=weights, custom_vocabulary=custom_vocabulary)
            except KeyError:
                print('Model %s not found!' % name)

    def refine(self, X, y, **kwargs):
        """
        Refits the trained models onto additional data. Not that this does NOT retrain the tokenizers, so it will not
        improve the vocabulary
        :param X: (array) Feature matrix
        :param y: (vector) Targets
        :param bootstrap: (bool) Bootstrap sample the refining data. Default True.
        """

        for model in self.models.values():
            model.refine(X, y, **kwargs)

    def predict(self, X):
        """
        Predicts the binary sentiment of a list of tweets by having the models vote
        :param X: (list-like of Strings) Input tweets
        :return: (list of Bool) Sentiment
        """

        predictions = self.predict_proba(X)

        return np.round(predictions)

    def predict_proba(self, X, **kwargs):
        """
        Predicts the continuous sentiment of a list of tweets by having the models vote
        :param X: (list-like of Strings) Input tweets
        :param kwargs: Parameters for predict_proba # TODO verify if this is necessary
        :return: (list of float) Sentiment
        """
        predictions = []

        for name, model in self.models.items():
            try:
                predictions.append(model.predict_proba(X, **kwargs).reshape(-1))
            except ValueError:
                print('Error using model %s - probably has not been trained' % name)

        return np.mean(predictions, axis=0)

    def save_models(self, names=None, path=None):
        """
    # TODO Rework
        Write models to disk for re-use. Uses self.model_path
        :param names: (List of Strings) List of models to save
        :param path: Path to save to. Defaults to self.model_path
        """
        if path is None:
            path = self.model_path

        if names is None:
            for name, model in self.models.items():
                model.export(path + '/' + name)
                print('Model %s saved' % name)
        else:
            for name in names:
                try:
                    self.models[name].export(path + '/' + name)
                    print('Model %s saved' % name)
                except KeyError:
                    print('Model %s not found!' % name)

    def load_models(self, filenames=None, path=None):
        """
    # TODO Rework
        Reloads saved models from the disk
        :param filenames: (list) List of model names to import
        :param path: (String) Directory to load from
        """
        if path is None:
            path = self.model_path

        if filenames is not None:
            for filename in filenames:
                self.load_model(path + '/' + filename)

        if filenames is None:
            filenames = [folder.name for folder in os.scandir(path) if folder.is_dir()]
            for filename in filenames:
                print(filename)
                self.load_model(filename, path)

    def load_model(self, filename, path):
        """
        # TODO Make IO failure more graceful
        # TODO this is ugly, consider refactoring
        :param filename:
        :return:
        """
        try:
            with open(path + '/' + filename + '/param.json') as infile:
                import importlib
                param = json.load(infile)
                module_name = param.pop('package')
                constructor_name = param.pop('Classifier')
                module = importlib.import_module(module_name)
                constructor = getattr(module, constructor_name)
                self.models[filename] = constructor(**param)
                self.models[filename].load_model(path + '/' + filename)
                print('Model {} loaded'.format(filename))

        except (FileNotFoundError, IOError, EOFError):
            try:
                self.models.pop(filename)
                print('Model {} failed to load completely...'.format(filename))
            except KeyError:
                print('Default parameters file not found or not working...')

            print('Trying %s' % path + '/' + filename)

            self.load_legacy_model(filename, path)

    def evaluate(self, X, y, metric=accuracy_score):
        """
        Evaluate each model
        :param X: (array) Feature matrix
        :param y: (vector) targets
        :param metric: (method) Metric to use
        # TODO try to improve performance by caching predictions
        """
        scores = {}
        for name, model in self.models.items():
            try:
                scores[name] = metric(y, model.predict(X))
                print('Model %s %s: %0.3f' % (name, metric.__name__, scores[name]))
            except ValueError:
                print('Error, %s probably has noct been trained' % name)

        if len(self.models.values()) > 1:
            scores['ensembled'] = metric(y, self.predict(X))
        return scores

    def load_legacy_model(self, filename, path):
        if os.path.exists(path + '/' + filename + '/bow_param.json'):
            try:
                from .models.bow_models import BoW_Model
                print('Loading BoW model {} from legacy parameter file'.format(filename))
                with open(path + '/' + filename + '/bow_param.json', 'r') as infile:
                    bow_param = json.load(infile)
                self.models[filename] = BoW_Model(**bow_param)
                self.models[filename].load_model(path + '/' + filename)
                print('BoW model %s loaded successfully' % filename)
            except FileNotFoundError:
                print('Model %s not found' % filename)
            except (IOError, EOFError):
                print('Problem reading %s files' % filename)

        elif os.path.exists(path + '/' + filename + '/lstm_param.json'):
            try:
                from .models.lstm_models import LSTM_Model
                print('Loading LSTM model {} from legacy parameter file'.format(filename))
                with open(path + '/' + filename + '/lstm_param.json', 'r') as infile:
                    lstm_param = json.load(infile)
                self.models[filename] = LSTM_Model(**lstm_param)
                self.models[filename].load_model(path + '/' + filename)
                print('LSTM model %s loaded successfully' % filename)

            except FileNotFoundError:
                print('Model %s not found' % filename)
            except (IOError, EOFError):
                print('Problem reading %s files' % filename)

        elif os.path.exists(path + '/' + filename + '/glove_param.json'):
            # elif True:
            try:
                from .models.lstm_models import GloVE_Model
                print('Loading GloVE model {} from legacy parameter file'.format(filename))
                glove_param = json.load(open(path + '/' + filename + '/glove_param.json', 'r'))
                self.models[filename] = GloVE_Model(None, **glove_param)
                self.models[filename].load_model(path + '/' + filename)

                print('Pre-trained embedding model %s loaded successfully' % filename)
            except FileNotFoundError:
                print('Model %s not found' % filename)
            except (IOError, EOFError):
                print('Problem reading %s files' % filename)

        else:
            print('Folder {} does not appear to be a saved model...'.format(filename))


class Classifier:
    """
    Parent Classifier class
    # TODO move redundant code here
    """

    def __init__(self):
        pass

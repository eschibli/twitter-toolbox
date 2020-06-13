import numpy as np
import pandas as pd
import pickle as pkl
import os

from random import choice
from sklearn.metrics import matthews_corrcoef, accuracy_score

from twitter_nlp_toolkit import tweet_sentiment_classifier
from sklearn.utils import resample
from tensorflow.keras.backend import clear_session

import tensorflow as tf


"""
Hyperparameters
"""

#glove_param_1 = {'max_iter': max_iter, 'vocab_size': vocab_size, 'bootstrap': 1, 'neurons': 30, 'hidden_neruons': 30,
#                'embed_vec_len': 200, 'finetune_embeddings': False, 'batch_size': 1028}

# Set CPU as available physical device
"""
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')
"""

bootstrap_sizes = [1]

tweet_samples = [10000]
review_samples = [1, 1, 1, 10000]


tweet_samples = [1000000, 300000, 1000000]
review_samples = [1, 1, 1, 10000, 100000, 1000000]
vocab_sizes = [10000, 30000, 50000, 100000]
neuron_counts = [30, 50, 100]
hidden_neruon_counts = [0, 0, 0]
embedding_depths = [50, 100, 200]
max_lengths = np.linspace(24, 36, dtype=int)
batch_sizes = [512, 1028]
learning_rates = [6E-5, 1E-4, 3E-4, 1E-3]
n_grams = [1, 2, 3, 4, 5]
feature_maps = [5, 10, 20, 30, 50, 100]
binary = [True, False]
"""
Other parameters
"""

train_filename = 'training_data'
test_filename = 'airline_data'  # Our test dataset is the airline data
max_length = 25
model_path = 'parameter_scan_models'

max_iter = int(500)
training_samples = None  # Fraction of training dataset to use - leave at None unless debugging
keep_neutral_tweets = 0  # Training data has unlabeled "neutral" tweets. Leave at 0
min_tweet_length = 5  # Minimum number of words in tweet to train

rebuild_model = True
reload_embeddings_index = False

# embedding_index_file = 'glove_index_50d.pkl'

"""
Import Data
"""

# Import data


train_filename = 'Train'
test_filename = 'Test'

# We will use a dataset of general tweets (double-check where it's from) for training


sent140_data = pd.read_csv('training_data/sent140_and_amazon_electronics.csv', encoding='latin-1',
                   names=['Index', 'Text', 'Labels', 'Source'], skiprows=1,
                   dtype={'Index': int, 'Text': str, 'Labels': int})

"""
data = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1',
                   names=['Null', 'Index', 'Date', 'Query', 'User',  'Text', 'Labels'], skiprows=1)
data['Weights'] = 1
"""

amazon_data = sent140_data[sent140_data['Source'] == 'amazon']
twitter_data = sent140_data[sent140_data['Source'] == 'twitter']

amazon_data['Weights'] = 1
twitter_data['Weights'] = 1

airline_data = pd.read_csv('./training_data/tweets_airline.csv', header=0,
                           names=['Index', 'Sentiment', 'Sentiment_confidence',
                                  'Negative_reason', 'Negative_reason_confidence',
                                  'Airline', 'Airline_sentiment_gold', 'Handle',
                                  'Negative_reason_gold', 'Retweet_count', 'Text',
                                  'Tweet_coord', 'Time', 'Location', 'Timezone'])
MIT_data = pd.read_csv('./training_data/tweets_MIT.txt', header=0, names=['Labels', 'Text'], sep='\t')

coachella_data = pd.read_csv('./training_data/tweets_coachella.csv', header=0, names=['Sentiment',
                                                                                      'coachella_yn', 'name',
                                                                                      'retweet_count',
                                                                                      'Text', 'tweet_coord',
                                                                                      'tweet_created', 'tweet_id',
                                                                                      'tweet_location',
                                                                                      'user_timezone'],
                             encoding='latin-1')

brands_data = pd.read_csv('./training_data/tweets_brands.csv', header=0, names=['Text', 'Subject', 'Sentiment'],
                          encoding='latin-1')


airline_data['Labels'] = (airline_data['Sentiment'] == 'positive') * 2
airline_data['Labels'] = airline_data['Labels'] + (airline_data['Sentiment'] == 'neutral') * 1
airline_data['Labels'] = airline_data['Labels'] / 2
airline_data.set_index('Labels')
airline_data = airline_data[airline_data.Labels != 0.5]

coachella_data['Labels'] = (coachella_data['Sentiment'] == 'positive')
brands_data['Labels'] = (brands_data['Sentiment'] == 'Positive emotion')


TweetClassifier = tweet_sentiment_classifier.SentimentAnalyzer()

brands_data['Text'] = brands_data['Text'].astype(str)
coachella_data['Text'] = coachella_data['Text'].astype(str)
airline_data['Text'] = airline_data['Text'].astype(str)
MIT_data['Text'] = MIT_data['Text'].astype(str)
amazon_data['Text'] = amazon_data['Text'].astype(str)
twitter_data['Text'] = twitter_data['Text'].astype(str)

#data['Text'] = pd.concat([twitter_data['Text'], MIT_data['Text']])
#data['Labels'] = pd.concat([twitter_data['Labels'], MIT_data['Labels']])

"""
Main Loop
"""
i = 0
while True:

    # Get embedding index

    # Choose random hyperparameters

    clear_session()
    embedding_depth = choice(embedding_depths)
    bootstrap_size = 1
    vocab_size = choice(vocab_sizes)
    neurons = choice(neuron_counts)
    hidden_neurons = choice(hidden_neruon_counts)
    max_length = choice(max_lengths)
    batch_size = choice(batch_sizes)
    learning_rate = choice(learning_rates)
    tweets = choice(tweet_samples)
    reviews = choice(review_samples)
    n_gram = choice(n_grams)
    feature_map = choice(feature_maps)
    finetune = choice([0, 1])
    """
    embedding_depth = 200
    bootstrap_size = 1
    vocab_size = 3E5
    neurons = 50
    hidden_neurons = 25
    max_length = 32
    batch_size = 512
    learning_rate = 3E-3
    tweets = 1E6
    reviews = 3E5
    """

    # Select the data
    amazon_samples = resample(amazon_data, n_samples=reviews, replace=False)
    twitter_samples = resample(twitter_data, n_samples=tweets, replace=False)
    coachella_samples = coachella_data
    brands_samples = brands_data
    train_data = pd.concat([twitter_samples, amazon_samples, coachella_data, brands_data])

    # Reinitialize the classifier
    TweetClassifier = tweet_sentiment_classifier.SentimentAnalyzer(model_path=model_path)

    params = {'bootstrap': bootstrap_size, 'embed_vec_len': embedding_depth, 'neurons': neurons,
              'hidden_neurons': hidden_neurons, 'max_length': max_length, 'batch_size': batch_size,
              'vocab_size': vocab_size, 'max_iter': 5000, 'learning_rate': learning_rate, 'patience': 20, 'finetune_embeddings': finetune,
              'n_grams':n_gram, 'feature_maps': feature_map}

    model_name = 'charlevel_' + str(embedding_depth) + 'D_' + str(neurons) + 'N_' + str(hidden_neurons) + 'N_' + str(vocab_size) + 'V_' \
                 + str(tweets) + 'T_' + str(reviews) + 'R_'+ str(n_gram) + 'NG_'+ str(feature_map) + 'FM_' + str(finetune) + 'F_'+ str(i)

    TweetClassifier.add_charlevel_model(model_name, **params)



    print('Training model %s ' % params)
    TweetClassifier.fit(train_data['Text'], train_data['Labels'])

    TweetClassifier.save_models([model_name])

    metrics = pd.Series(params).to_frame().T
    metrics['tweets'] = tweets
    metrics['reviews'] = reviews
    metrics['name'] = model_name
    airline_pred = TweetClassifier.predict(airline_data['Text']).reshape(-1)
    metrics['val_acc'] = TweetClassifier.models[model_name].accuracy
    metrics['airline_acc'] = accuracy_score(airline_data['Labels'], airline_pred)
    metrics['airline_mcc'] = matthews_corrcoef(airline_data['Labels'], airline_pred)

    print('Airlines accuracy: %0.3f' % metrics['airline_acc'])
    print('Airlines MCC: %0.3f' % metrics['airline_mcc'])

    metrics.to_csv('charlevel_metrics.csv', mode='a', header=(not os.path.exists('metrics.csv')))
    del TweetClassifier
    clear_session()

    i = i + 1

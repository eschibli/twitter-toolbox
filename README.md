
# Twitter Toolbox

A suite of tools for collecting, pre-processing, analyzing and sentiment-scoring twitter data. A additional brief walkthrough can be found [here](https://medium.com/@eric.schibli/simple-twitter-analytics-with-twitter-nlp-toolkit-7d7d79bf2535).

Install:
```bash
pip install twitter-nlp-toolkit
```

To utilize the sentiment analysis package, you will also need to install SpaCy's small English language model.

```bash
python -m spacy download en_core_web_sm
```

While the package is still under active development, the following functionality is expected to be stable:

## Listener

twitter_nlp_toolkit.twitter_listener is the listener module, which can be used to monitor Twitter and stream
tweets to disk in .json format.

```python
keywords = ["python"]
stream = twitter_listener.TwitterStreamListener(**credentials)
stream.collect_from_stream(max_tweets=10,output_json_name="python_tweets.json", target_words=keywords)
```
"keywords" uses the Twitter API. Documentation and tips for setting up smart keyword queries can be found [here](https://developer.twitter.com/en/docs/tweets/rules-and-filtering/overview/standard-operators)

"credentials" contains your Twitter API key, which can be obtained for free [here](https://developer.twitter.com/en/docs/basics/developer-portal/overview)

The module also contains a parser to convert the .json-formatted tweets into .csv for easy use (ie, with Pandas).

```python
parser = tweet_json_parser.json_parser()
parser.stream_json_file(json_file_name="python_tweets.json",output_file_name="parsed_python_tweets.csv")
```

## Bulk Downloader

twitter_nlp_toolkit.twitter_REST_downloader is the bulk download module, which can be used to collect the last
200 (or so) tweets from a single user.

```python
downloader = twitter_REST_downloader.bulk_downloader(**credentials)
downloader.get_tweets_csv_for_this_user("@nytimes","nyt_tweet_output.csv")
```

twitter_nlp_toolkit.tweet_sentiment_classifier is the sentiment analysis module, which can be used to
classify the sentiment of tweets.

```python
Classifier = tweet_sentiment_classifier.SentimentAnalyzer()
Classifier.load_small_ensemble()
Classifier.predict(['I am happy', 'I am sad', 'I am cheerful', 'I am mad']) # will return [1, 0, 1, 0]
```

Currently only two ensembles are provided: the small ensemble, which uses bag-of-words logistic regression model and
two long short-term memory neural networks, and the large ensemble, which uses the bog-of-words model, two larger LSTM
networks, and a Google BERT model. The large ensemble is more accurate (and expected to become much more accurate), but
is extremely resource intensive and as such, isn't recommended for processing large numbers of tweets unless you have
a powerful GPU.

These ensembles were trained primarily on the [Sent140 dataset](http://help.sentiment140.com/) and primarily tested
against the US Airlines dataset previously hosted on Crowdflower.com.

Please see the jupyter notebook (.ipynb) files at the root directory for further demonstrations of working code.

## Advanced Use

If you have domain-specific training data, you can refine the ensembles:

```
Classifier.refine(train_x, train_y)
Classifier.save_models()
# To reload your saved models, you can run Classifier.load_models()
```

Other advanced use, such as building your own models, is possible but is not currently recommended as the models
are still in development. Further documentation will be added once development stabilizes.

## Bugs, issues, contributions, and feature requests

The developers are always open to feature requests, bugs reports, pull requests, and new opportunities to collaborate.
Don't hesitate to reach out with questions, beedback, or requests.

Developers:

* Moe Antar (@Moe520) 
    * Twitter json parser and formatter
    * Twitter bulk downloader 
    * Twitter Listener, in collaboration with Dr. Mirko Miorelli (https://github.com/mirkomiorelli)
    * File downloader
    * General maintenance and deployment procedures
    
* Eric Schibli (@eschibli)
    * Data pre-processing pipeline
    * Natural language processing algorithms
    * Bag-of-words, LSTM neural networks, and BERT model assembly and training
    * General model optimization



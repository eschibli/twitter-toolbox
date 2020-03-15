import tweepy
import time
import json
import os
import csv

# Errors dictionary
ht_err = {0: 'Python version 3.0.0 or higher is required...',
          1: 'Failed to load Tweepy...',
          2: 'smtplib library not installed...',
          3: 'csv library not installed...',
          4: 'Unable to retrieve Twitter credentials... ',
          5: 'Unable to establish connection. Make sure your credentials are correct (Try re-generating your tokens).',
          6: 'Unable to retrieve keywords list...',
          7: 'Keyboard interrupt, stopping stream...',
          8: 'Unable to re-establish connection...'}


def c_exit(err_id):
    print("ERROR: " + ht_err[err_id])
    print("Exiting now...")
    time.sleep(1)
    exit(1)


# Initializing a listener class that streams from  Twitter
class StdOutListener(tweepy.StreamListener):

    # The init function of a class allows us to have variables that
    # are set to specific values when an instance of the class is intiialized
    # We can then change these attributes from outside the class by calling CLASSNAME.ATTRIBUTENAME = something
    def __init__(self, api=None):
        super(StdOutListener, self).__init__()
        # Initialize the customization attributes that we want to be able to change from the outside
        self.num_tweets = 0
        self.tweet_limit = 100
        self.file_label = "streamed_tweets"
        self.notification_interval = 1000

    def set_tweet_limit(self, tweet_quantity):
        self.tweet_limit = tweet_quantity

    def set_file_label(self, file_label):
        self.file_label = file_label

    def set_console_notification_interval(self, console_notification_interval):
        self.notification_interval = console_notification_interval

    def sanitize_string(self, s):
        result = s.replace(',', '')
        result = s.replace('\n', '')
        return result.rstrip()

    def on_data(self, data):
        # Based on the interval chosen by the user, let the user know
        # how many tweets have been collected so far
        if (self.num_tweets % self.notification_interval == 0):
            print("\t" + str(self.num_tweets) + " tweets collected...")

            # Increment the number of tweets collected by 1
            self.num_tweets += 1

            # Open file where to save tweets
            # f = open('%s.json' %self.file_label, 'a')

            # This is the main script that collects the tweets
            # It will terminate when it hits the tweets limit
            if (self.num_tweets < self.tweet_limit):
                try:
                    with open('%s' % self.file_label, 'a') as f:
                        f.write(data)
                except KeyboardInterrupt:
                    print("Keyboard Interrupt: Ending Stream")
                except BaseException as e:
                    print(str(e))
                return True
            else:
                print("Tweet Limit Reached: (%d) .... Closing Stream " % self.num_tweets)
                return False

    def on_error(self, status):
        print(status)


# Set listener parameters
def initialize_listener(max_tweets, file_label, console_interval):
    print("\nInitializing listener... ")

    # Initialize listener
    created_listener = StdOutListener()
    created_listener.set_tweet_limit(max_tweets)
    created_listener.set_file_label(file_label)
    created_listener.set_console_notification_interval(console_interval)
    return created_listener


def collect_tweets_from_stream(subjects, max_tweets, file_label,
                               auth_credentials, console_notification_interval, listener_object):
    my_listener = StdOutListener()
    my_listener.set_tweet_limit(max_tweets)
    my_listener.set_file_label(file_label)
    my_listener.set_console_notification_interval(
        console_notification_interval)
    stream = tweepy.Stream(auth_credentials, listener_object)

    print("Beginning Stream")
    print("Will collect until %d tweets are reached" % max_tweets)
    print("Output file will be called %s .csv " % file_label)

    try:
        stream.filter(track=subjects)
    except KeyboardInterrupt as e:
        print("Keyboard Interrupt: Stopping Stream")
    except:
        print("Listening function: Termination Caused by Unknown Error")

    print("Beginning Stream")
    print("Will collect until %d tweets are reached" % max_tweets)
    print("Output file will be called %s .csv " % file_label)

    try:
        stream.filter(track=subjects)
    except KeyboardInterrupt as e:
        print("Keyboard Interrupt: Stopping Stream")
    except:
        print("Listening function: Termination Caused by Unknown Error: ", str(e))


def collect_stream(my_listener, auth_credentials, keywords):
    """
    # Stream from twitter and collect dat
    :param my_listener:
    :param auth_credentials:
    :param keywords:
    :return:
    """
    count = 1
    # Maximum connections: 3 in 15 minutes
    while count < 4:
        print("\nConnecting listener to stream...")
        time.sleep(0.5)
        stream = tweepy.Stream(auth_credentials, my_listener)
        try:
            print("\n\tStreaming now...")
            stream.filter(track=keywords, languages=["en"])
            return
        # If keyboard exit interrupt
        except KeyboardInterrupt as e:
            c_exit(7)
        except:
            print("\n\tListening function: Termination Caused by Unknown Error...")
            print("\tWill attempt restart after 30 seconds...")
            for s in range(30):
                print("\t" + "\r" + str(s), '')
                time.sleep(1)

        stream.disconnect()

        count += 1

    if count == 2:
        c_exit()

    return


class stream_listener:
    # Initializer / Instance Attributes
    def __init__(self, consumer_key, consumer_secret, access_token, access_secret):
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token = access_token
        self.access_secret = access_secret
        self.GOOD_EMOJIS = ["😀", "😁", "😃", "😄", "😅", "😆", "😉",
                            "😊", "😋", "😎", "😗", "😙", "😚", "☺", "🙂", "🤗", "🤩", "😝"]
        self.BAD_EMOJIS = ['😣', '😥', '😯', '😫', '☹', '🙁', '😖', '😞', '😟', '😤', '😢',
                           '😭', '😦', '😧', '😡', '😠', '🤬', '🤢', '🤮', '🥺', '🤥', '😈', '👿', '🖕', '🖕🏻', '💔']
        self.auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)
        self.auth.set_access_token(self.access_token, self.access_secret)

    def collect_from_stream(self, max_tweets, output_json_name, console_interval, target_words):
        current_listener = initialize_listener(max_tweets, output_json_name, console_interval)
        collect_stream(current_listener, self.auth, target_words)

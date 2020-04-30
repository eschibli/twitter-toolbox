import csv
import datetime
import json
import os
import time
import tweepy


# Initializing a listener class that streams from  Twitter (Inherits from StreamListener of the tweepy library )
class TwitterStreamListener(tweepy.StreamListener):

    def __init__(self, consumer_key, consumer_secret, access_token, access_secret):
        super(TwitterStreamListener, self).__init__()
        # Initialize the customization attributes that we want to be able to change from the outside
        self.num_tweets = 0
        self.tweet_limit = 100
        self.file_label = "streamed_tweets"
        self.notification_interval = 1000
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
        # Errors dictionary
        self.ht_err = {0: 'Python version 3.0.0 or higher is required...',
                       1: 'Failed to load Tweepy...',
                       2: 'smtplib library not installed...',
                       3: 'csv library not installed...',
                       4: 'Unable to retrieve Twitter credentials... ',
                       5: 'Unable to establish connection. Make sure your credentials are correct '
                          '(Try re-generating your tokens).',
                       6: 'Unable to retrieve keywords list...',
                       7: 'Keyboard interrupt, stopping stream...',
                       8: 'Unable to re-establish connection...'}

        # Set listener parameters

    def c_exit(self, err_id):
        print("ERROR: " + self.ht_err[err_id])
        print("Closing Stream...")
        time.sleep(1)
        exit(1)

    # Outside caller will call this with their credentials and query
    def collect_from_stream(self, max_tweets, output_json_name, console_interval, target_words):
        self.initialize_listener(max_tweets, output_json_name, console_interval)
        self.begin_streaming_loop(self.auth, target_words)

    def begin_streaming_loop(self, auth_credentials, keywords):
        """
        # Stream from twitter and collect data
        # If connection is lost will wait appropriate amount of time and attempt to reconnect
        :param auth_credentials:
        :param keywords:
        :return:
        """

        last_connection_attempt = time.time()
        connections_in_last_15_minutes = 1
        # Maximum connections: 3 in 15 minutes
        while True:
            # If it has been 15 minutes since last connection request, reset the timer
            if time.time() - last_connection_attempt > 900:
                connections_in_last_15_minutes = 1
                last_connection_attempt = time.time()

            if connections_in_last_15_minutes > 3:
                reattempt_time = datetime.datetime.now() + datetime.timedelta(seconds=900)
                print(
                    "Max connection attempts in 15 minutes exceeded. Will attempt to reconnect when 15 minute window "
                    "ends at: ", str(reattempt_time))
                time.sleep(900)

            print("\nConnecting listener to stream...")
            time.sleep(0.5)
            stream = tweepy.Stream(auth_credentials, self)
            try:
                print("\n\tStreaming now...")
                stream.filter(track=keywords, languages=["en"])
                return
            # If keyboard exit interrupt
            except KeyboardInterrupt as e:
                self.c_exit(7)
            except:
                print("\n\tListening function: Termination Caused by Unknown Error...")
                print("\tWill attempt restart after ", str(30 * connections_in_last_15_minutes), " seconds")
                for s in range(30 * connections_in_last_15_minutes):
                    print("\t" + "\r" + str(s), '')
                    time.sleep(1)

            stream.disconnect()

            connections_in_last_15_minutes += 1

        return

    def initialize_listener(self, max_tweets, file_label, console_interval):
        print("\nInitializing listener... ")
        self.set_tweet_limit(max_tweets)
        self.set_file_label(file_label)
        self.set_console_notification_interval(console_interval)

    def set_tweet_limit(self, tweet_quantity):
        self.tweet_limit = tweet_quantity

    def set_file_label(self, file_label):
        self.file_label = file_label

    def set_console_notification_interval(self, console_notification_interval):
        self.notification_interval = console_notification_interval

    def on_data(self, data):
        # Based on the interval chosen by the user, let the user know
        # how many tweets have been collected so far
        if self.num_tweets % self.notification_interval == 0:
            print("\t" + str(self.num_tweets) + " tweets collected...", end="\r")

        # Increment the number of tweets collected by the number of tweets received
        self.num_tweets += 1

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
            self.num_tweets = 0
            return False

    def on_error(self, status):
        print(status)

import tweepy
import csv

class bulk_downloader:
    def __init__(self,consumer_key,consumer_secret,access_key,access_secret):
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_key = access_key
        self.access_secret = access_secret

    def get_tweets_csv_for_this_user(self,screen_name,output_file_name):
        #Script adapted from: https://gist.github.com/yanofsky/5436496
        # Only the most recent 3200 tweets by this user will be collected due to API limits imposed by twitter
        
        #authorize twitter, initialize tweepy
        auth = tweepy.OAuthHandler(self.consumer_key,self.consumer_secret)
        auth.set_access_token(self.access_key,self.access_secret)
        api = tweepy.API(auth)
        
        #initialize a list to hold all the tweepy Tweets
        alltweets = []    
        
        #make initial request for most recent tweets (200 is the maximum allowed count)
        new_tweets = api.user_timeline(screen_name = screen_name,count=200)
        
        #save most recent tweets
        alltweets.extend(new_tweets)
        
        #save the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1
       
        #transform the tweepy tweets into a 2D array that will populate the csv    
        outtweets = [[tweet.id_str, tweet.created_at,str(tweet.text.encode("utf-8").decode('utf8')),tweet.retweet_count,tweet.favorite_count,tweet.in_reply_to_status_id_str,tweet.in_reply_to_screen_name,screen_name] for tweet in alltweets]
        
        #keep grabbing tweets until there are no tweets left to grab
        while len(new_tweets) > 0:
            #all subsiquent requests use the max_id param to prevent duplicates
            new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
            
            #save most recent tweets
            alltweets.extend(new_tweets)
            
            #update the id of the oldest tweet less one
            oldest = alltweets[-1].id - 1
        
        #write the csv    
        with open(output_file_name, 'a',encoding='utf-8',newline='') as f:
            writer = csv.writer(f)
            writer.writerows(outtweets)
        
        pass
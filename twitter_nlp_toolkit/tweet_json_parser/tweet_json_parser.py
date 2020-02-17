# twitter sends its data as a json
import json
# we will convert the json data into a csv
import csv
# numpy's np.nan helps us deal with missing values (and there is a TON of missing values in tweets) 
import numpy as np
# the fix_encoding function from the FTFY package takes care of weird characters in tweets
import ftfy

class json_parser:
    
                 
    def stream_json_file(self,json_file_name ,output_file_name = "parsed_tweets.csv", stop_at = 1000000000,verbose=False):
        # Create an empty csv file that we will append to
        # Create a header row for it
        if verbose:
            print("Initalizing Output File:  %s"%output_file_name)
            print("Generating Header Row")
        with open('%s'%output_file_name,'w') as f:
            f.write('id,text,created_at,language,retweet_count,screen_name,country,user_followers_count,time_zone,user_account_location,longitude,lattitude,name\n') #  Column headers and a trailing new line . MAKE SURE the /n is attached to the last field: eg. text/n
        
        
        tweet_counter = 0
        
        for i in open(json_file_name): 
            
            if tweet_counter > stop_at:
                break
                
            
                
            try:
                # Put the data from the current tweet into a list
                # Parse the current tweet
                current_tweet = json.loads(i)
                
                ##################################################################
                
                ## Elements that are 1 level deep ## ##
                
                # Get the id or insert a nan if not found
                if 'id' in current_tweet:
                    id = current_tweet['id']
                else:
                    id= np.nan
                 
                # Get text or insert nan
                
                if 'text' in current_tweet:
                    text = current_tweet['text']
                    
                    # the fix_encoding function from the FTFY package takes care of weird characters
                    text = ftfy.fix_encoding(text)
                else:
                    text = np.nan
                    
                    
                # Get created_at or insert nan
                
                if 'created_at' in current_tweet:
                    created_at = current_tweet['created_at']
                    
                else:
                    created_at = np.nan
                
                # Get language or insert nan
               
                if 'lang' in current_tweet:
                    language = current_tweet['lang']
         
                else:
                    language = np.nan     
                
                
                
                # get retweet count or insert nan
                    
                if 'retweet_count' in current_tweet:
                    retweet_count = current_tweet['retweet_count']
                else:
                    retweet_count = np.nan
                    
                ## Elements that are 2 levels deep ### ###
                
                # For elements that are 2 layers deep use != None when searching because javascript uses None as its null operator    
                
                # get screen name or insert nan
                
                if 'user' in current_tweet and 'screen_name' in current_tweet['user']:
                    screen_name = current_tweet['user']['screen_name']
                else:
                    screen_name = np.nan
                
                # get name or insert nan
                if 'user' in current_tweet and 'name' in current_tweet['user']:
                    name = current_tweet['user']['name']
                else:
                    name = np.nan    
                    
                    
                # get country or insert nan

                if current_tweet['place'] != None and current_tweet['place']['country'] != None:
                    country = current_tweet['place']['country']
                else:
                    country = np.nan
                
                # get the author's follower count or nan

                if current_tweet['user'] != None and current_tweet['user']['followers_count'] != None:
                    followers_count = current_tweet['user']['followers_count']
                else:
                    followers_count = np.nan
                
                
                # get the timezone or nan
                if current_tweet['user'] != None and current_tweet['user']['time_zone'] != None:
                    time_zone = current_tweet['user']['time_zone']
                else:
                    time_zone = np.nan
                    
                # get the account location or insert nan
                
                if current_tweet['user'] != None and current_tweet['user']['location'] != None:
                    account_location = current_tweet['user']['location']
                    account_location = ftfy.fix_encoding(account_location)
                else:
                    account_location = np.nan
                    
                ###### Elements that are 3 levels deep ##################################
                
                if current_tweet['coordinates'] != None and current_tweet['coordinates']['coordinates'] != None and len(current_tweet['coordinates']['coordinates'])==2:
                    longitude = current_tweet['coordinates']['coordinates'][0]
                else:
                    longitude = np.nan
                    
                if current_tweet['coordinates'] != None and current_tweet['coordinates']['coordinates'] != None and len(current_tweet['coordinates']['coordinates'])==2:
                    lattitude = current_tweet['coordinates']['coordinates'][1]
                else:
                    lattitude = np.nan
            
                ######################################################################################################
                # Assemble the row
                cleaned_current_tweet = [id,text,created_at,language, retweet_count, screen_name, country, followers_count,time_zone,account_location,longitude,lattitude,name]
            
                # Increment the Tweet Counter
                tweet_counter = tweet_counter + 1
                
                # Give the user a progress update
                if tweet_counter % 1000 == 0 and verbose:
                    print(" %d Tweets Parsed so far....." %tweet_counter)
                
                #append the current tweet as a row to the csv
                with open('%s'%output_file_name,'a',newline='') as f:
                    writer=csv.writer(f)
                    writer.writerow(cleaned_current_tweet)
                
            except:
                pass
        if verbose:
            print(" ")
            print(" Parsing Complete:    %d Tweets Parsed " %tweet_counter)
import datetime
import pandas as pd
import json
import csv

"""Author: Taylor Robbins
this is a class to help with framing data in the correct way for classification of fake or bot twitter accounts"""


# Class to hold DataFrame making functions

class Framer:

    def __init__(self):
        pass

    # function to get training data from Cambridge csv and return DataFrame, trimming it to fit features I've selected
    def get_from_csv(self, csv_file):
        train_frame = pd.read_csv(csv_file)
        tf = train_frame.drop(['user_replied', 'tweet_frequency', 'favourite_tweet_ratio', 'sources_count', 'urls_count',
                               'cdn_content_in_kb', 'source_identity'], axis=1)
        tf['tweet_count'] = tf['user_tweeted'] + tf['user_retweeted']
        tf = tf.drop(['user_tweeted', 'user_retweeted'], axis=1)
        return tf

    # function to get testing data from real data json and return DataFrame
    def get_from_json(self, json_file, bots=None, count=0):

        # read tweet objects from json into list of dicts
        tweets = []
        for line in open(json_file, 'r', encoding='utf-8'):
            tweets.append(json.loads(line))

        # remove extra dict objects with info key
        for tweet in tweets:
            if tweet.get('info'):
                tweets.remove(tweet)

        # extract features, not using all of them at first
        tweet_list = []
        if bots is None:
            if count == 0:
                for tweet in tweets:
                    user_name = tweet['user']['name']
                    screen_name = tweet['user']['screen_name']
                    profile_age = datetime.datetime.strptime(tweet['user']['created_at'], "%a %b %d %H:%M:%S +0000 %Y")
                    tweet_age = datetime.datetime.strptime(tweet['created_at'], "%a %b %d %H:%M:%S +0000 %Y")
                    age = tweet_age - profile_age
                    followers = tweet['user']['followers_count']
                    following = tweet['user']['friends_count']
                    if following != 0:
                        ratio = followers/following
                    else:
                        ratio = -1
                    user_lists = tweet['user']['listed_count']
                    user_favorites = tweet['user']['favourites_count']
                    tweet_count = tweet['user']['statuses_count']
                    tweet_replies = tweet['reply_count']
                    tweet_favorites = tweet['favorite_count']
                    retweets = tweet['retweet_count']
                    protected = tweet['user']['protected']
                    verified = tweet['user']['verified']

                    tweet_dict = {}
                    tweet_dict.update({
                                       'Favorites per Tweet': tweet_favorites, 'Follow Ratio': ratio, 'Profile Age': age.days,
                                       'Retweets per Tweet': retweets, 'Screen Name': screen_name,
                                       'Tweet Count': tweet_count, 'User Favorites': user_favorites, 'User Lists': user_lists,})

                    tweet_list.append(tweet_dict)

        else:
            for tweet in tweets:
                user_name = tweet['user']['name']
                screen_name = tweet['user']['screen_name']
                profile_age = datetime.datetime.strptime(tweet['user']['created_at'], "%a %b %d %H:%M:%S +0000 %Y")
                tweet_age = datetime.datetime.strptime(tweet['created_at'], "%a %b %d %H:%M:%S +0000 %Y")
                age = tweet_age - profile_age
                followers = tweet['user']['followers_count']
                following = tweet['user']['friends_count']
                if following != 0:
                    ratio = followers/following
                else:
                    ratio = -1
                user_lists = tweet['user']['listed_count']
                user_favorites = tweet['user']['favourites_count']
                tweet_count = tweet['user']['statuses_count']
                tweet_replies = tweet['reply_count']
                tweet_favorites = tweet['favorite_count']
                retweets = tweet['retweet_count']
                protected = tweet['user']['protected']
                verified = tweet['user']['verified']

                if screen_name in bots:
                    tweet_dict = {}
                    tweet_dict.update({
                                       'Favorites per Tweet': tweet_favorites, 'Follow Ratio': ratio, 'Profile Age': age.days,
                                       'Retweets per Tweet': retweets, 'Screen Name': screen_name,
                                       'Tweet Count': tweet_count, 'User Favorites': user_favorites, 'User Lists': user_lists,})
                    tweet_list.append(tweet_dict)
                else:
                    continue

        # make dataframe from features
        return pd.DataFrame(tweet_list)

    # adds labels as a new column for a dataframe of either bots or humans
    # for the training data from cambridge,
    # if human, bot = 0, else bot
    def make_labels(self, data, bot=False):
        labels = []
        match_list = data['Screen Name']
        if bot:
            for name in match_list:
                labels.append(1)
        else:
            for name in match_list:
                labels.append(0)
        data['bot'] = labels
        return data

    # finds accounts with suspicious follow ratio and age:tweet ratio for further inspection
    # this will help in the process of labeling the new data by hand
    def generate_bots(self, real_data):
        possible_bots = []
        first = True

        with open(real_data, 'r') as csvf:
            reader = csv.reader(csvf)
            for row in reader:
                if first:
                    first = False
                    continue
                if float(row[2]) < 0.5 and float(row[3]) < 100.0:
                    possible_bots.append(row)
        return possible_bots

    # iterate through a list of possible bots and remove duplicate entries, which are just tweets by the same account
    def get_unique_bots(self, possibles):
        uniques = []
        names = []

        with open(possibles, 'r') as csvf:
            reader = csv.reader(csvf)
            first = True

            for row in reader:
                if first:
                    uniques.append(row)
                    names.append(row[5])
                    first = False
                else:
                    if row[5] in names:
                        continue
                    else:
                        uniques.append(row)
                        names.append(row[5])
        return uniques

    # takes two dataframes of humans and bots, and combines them to simulate fake account percentage
    # outside of the #metoo movement, randomized for training
    def human_bot_sim(self, humans, bots):
        new_df = pd.concat([humans, bots])
        return new_df.sample(frac=1)


f = Framer()














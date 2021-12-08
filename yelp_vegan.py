from bs4 import BeautifulSoup
import requests
from lxml import html
import argparse
import json
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
import re, string, random

def take_user_input():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('place', help='Location')
    search_query_help = """Available search queries are:\n
                            Restaurants,\n
                            Vegan"""
    argparser.add_argument('search_query', help=search_query_help)
    args = argparser.parse_args()
    place = args.place
    search_query = args.search_query
    return search_query, place

def scrape_yelp_business(yelp_url):
    soup=BeautifulSoup(requests.get(yelp_url).content,'lxml')
    restaurantReviews = {}
    for item in soup.select('[class*=container]'):
        try:
            if item.find('h4'):
                name = item.find('h4').get_text()
                split_title = name.split('\xa0')
                if len(split_title) > 1:
                    restaurant_name = split_title[1]
                    url = soup.find_all('a', {'name': restaurant_name, 'href': True})
                    restaurant_url = "https://www.yelp.com" + url[0]['href']

                    #Beautiful Soup for scraping yelp reviews given business's yelp url
                    restaurantReviews[restaurant_name] = scrape_yelp_reviews(restaurant_url)
        except Exception as e:
            raise e
            print('no search results for your keyword and location')
    return restaurantReviews


def scrape_yelp_reviews(restaurant_url):
    soup = BeautifulSoup(requests.get(restaurant_url).content, "html.parser")
    reviews = soup.find_all('span', {'class': "raw__09f24__T4Ezm"})
    start_review = False
    reviews_lst = []
    for review in reviews:
        current_review = review.get_text()
        if ("Start your review of" in current_review):
            start_review = True
            continue
        if (start_review):
            reviews_lst.append(current_review)
        else:
            continue
    return reviews_lst

def doc_selection(query, restaurantReviews):
    improvedRestaurantReviews = {}
    for restaurant,reviews in restaurantReviews.items():
        selected_reviews = []
        for review in reviews:
            if query in review.lower():
                selected_reviews.append(review)
        improvedRestaurantReviews[restaurant] = selected_reviews
    return improvedRestaurantReviews

def get_training_data():
    positive_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tokens = twitter_samples.tokenized('negative_tweets.json')

    cleaned_positive_tokens = []
    for tokens in positive_tokens:
        cleaned_lst = []
        for token in tokens:
            if token.lower() not in stopwords.words('english'):
                cleaned_lst.append(token.lower())
        cleaned_positive_tokens.append(cleaned_lst)

    cleaned_negative_tokens = []
    for tokens in negative_tokens:
        cleaned_lst = []
        for token in tokens:
            if token.lower() not in stopwords.words('english'):
                cleaned_lst.append(token.lower())
        cleaned_negative_tokens.append(cleaned_lst)

    model_positive = []
    for tweet_tokens in cleaned_positive_tokens:
        model_positive.append(dict([token, True] for token in tweet_tokens))

    model_negative = []
    for tweet_tokens in cleaned_negative_tokens:
        model_negative.append(dict([token, True] for token in tweet_tokens))

    positive_dataset = []
    for dict_item in model_positive:
         positive_dataset.append((dict_item, "Positive"))

    negative_dataset = []
    for dict_item in model_negative:
         negative_dataset.append((dict_item, "Negative"))
    return (positive_dataset + negative_dataset)

def get_most_positively_reviewed_business(restaurantReviews, model):
    max_pos_count = 0
    best_restaurant = ""
    for restaurant,reviews in restaurantReviews.items():
        curr_pos_count = 0
        for review in reviews:
            review_tokens = []
            for token in word_tokenize(review):
                token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                               '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
                if token.lower() not in stopwords.words('english'):
                    review_tokens.append(token.lower())

            if model.classify(dict([token, True] for token in review_tokens)) == "Positive":
                curr_pos_count+=1

        if curr_pos_count > max_pos_count:
            max_pos_count = curr_pos_count
            best_restaurant = restaurant
    return best_restaurant

if __name__ == "__main__":

    #User input
    keyword, location = take_user_input()
    yelp_url = "https://www.yelp.com/search?find_desc=%s&find_loc=%s" % (keyword,location)

    #generate a dictionary of yelp's search result's restaurants and its reviews
    restaurantReviews = scrape_yelp_business(yelp_url)

    #Document Selection (relevant reviews have "vegan" in them)
    restaurantReviews = doc_selection("vegan", restaurantReviews)

    #extract and pre-process twitter data
    dataset = get_training_data()

    #train the Sentiment Analysis Model with cleaned training data
    model = NaiveBayesClassifier.train(dataset[:8000])

    #Script output
    best_vegan_restaurant = get_most_positively_reviewed_business(restaurantReviews, model)
    print("The best " + keyword + " restaurant in " + location + " is " + best_vegan_restaurant + "!")

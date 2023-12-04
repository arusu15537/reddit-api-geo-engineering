import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer


# Download the VADER lexicon used by SentimentIntensityAnalyzer
# nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
csv_file_path = 'reddit_geoengineering_posts.csv'

df = pd.read_csv(csv_file_path, usecols=['ID', 'Title', 'Text'])


def detect_negative_opinions(text):
    sentiment_scores = sid.polarity_scores(text)

    # The 'compound' score is a normalized combination of positive, negative, and neutral scores
    compound_score = sentiment_scores['compound']

    return compound_score

def detect_negative_opinions_binary(text):
    sentiment_scores = sid.polarity_scores(text)

    # The 'compound' score is a normalized combination of positive, negative, and neutral scores
    compound_score = sentiment_scores['compound']

    # Define a threshold for negativity (you can adjust this based on your needs)
    negativity_threshold = -0.5

    return compound_score < negativity_threshold


# Processing the opinions of text
for index, row in df.iterrows():
    print('Processing datapoint: ' + str(index))
    detect_negative_opinions(str(row['Text']))


# Example usage:
reddit_comment = \
    "Until Citizens United is overturned, this is what we are relegated to."
is_negative = detect_negative_opinions_binary(reddit_comment)

print(is_negative)

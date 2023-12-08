import nltk
import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('wordnet')


sid = SentimentIntensityAnalyzer()
csv_file_path = 'reddit_geoengineering_posts.csv'

df = pd.read_csv(csv_file_path, usecols=['ID', 'Title', 'Text', 'Topic'])


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


def getSentimentScores(binary: bool, dataframe: pd.DataFrame):
    sentiments = []
    for index, row in dataframe.iterrows():
        print('Processing datapoint: ' + str(index))
        if binary:
            sentiments.append(detect_negative_opinions_binary(str(row['Text'])))
        else:
            sentiments.append(detect_negative_opinions(str(row['Text'])))
    return sentiments


def list_to_upper(lst):
    return [term.upper() for term in lst]


carbon_capture_terms = list_to_upper(['direct air capture', 'co2 removal', 'carbon capture', 'carbon capture and storage'])



def filter_dataframe(dataframe, filter_list):
    returned_df = pd.DataFrame(columns=dataframe.columns)
    print(returned_df)
    accept_comments = False
    for index, row in dataframe.iterrows():
        if str(row['Topic']) in filter_list:
            returned_df = pd.concat([returned_df, pd.DataFrame([row])], ignore_index=True)
            accept_comments = True
        elif row['Topic'].__eq__('COMMENT') and accept_comments:
            returned_df = pd.concat([returned_df, pd.DataFrame([row])], ignore_index=True)
        else:
            accept_comments = False
    return returned_df


df_cc = filter_dataframe(df, carbon_capture_terms)
srm_terms = ['solar radiation management', 'stratospheric aerosol injection', 'aerosol injection']
general_terms = ['geoengineering', 'geo-engineering']
srm_terms = list_to_upper(srm_terms)
df_srm = filter_dataframe(df, srm_terms)

df_general = filter_dataframe(df, list_to_upper(general_terms))


print(df_cc)
sentiment_scores = getSentimentScores(False, df)

# Remove neutral opinions (0)
sentiment_scores = [x for x in sentiment_scores if x != 0]

print(sentiment_scores)
sentiment_statistics = {}

def perform_sentiment_analysis():
    mean_sentiment = np.mean(sentiment_scores)
    median_sentiment = np.median(sentiment_scores)
    std_dev_sentiment = np.std(sentiment_scores)
    percentile_25 = np.percentile(sentiment_scores, 25)
    percentile_75 = np.percentile(sentiment_scores, 75)
    statistics = {
        'Mean': mean_sentiment,
        'Median': median_sentiment,
        'Standard Deviation': std_dev_sentiment,
        '25th Percentile': percentile_25,
        '75th Percentile': percentile_75
    }
    return statistics


def print_sentiment_analysis():
    for stat, value in sentiment_statistics.items():
        print(f"{stat}: {value}")


# Store statistics in a dictionary



bin_edges = [i / 100 for i in range(-100, 101, 5)]
cmap = LinearSegmentedColormap.from_list(
    'RedToGreen', ['#FF0000', '#00FF00'], N=len(bin_edges))
plt.figure(figsize=(10, 6))
hist, edges, _ = plt.hist(sentiment_scores, bins=bin_edges, edgecolor='black', color='blue', alpha=0.7, linewidth=1.2)

fig, ax = plt.subplots(figsize=(10, 6))


ax.set_xlabel('Sentiment Score')
ax.set_ylabel('Frequency')
ax.set_title('Sentiment Analysis Distribution')

n_bins = len(bin_edges) - 1

for i in range(n_bins):
    plt.bar(bin_edges[i], hist[i], width=bin_edges[i + 1] - bin_edges[i],
            color=cmap(i / n_bins), edgecolor='black')

for i, freq in enumerate(hist):
    plt.text(edges[i] + 0.01, freq + 1, str(int(freq)), color='black', fontweight='bold')

ax.grid(axis='y', linestyle='--', alpha=0.7)

sm = plt.cm.ScalarMappable(cmap=cmap)
sm.set_array([])
plt.xticks(np.arange(-1, 1.05, 0.05))

plt.show()


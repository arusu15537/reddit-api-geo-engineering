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

df = pd.read_csv(csv_file_path, usecols=['ID', 'Title', 'Text', 'Topic', 'Number of Comments', 'Score'])
ranges = [-1.0, -0.95, -0.9, -0.85, -0.8, -0.75, -0.7, -0.65, -0.6, -0.55, -0.5, -0.45, -0.4, -0.35, -0.3, -0.25, -0.2,
          -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
          0.8, 0.85, 0.9, 0.95, 1.0]


def detect_negative_opinions(text):
    sentiment_scores = sid.polarity_scores(text)
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


def get_sentiment_scores(binary: bool, dataframe: pd.DataFrame, score_metric: bool):
    sentiments = {}
    for index, row in dataframe.iterrows():
        print('Processing datapoint: ' + str(index))
        if binary:
            sentiment = detect_negative_opinions_binary(str(row['Text']))
            if sentiment != 0:
                sentiment = min(ranges, key=lambda x: abs(x - sentiment))
                if sentiment in sentiments:
                    if score_metric:
                        sentiments[sentiment] += row['Number of Comments'] + row['Score']
                    else:
                        sentiment[sentiment] += 1
                else:
                    if score_metric:
                        sentiments[sentiment] = row['Number of Comments'] + row['Score']
                    else:
                        sentiments[sentiment] = 1
        else:
            if str(row['Text']).__eq__('') and not str(row['Topic']).__eq__('COMMENT'):
                sentiment = detect_negative_opinions(str(row['Title']))
            else:
                sentiment = detect_negative_opinions(str(row['Text']))

            if sentiment != 0:
                sentiment = min(ranges, key=lambda x: abs(x - sentiment))
                if sentiment in sentiments:
                    if score_metric:
                        sentiments[sentiment] += row['Number of Comments'] + row['Score']
                    else:
                        sentiments[sentiment] += 1
                else:
                    if score_metric:
                        sentiments[sentiment] = row['Number of Comments'] + row['Score']
                    else:
                        sentiments[sentiment] = 1
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



# Remove neutral opinions (0)

sentiment_statistics = {}


def perform_sentiment_analysis(score_metric: bool):
    sentiment_scores = get_sentiment_scores(False, df, score_metric)
    sentiments = np.array(list(sentiment_scores.keys()))
    frequencies = np.array(list(sentiment_scores.values()))
    mean_sentiment = np.average(sentiments, weights=frequencies)
    std_dev_sentiment= np.sqrt(np.average((sentiments - np.average(sentiments, weights=frequencies)) ** 2,
                                      weights=frequencies))
    percentile_25 = np.percentile(np.repeat(sentiments, frequencies), 25)
    percentile_75 = np.percentile(np.repeat(sentiments, frequencies), 75)

    statistics = {
        'Mean': mean_sentiment,
        'Standard Deviation': std_dev_sentiment,
        '25th Percentile': percentile_25,
        '75th Percentile': percentile_75
    }
    return statistics, sentiment_scores


def print_sentiment_analysis():
    for stat, value in sentiment_statistics.items():
        print(f"{stat}: {value}")


def create_bins(data_dict, bin_size):
    # Extract keys and values from the dictionary
    keys = list(data_dict.keys())
    values = list(data_dict.values())

    # Define the range of values (-1 to 1)
    value_range = np.arange(-1, 1.05, bin_size)

    # Calculate the bin edges
    bin_edges = value_range - bin_size / 2
    bin_edges = np.append(bin_edges, value_range[-1] + bin_size / 2)

    return keys, values, bin_edges


def map_sentiments_to_histogram(score_metric: bool, figure):
    statistics, sentiment_scores = perform_sentiment_analysis(score_metric)
    print(statistics)
    keys = list(sentiment_scores.keys())
    values = list(sentiment_scores.values())
    plt.figure(figure)
    bin_width = 0.05
    bin_edges = np.arange(min(keys), max(keys) + bin_width, bin_width)
    hist, edges, _ = plt.hist(keys, bins=bin_edges, weights=values, edgecolor='black', alpha=0.7, linewidth=1.2)
    n_bins = len(bin_edges) - 1
    cmap = LinearSegmentedColormap.from_list(
        'RedToGreen', ['#FF0000', '#00FF00'], N=len(bin_edges))
    for i in range(n_bins):
        plt.bar(bin_edges[i], hist[i], width=bin_edges[i + 1] - bin_edges[i],
                color=cmap(i / n_bins), edgecolor='black', align='edge')
    plt.xlabel('Opinion')
    if score_metric:
        plt.ylabel('Opinion score')
    else:
        plt.ylabel('Opinion frequency')
    plt.show()



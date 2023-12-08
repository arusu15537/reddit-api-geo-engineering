import praw
from praw.models import MoreComments
import csv
import pandas as pd
import sentiment_analysis

data_points = 0

client_id = 'YYr-1yy7SSBNFKDfRBLvnA'
client_secret = '-i0blTrr4eoGSi_H8GfGAbb-AiaFPQ'
user_agent = 'rusuMercedesBenz'

broad_search_terms = ['geoengineering', 'geo-engineering', 'solar radiation management', 'carbon capture',
                      'carbon capture and storage', 'stratospheric aerosol injection',
                      'direct air capture', 'co2 removal', 'aerosol injection']
reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
csv_data = [['ID', 'Title', 'Author', 'Score', 'Number of Comments', 'Text', 'Topic']]


def get_comments(comments, passed_csv, data_point=data_points):
    index = 0
    for comment in comments:
        if not isinstance(comment, MoreComments) and not comment.is_submitter and not comment.author.__eq__('') \
                and sentiment_analysis.detect_negative_opinions(comment.body) != 0:
            print(sentiment_analysis.detect_negative_opinions(comment.body))
            passed_csv.append([comment.id, comment.parent_id, comment.author, comment.score, comment.replies.__len__(),
                               comment.body, 'COMMENT'])
            index += 1
            data_point += 1
            print(str(data_point) + ' C')

        if index > 9:
            break


def search_reddit(title_includes, data_point=data_points):
    search_results = reddit.subreddit("all").search(title_includes, sort='relevance', time_filter='all', limit=50)
    for submission in search_results:
        csv_data.append([submission.id, submission.title, submission.author.name, submission.score,
                         submission.num_comments, submission.selftext, title_includes.upper()])
        get_comments(submission.comments.list(), csv_data)
        data_point += 1


for topic in broad_search_terms:
    search_reddit(topic)

csv_file_path = 'reddit_geoengineering_posts.csv'
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(csv_data)

df = pd.read_csv(csv_file_path)

df_no_duplicates = df.drop_duplicates(subset='ID')
df_no_duplicates = df_no_duplicates.dropna(subset=['Topic'])
df_no_duplicates.to_csv(csv_file_path, index=False, encoding='utf-8')

print(f"Search results for '{broad_search_terms}' saved to {csv_file_path}")

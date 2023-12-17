from typing import List, Tuple

import nltk
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sentiment_analysis
from matplotlib.colors import LinearSegmentedColormap
from nltk import TreebankWordTokenizer
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from gensim import corpora, models
from nltk.stem import WordNetLemmatizer


tokenizer = TreebankWordTokenizer()
G = nx.Graph()


def prepare_file_for_processing():
    csv_file_path = 'reddit_geoengineering_posts.csv'
    nltk.download('averaged_perceptron_tagger')
    df_read = pd.read_csv(csv_file_path)
    return df_read


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


def list_to_upper(lst):
    return [term.upper() for term in lst]


carbon_capture_terms = list_to_upper(
    ['direct air capture', 'co2 removal', 'carbon capture', 'carbon capture and storage'])
srm_terms = list_to_upper(['solar radiation management', 'stratospheric aerosol injection'])
all_search_terms = ['geoengineering', 'geo-engineering', 'solar radiation management', 'carbon capture',
                      'carbon capture and storage', 'stratospheric aerosol injection',
                      'direct air capture', 'co2 removal', 'aerosol injection']

def preprocess_scores(dataframe, list_of_term_lists):
    df_array = [dataframe]
    impacts = []
    for ls in list_of_term_lists:
        df_array.append(filter_dataframe(df_array[0], ls))
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

    for df in df_array:
        if not df.empty:
            grouped = df.groupby('Author')
            summed_score = grouped[['Score', 'Number of Comments']].sum()
            stop_words = set(stopwords.words('english'))
            ps = PorterStemmer()
            lemmatizer = WordNetLemmatizer()
            node_sizes_scores = summed_score['Score'].tolist()
            node_sizes_comments = summed_score['Number of Comments'].tolist()
            max_comments = max(node_sizes_comments)
            max_scores = max(node_sizes_scores)
            min_scores = min(node_sizes_scores)
            min_comments = min(node_sizes_comments)
            max_impact = max_comments + max_scores
            min_impact = min_comments + min_scores
            impacts.append((min_impact, max_impact))
            df_sorted = df.sort_values('Score', ascending=False)
            df_sorted.to_csv('sorted_df.csv')
    return df_array, impacts


def min_max_approx(value, minim, maxim):
    value = (value - minim) / (maxim - minim)
    return value


def preprocess_text(text):
    words = tokenizer.tokenize(str(text))

    tags = nltk.pos_tag(words)
    adjectives = [w for w, t in tags if t == 'JJ']
    return adjectives

# def find_score(author_name):
#     return df_summed_score[df_summed_score['Author'] == row[name]]['Score'].values[0]
#                                        + df_summed_score[df_summed_score['Author'] == row['Author']][
#                                            'Number of Comments'].values[0]


def add_nodes_and_edges_to_graph(graph, df, impacts):
    for i in range(len(df)):
        if not df[i].empty:
            for index, row in df[i].iterrows():
                author = row['Author']
                grouped = df[i].groupby('Author')
                summed_score = grouped[['Score', 'Number of Comments']].sum()
                size = min_max_approx(summed_score.loc[author, 'Score']+summed_score.loc[author, 'Number of Comments'],
                                      impacts[i][0],
                                      impacts[i][1])
                color = sentiment_analysis.detect_negative_opinions(str(row['Text']))
                print(row['Text'])
                print(color)
                size = abs(size)*50000 + 200


                if row['Topic'].__eq__('COMMENT'):
                    destination = row['Title']
                    destination = destination[3:]
                    goes_to = df[i][df[i]['ID'] == destination]
                    if not goes_to.empty:
                        destination_row = goes_to.iloc[0]
                        destination = destination_row['Author']
                        graph.add_node(author, size=size, color=color)
                        graph.add_node(destination, size=size, color=color)
                        graph.add_edge(author, destination)
                else:
                    graph.add_node(author, size=size, color=color)


def draw_network_and_wordcloud(df):
    df['preprocessed_text'] = df['Text'].apply(preprocess_text)
    df_array, impacts = preprocess_scores(df, [all_search_terms])
    add_nodes_and_edges_to_graph(G, df_array, impacts)

    all_words = [word for words in df['preprocessed_text'] for word in words]

    # adjectives = [t in all_words if t == 'JJ']

    common_words = Counter(all_words).most_common(10)  # Adjust the number as needed

    print("Most Common Words:")
    print(common_words)

    # Create a dictionary from the preprocessed text
    dictionary = corpora.Dictionary(df['preprocessed_text'])

    # Create a document-term matrix
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in df['preprocessed_text']]

    # Train the LDA model
    lda_model = models.LdaModel(doc_term_matrix, num_topics=5, id2word=dictionary, passes=15)

    # Print the topics
    print("Topics:")
    topic: list[tuple[str, float]]
    for idx, topic in lda_model.print_topics():
        print(f"Topic {idx}: {topic}")

    text_data = ' '.join(df['preprocessed_text'].astype(str))

    # Generate a word cloud
    wordcloud = WordCloud(width=1600, height=800, background_color='white').generate(text_data)
    print('Number of nodes in graph')
    print(G.number_of_nodes())
    cmap = LinearSegmentedColormap.from_list('custom_colormap', ['red', 'yellow', 'green'], N=256)
    node_colors = [cmap(val) for val in nx.get_node_attributes(G, 'color').values()]

    node_sizes = [G.nodes[node]['size'] for node in G.nodes]
    n_bins = len(node_sizes)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    pos = nx.spring_layout(G)

    nx.draw_networkx(G, with_labels=True, node_color=node_colors, node_size=node_sizes, font_size=7, ax=ax1, pos=pos)

    ax1.set_title('Social Network Structure')

    ax2.imshow(wordcloud, interpolation='bilinear')
    ax2.axis('off')
    ax2.set_title('Word cloud')
    sentiment_analysis.map_sentiments_to_histogram(False, 3)
    plt.show()

def main():
    df = prepare_file_for_processing()
    draw_network_and_wordcloud(df)

if __name__ == '__main__':
    main()


#
#               __====-_  _-====___
#               _--^^^#####//      \#####^^^--_
#            _-^##########// (    ) \##########^-_
#           -############//  |\^^/|  \############-
#         _/############//   (@::@)   \############\_
#        /#############((     \\//     ))#############\
#       -###############\\    (oo)    //###############-
#      -#################\\  / " " \  //#################-
#     -###################\\/  (   )  //###################-
# _#/|##########/\######(   /(  .  )\   )######/\##########|\#_
#  |/  |/  |/  |/  |/  |/  |/  |/  |/  |/  |/  |/  |/  |/  |/  |/

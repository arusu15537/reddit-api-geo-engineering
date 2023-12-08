import nltk
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
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


csv_file_path = 'reddit_geoengineering_posts.csv'
nltk.download('averaged_perceptron_tagger')
tokenizer = TreebankWordTokenizer()

df = pd.read_csv(csv_file_path)
print(df)
print(type(df['Author'].loc[0]))
grouped = df.groupby('Author')
df_summed_score = grouped[['Score', 'Number of Comments']].sum()
print(df_summed_score)


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

df_cc = filter_dataframe(df, carbon_capture_terms)
df_srm = filter_dataframe(df, srm_terms)
df = df_srm

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

G = nx.Graph()

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

node_sizes_scores = df_summed_score['Score'].tolist()
node_sizes_comments = df_summed_score['Number of Comments'].tolist()

max_comments = max(node_sizes_comments)
max_scores = max(node_sizes_scores)

min_scores = min(node_sizes_scores)
min_comments = min(node_sizes_comments)

max_impact = max_comments + max_scores
min_impact = min_comments + min_scores


def min_max_approx(value, minim, maxim):
    value = (value - minim) / (maxim - minim)
    return round(max(1, value * 1000))


def preprocess_text(text):
    words = tokenizer.tokenize(str(text))

    tags = nltk.pos_tag(words)
    adjectives = [w for w, t in tags if t == 'JJ']
    return adjectives

# def find_score(author_name):
#     return df_summed_score[df_summed_score['Author'] == row[name]]['Score'].values[0]
#                                        + df_summed_score[df_summed_score['Author'] == row['Author']][
#                                            'Number of Comments'].values[0]


def add_nodes_and_edges_to_graph(graph):
    for index, row in df.iterrows():
        author = row['Author']
        size = min_max_approx(df_summed_score.loc[author,'Score']+df_summed_score.loc[author,'Number of Comments'], max_impact,
                              min_impact)

        if row['Topic'].__eq__('COMMENT'):
            destination = row['Title']
            destination = destination[3:]
            goes_to = df[df['ID'] == destination]
            if not goes_to.empty:
                destination_row = goes_to.iloc[0]
                destination = destination_row['Author']

                graph.add_node(author, size=size, color=size)
                graph.add_node(destination, size=size, color=size)
                graph.add_edge(author, destination)
        else:
            graph.add_node(author, size=size, color=size)


df['preprocessed_text'] = df['Text'].apply(preprocess_text)


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
for idx, topic in lda_model.print_topics():
    print(f"Topic {idx}: {topic}")

text_data = ' '.join(df['preprocessed_text'].astype(str))

# Generate a word cloud
wordcloud = WordCloud(width=1600, height=800, background_color='white').generate(text_data)
print('Number of nodes in graph')
add_nodes_and_edges_to_graph(G)
print(G.number_of_nodes())
node_colors = [G.nodes[node]['color'] for node in G.nodes()]

pos = nx.spring_layout(G)
node_sizes = [G.nodes[node]['size'] for node in G.nodes]
n_bins = len(node_sizes)

# Display the word cloud using matplotlib
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
nx.draw_networkx(G, with_labels=True, node_color=node_colors, node_size=node_sizes, font_size=7, ax=ax1, pos=pos)
ax1.set_title('Social Network Structure')

ax2.imshow(wordcloud, interpolation='bilinear')
ax2.axis('off')
ax2.set_title('Word cloud')
plt.tight_layout()
plt.show()

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

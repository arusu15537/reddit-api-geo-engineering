import nltk
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from gensim import corpora, models
from nltk.stem import WordNetLemmatizer

csv_file_path = 'reddit_geoengineering_posts.csv'

df = pd.read_csv(csv_file_path)

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

G = nx.Graph()

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

node_sizes_scores = df['Score'].tolist()
node_sizes_comments = df['Number of Comments'].tolist()

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
    words = word_tokenize(str(text))
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalpha() and word.lower() not in stop_words]
    return words


def add_nodes_and_edges_to_graph(graph):
    for index, row in df.iterrows():
        if row['Topic'].__eq__('COMMENT'):
            destination = row['Title']
            destination = destination[3:]
            goes_to = df[df['ID'] == destination]
            if not goes_to.empty:
                destination_row = goes_to.iloc[0]
                destination = destination_row['Author']
                graph.add_node(row['Author'], size=min_max_approx((row['Score'] + row['Number of Comments']), max_impact,
                                                              min_impact))
                graph.add_node(destination, size=min_max_approx((row['Score'] + row['Number of Comments']), max_impact,
                                                                min_impact))
                graph.add_edge(row['Author'], destination)
        else:
            graph.add_node(row['Author'], size=min_max_approx((row['Score'] + row['Number of Comments']), max_impact,
                                                          min_impact))


df['preprocessed_text'] = df['Text'].apply(preprocess_text)

all_words = [word for words in df['preprocessed_text'] for word in words]

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

pos = nx.spring_layout(G)
node_sizes = [G.nodes[node]['size'] for node in G.nodes]


# Display the word cloud using matplotlib
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
nx.draw_networkx(G, with_labels=True, node_size=node_sizes, font_size=10, ax=ax1, pos=pos)
ax1.set_title('Social Network Structure')

ax2.imshow(wordcloud, interpolation='bilinear')
ax2.axis('off')
ax2.set_title('Word cloud')
plt.tight_layout()
plt.show()

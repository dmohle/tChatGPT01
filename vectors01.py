import nltk
import string

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize



# Sample large text data
text = """
Put a long text here. You can paste multiple paragraphs, articles, or even books. Classification is probably the most popular task that you would deal with in real life. Text in the form of blogs, posts, articles, etc. is written every second. It is a challenge to predict the information about the writer without knowing about him/her. We are going to create a classifier that predicts multiple features of the author of a given text. We have designed it as a Multilabel classification problem. Dataset Blog Authorship Corpus
Over 600,000 posts from more than 19 thousand bloggers The Blog Authorship Corpus consists of the collected posts of 19,320 bloggers gathered from blogger.com in August 2004. The corpus incorporates a total of 681,288 posts and over 140 million words - or approximately 35 posts and 7250 words per person. Each blog is presented as a separate file, the name of which indicates a blogger id# and the blogger’s self-provided gender, age, industry, and astrological sign. (All are labeled for gender and age but for many, industry and/or sign is marked as unknown.)
All bloggers included in the corpus fall into one of three age groups: 8240 "10s" blogs (ages 13-17), 8086 "20s" blogs(ages 23-27) 2994 "30s" blogs (ages 33-47)
For each age group, there is an equal number of male and female bloggers. Each blog in the corpus includes at least 200 occurrences of common English words. All formatting has been stripped with two exceptions. Individual posts within a single blogger are separated by the date of the following post and links within a post are denoted by the label urllink.
An aspect list is for example, "pop, tinny wide hi hats, mellow piano melody, high pitched female vocal melody, sustained pulsating synth lead".
The caption consists of multiple sentences about the music, e.g., "A low sounding male voice is rapping over a fast paced drums playing a reggaeton beat along with a bass. Something like a guitar is playing the melody along. This recording is of poor audio-quality. In the background a laughter can be noticed. This song may be playing in a bar." The text is solely focused on describing how the music sounds, not the metadata like the artist name.
The labeled examples are 10s music clips from the AudioSet dataset (2,858 from the eval and 2,663 from the train split). See the paper and dataset for more information.
Please cite the corresponding paper, when using this dataset

"""

# Preprocess the data
def preprocess(text):
    # Tokenize and remove punctuation
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    return tokens

tokens = preprocess(text)
sentences = [tokens[i:i+5] for i in range(0, len(tokens), 5)]

# Train Word2Vec
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")

# Load model and get similar words
model = Word2Vec.load("word2vec.model")
similar_words = model.wv.most_similar("posts", topn=5)
print("\n\nSimilar words to 'posts' are:\n")
print(similar_words)


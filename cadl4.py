# CADL4: Topic Modeling with LDA

import gensim
from gensim import corpora
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already installed
nltk.download('stopwords')

# Sample text corpus (different text)
corpus = [
    "Python and JavaScript are widely used in web development.",
    "NASA is preparing for a new Moon mission in 2026.",
    "Cryptocurrency markets are highly volatile and unpredictable.",
    "Doctors recommend regular exercise for better mental health.",
    "Climate policies are shaping the future of renewable energy."
]

# Preprocessing
stop_words = stopwords.words('english')
texts = [
    [word.lower() for word in doc.split() if word.lower() not in stop_words]
    for doc in corpus
]

# Dictionary and Corpus
dictionary = corpora.Dictionary(texts)
doc_term_matrix = [dictionary.doc2bow(text) for text in texts]

# LDA Model
lda_model = gensim.models.LdaModel(
    doc_term_matrix, 
    num_topics=3, 
    id2word=dictionary, 
    passes=10
)

# Print topics
print("🔹 Topics Identified")
for idx, topic in lda_model.print_topics(num_words=5):
    print(f"Topic {idx}: {topic}")

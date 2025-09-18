# CADL2: Feature Extraction

# Check Python interpreter
import sys
print("🔹 Using Python from:", sys.executable)

# Import scikit-learn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Dataset (sample text corpus)
docs = [
    "Artificial intelligence is transforming the world.",
    "Climate change is one of the biggest global challenges.",
    "Fitness and healthy diet improve overall well-being.",
    "Space exploration opens new possibilities for humanity.",
    "Technology companies are investing in renewable energy."
]

# Bag of Words
print("\n🔹 Bag of Words Representation")
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(docs)
print("Vocabulary:", vectorizer.get_feature_names_out())
print("BoW Array:\n", X_bow.toarray())

# TF-IDF
print("\n🔹 TF-IDF Representation")
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(docs)
print("Vocabulary:", tfidf.get_feature_names_out())
print("TF-IDF Array:\n", X_tfidf.toarray())

import spacy
from nltk.stem import PorterStemmer
from nltk import download
from nltk.corpus import stopwords

# Download NLTK stopwords if not already downloaded
download('stopwords')

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# NLTK stemmer
stemmer = PorterStemmer()

# NLTK stopwords
stop_words = set(stopwords.words("english"))

# Sample text
text = "The quick brown foxes were jumping over the lazy dogs in the park."

# spaCy document object
doc = nlp(text)

print("Original Words | Lemma | Stem | Is Stopword")
print("-" * 50)

for token in doc:
    word = token.text
    if word.isalpha():  # Ignore punctuation/numbers
        lemma = token.lemma_
        stem = stemmer.stem(word.lower())
        is_stop = word.lower() in stop_words
        print(f"{word:<14} | {lemma:<10} | {stem:<10} | {is_stop}")

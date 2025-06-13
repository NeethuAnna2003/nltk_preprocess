import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample text
text = "US reaffirms support to India on terror, but its Military General says ties with Pakistan also vital to counter IS-KP"

# Sentence Tokenization
print("Sentence Tokenization:")
sentences = sent_tokenize(text)
print(sentences)

# Word Tokenization
print("\nWord Tokenization:")
words = word_tokenize(text)
print(words)

# Stop Word Removal
print("\nAfter Stop Word Removal:")
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.lower() not in stop_words and word.isalpha()]
print(filtered_words)

# Stemming
print("\nStemming:")
stemmer = PorterStemmer()
for word in filtered_words:
    print(f"{word} --> {stemmer.stem(word)}")

# Lemmatization
print("\nLemmatization:")
lemmatizer = WordNetLemmatizer()
for word in filtered_words:
    print(f"{word} --> {lemmatizer.lemmatize(word)}")
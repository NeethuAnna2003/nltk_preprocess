import spacy
import nltk
from nltk.stem import PorterStemmer

# Download required NLTK resources
nltk.download('punkt')

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Initialize NLTK stemmer
stemmer = PorterStemmer()

# Sample text
text = "US reaffirms support to India on terror, but its Military General says ties with Pakistan also vital to counter IS-KP"

# Tokenize and process text using spaCy
doc = nlp(text)

print("\n--- Preprocessing Steps ---")

for token in doc:
    if not token.is_stop and not token.is_punct and not token.is_space:
        print(f"Original: {token.text}")
        print(f" - Stemmed     : {stemmer.stem(token.text)}")
        print(f" - Lemmatized  : {token.lemma_}")
        print(f" - Is Stopword : {token.is_stop}")
        print()

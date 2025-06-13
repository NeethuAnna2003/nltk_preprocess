import spacy

# Load the English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

# Sample text
text = "Hello there! Welcome to NLP using spaCy. Let's tokenize this sentence."

# Process the text
doc = nlp(text)

# Word Tokenization
print("\nWord Tokenization:")
for token in doc:
    print(token.text)

# Sentence Tokenization
print("\nSentence Tokenization:")
for sent in doc.sents:
    print(sent.text)

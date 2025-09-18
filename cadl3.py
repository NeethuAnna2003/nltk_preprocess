# CADL3: Named Entity Recognition (NER)

import spacy
import pandas as pd

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Example dataset: news-style sentences
text = """
Elon Musk founded SpaceX in California in 2002.
Apple is opening a new office in Bangalore, India.
Barack Obama gave a speech in Berlin last summer.
Amazon invested $2 billion in renewable energy projects.
"""

# Process text
doc = nlp(text)

# Extract named entities
print("🔹 Named Entities")
for ent in doc.ents:
    print(ent.text, ent.label_)

# Structured info in a DataFrame
entities = [(ent.text, ent.label_) for ent in doc.ents]
df = pd.DataFrame(entities, columns=["Entity", "Type"])
print("\nStructured Info:\n", df)

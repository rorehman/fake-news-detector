# api/utils.py
import spacy

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text: str) -> str:
    """
    Lowercase, lemmatize, and remove stopwords/punctuation/whitespace.
    """
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]
    return ' '.join(tokens)

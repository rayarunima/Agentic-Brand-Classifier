import spacy
from spacy.cli import download
# Ensure curated transformer components are registered before loading model
import spacy_curated_transformers  # noqa: F401
import threading

# Upgrade to transformer-based model for better NER
MODEL_NAME = "en_core_web_trf"
_nlp_lock = threading.Lock()
_nlp_loaded = False

# Load SpaCy model once (thread-safe)
try:
    nlp = spacy.load(MODEL_NAME)
    _nlp_loaded = True
except OSError:
    with _nlp_lock:
        if not _nlp_loaded:
            download(MODEL_NAME)
            nlp = spacy.load(MODEL_NAME)
            _nlp_loaded = True

def extract_entities(sentence):
    """
    Extract entities from sentence using SpaCy.
    Thread-safe: uses shared nlp model instance.
    For batch processing, consider using nlp.pipe() instead.
    """
    # Use thread-safe processing (SpaCy models are thread-safe for inference)
    doc = nlp(sentence)
    result = {
        "tokens": [],
        "entities": [],
        "pos": {},
    }
    # POS tagging
    for token in doc:
        pos_tag = token.pos_
        if pos_tag not in result["pos"]:
            result["pos"][pos_tag] = []
        result["pos"][pos_tag].append(token.text)
        result["tokens"].append({
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "tag": token.tag_,
            "dep": token.dep_,
            "is_alpha": token.is_alpha,
            "is_stop": token.is_stop
        })
    # Named Entity Recognition
    for ent in doc.ents:
        result["entities"].append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        })
    return result

if __name__ == "__main__":
    sentence = input("Enter a sentence: ")
    output = extract_entities(sentence)
    import pprint
    pprint.pprint(output)

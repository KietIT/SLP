import re
import unicodedata

PUNCT_RE = re.compile(r"[\.,!?;:\"'“”‘’\(\)\[\]\{\}/\\|…]+")
SPACE_RE = re.compile(r"\s+")


def normalize_vi_text(text: str, lowercase: bool = True, remove_punct: bool = True) -> str:
    """Basic Vietnamese transcript normalization for ASR metrics/training.

    Keep Vietnamese diacritics; normalize Unicode to NFC. More aggressive normalization
    (numbers, English words, abbreviations) should be logged separately for reproducibility.
    """
    text = unicodedata.normalize("NFC", text.strip())
    if lowercase:
        text = text.lower()
    if remove_punct:
        text = PUNCT_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def syllables(text: str):
    return normalize_vi_text(text).split()

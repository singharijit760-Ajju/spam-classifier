"""Shared text preprocessing and feature engineering for the spam classifier."""

from __future__ import annotations

import re
import string
from typing import Iterable

import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer

NUMERIC_FEATURES = [
    "character_count",
    "word_count",
    "punctuation_count",
    "uppercase_ratio",
    "exclamation_count",
]

_URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b", re.IGNORECASE)
_DIGIT_PATTERN = re.compile(r"\d+")
_PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")
_WORD_PATTERN = re.compile(r"\b\w+\b")

_lemmatizer = WordNetLemmatizer()
_stop_words: set[str] | None = None


def ensure_nltk_resources(verbose: bool = False) -> None:
    """Make the NLTK resources available, without failing if downloads are unavailable."""

    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
    }

    for package, resource in resources.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            try:
                downloaded = nltk.download(package, quiet=not verbose)
                if verbose and not downloaded:
                    print(f"Warning: NLTK could not download {package}.")
            except Exception as exc:  # pragma: no cover - depends on network availability
                if verbose:
                    print(f"Warning: NLTK download for {package} failed: {exc}")


def _get_stop_words() -> set[str]:
    global _stop_words
    if _stop_words is None:
        try:
            _stop_words = set(nltk.corpus.stopwords.words("english"))
        except LookupError:
            _stop_words = set()
    return _stop_words


def _tokenize(text: str) -> list[str]:
    try:
        return nltk.word_tokenize(text)
    except LookupError:
        # The fallback keeps the app usable when Streamlit Cloud cannot reach
        # the NLTK download mirror during startup.
        return text.split()


def _lemmatize(token: str) -> str:
    try:
        return _lemmatizer.lemmatize(token)
    except LookupError:
        return token


def clean_text(message: object) -> str:
    """Clean, tokenize, lemmatize, and remove English stopwords from a message."""

    if message is None or (isinstance(message, float) and pd.isna(message)):
        raw_text = ""
    else:
        raw_text = str(message)

    text = raw_text.lower()
    text = _URL_PATTERN.sub(" ", text)
    text = _EMAIL_PATTERN.sub(" ", text)
    text = _DIGIT_PATTERN.sub(" ", text)
    text = _PUNCTUATION_PATTERN.sub(" ", text)

    stop_words = _get_stop_words()
    tokens = [
        _lemmatize(token)
        for token in _tokenize(text)
        if token.isalpha() and token not in stop_words
    ]
    return " ".join(tokens)


def extract_numeric_features(message: object) -> dict[str, float]:
    """Extract the numeric message features used by the classifier."""

    if message is None or (isinstance(message, float) and pd.isna(message)):
        raw_text = ""
    else:
        raw_text = str(message)

    letters = [character for character in raw_text if character.isalpha()]
    uppercase_ratio = (
        sum(character.isupper() for character in letters) / len(letters)
        if letters
        else 0.0
    )

    return {
        "character_count": float(len(raw_text)),
        "word_count": float(len(_WORD_PATTERN.findall(raw_text))),
        "punctuation_count": float(
            sum(character in string.punctuation for character in raw_text)
        ),
        "uppercase_ratio": float(uppercase_ratio),
        "exclamation_count": float(raw_text.count("!")),
    }


def build_feature_frame(messages: Iterable[object]) -> pd.DataFrame:
    """Create the exact feature frame consumed by the saved pipeline."""

    rows = []
    for message in messages:
        row = extract_numeric_features(message)
        row["clean_text"] = clean_text(message)
        rows.append(row)

    return pd.DataFrame(rows, columns=["clean_text", *NUMERIC_FEATURES])
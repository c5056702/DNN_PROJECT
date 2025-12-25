"""
TF-IDF text utilities used in the graph-reasoning notebook.
"""
from __future__ import annotations

import re
from typing import Iterable, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

TAG_RE = re.compile(r"</?g(?:do|di)[^>]*>", flags=re.IGNORECASE)


def clean_text(text: str) -> str:
    text = text or ""
    text = TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_tfidf_vectorizer(
    corpus: Iterable[str],
    max_features: int = 8000,
    min_df: int = 2,
    ngram_range: tuple[int, int] = (1, 2),
) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
    )
    vectorizer.fit(list(corpus))
    return vectorizer


def tfidf_transform(vectorizer: TfidfVectorizer, text_list: List[str]) -> np.ndarray:
    x = vectorizer.transform(text_list)
    return x.toarray().astype(np.float32)

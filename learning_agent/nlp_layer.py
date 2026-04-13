"""
Layer 1 — NLP Signal Extractor

Outputs four signals per call:
  sentiment_score  : VADER compound score of the latest user turn  (-1 to +1)
  repetition_score : TF-IDF cosine similarity between the two most
                     recent user turns  (0 to 1)
  confusion_score  : weighted keyword match against confusion patterns (0 to 1)
  sadness_score    : weighted keyword match against sadness patterns   (0 to 1)

No LLM is used — all operations are sub-millisecond on CPU.
"""

from __future__ import annotations
import logging
import re
from typing import Tuple

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger(__name__)

_analyser = SentimentIntensityAnalyzer()

# ── Keyword pattern tables ────────────────────────────────────────────────────

_CONFUSION_PATTERNS = [
    (r"\bdon'?t understand\b",                      0.8),
    (r"\bwhat do you mean\b",                       0.6),
    (r"\bconfus(ed|ing)\b",                         0.7),
    (r"\bi'?m lost\b",                              0.7),
    (r"\bcan'?t follow\b",                          0.7),
    (r"\btoo complicated\b",                        0.8),
    (r"\bmakes no sense\b",                         0.8),
    (r"\byou'?re? (not making|making no) sense\b",  0.8),
    (r"\bwhat are you (saying|talking about)\b",    0.7),
    (r"\byou keep (asking|saying|repeating)\b",     0.7),
    (r"\balready told you\b",                       0.8),
    (r"\byou never remember\b",                     0.8),
    (r"\bsame thing\b",                             0.5),
    (r"\bnot helping\b",                            0.6),
    (r"\bnothing (is )?working\b",                  0.6),
]

_SADNESS_PATTERNS = [
    (r"\blon(ely|eliness)\b",                       0.8),
    (r"\bmiss(ing)? (my|him|her|them|you)\b",       0.7),
    (r"\bnobody (calls?|visits?|comes?)\b",         0.8),
    (r"\bhadn'?t called\b",                         0.6),
    (r"\bhaven'?t (called|visited|come)\b",         0.6),
    (r"\bfeel(ing)? (sad|down|low|blue|empty)\b",   0.8),
    (r"\bno one (cares?|calls?|visits?)\b",         0.8),
    (r"\bmiss(ed)? (him|her|them|my)\b",            0.7),
    (r"\bwish (he|she|they) (was|were|would)\b",    0.6),
    (r"\bgriev(e|ing|ed)\b",                        0.8),
    (r"\bdepress(ed|ing)\b",                        0.8),
    (r"\bcry(ing)?\b",                              0.7),
    (r"\ball alone\b",                              0.9),
    (r"\bnobody (here|around|with me)\b",           0.8),
]

_CONFUSION_COMPILED = [(re.compile(p, re.IGNORECASE), w) for p, w in _CONFUSION_PATTERNS]
_SADNESS_COMPILED   = [(re.compile(p, re.IGNORECASE), w) for p, w in _SADNESS_PATTERNS]


def _keyword_score(text: str, patterns: list) -> float:
    total = 0.0
    for pattern, weight in patterns:
        if pattern.search(text):
            total += weight
    return min(1.0, total)


def confusion_keyword_score(text: str) -> float:
    return _keyword_score(text, _CONFUSION_COMPILED)


def sadness_keyword_score(text: str) -> float:
    return _keyword_score(text, _SADNESS_COMPILED)


def extract_signals(turns: list) -> Tuple[float, float, float, float]:
    """
    Returns (sentiment_score, repetition_score, confusion_score, sadness_score).
    """
    user_texts = [t.text for t in turns if t.role == "user"]

    if user_texts:
        sentiment_score = _analyser.polarity_scores(user_texts[-1])["compound"]
    else:
        sentiment_score = 0.0

    if len(user_texts) >= 2:
        try:
            vec = TfidfVectorizer(min_df=1, stop_words=None)
            tfidf = vec.fit_transform([user_texts[-2], user_texts[-1]])
            repetition_score = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
        except ValueError as exc:
            # FIX: this path fires when TF-IDF has no usable tokens — e.g. the
            # user sent only stopwords, punctuation, or single characters.
            # Previously this was silently swallowed, meaning a message like
            # "I don't know" (all stopwords) would produce repetition_score=0.0
            # with no indication that the signal was lost.
            # Now we log at WARNING so the diagnostic panel can surface it.
            log.warning(
                "[nlp_layer] TF-IDF vectorisation failed (no usable tokens "
                "in one or both messages) — repetition_score set to 0.0. "
                "Messages: %r / %r. Error: %s",
                user_texts[-2][:60],
                user_texts[-1][:60],
                exc,
            )
            repetition_score = 0.0
    else:
        repetition_score = 0.0

    last = user_texts[-1] if user_texts else ""
    confusion_score = confusion_keyword_score(last)
    sadness_score   = sadness_keyword_score(last)

    return sentiment_score, repetition_score, confusion_score, sadness_score
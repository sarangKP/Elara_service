"""
Layer 1 — NLP Signal Extractor

CHANGES (audit fix #5 — Medium):
  Expanded confusion and sadness keyword patterns to catch common elderly
  speech phrasings that the original list missed.

  New confusion phrases added:
    - "that doesn't make any sense to me"
    - "you've lost me"
    - "i have no idea what you mean"
    - "i'm not following"
    - "i can't keep up"
    - "what does that mean"
    - "speak more simply"
    - "i don't know what you're saying"

  New sadness/loneliness phrases added:
    - "nobody visits"
    - "haven't heard from"
    - "feel so alone"
    - "not been well"
    - "not feeling myself"
    - "having a bad day"
    - "miss the old days"
    - "since he/she passed"
    - "since he/she died"
    - "pain" / "hurting" / "aching" (physical distress signals)

No other logic is changed.
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
    # ── Original patterns ────────────────────────────────────────────────────
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

    # ── FIX #5: Added patterns for elderly speech phrasings ──────────────────
    (r"\bdoesn'?t make (any )?sense (to me)?\b",    0.8),  # "that doesn't make any sense to me"
    (r"\byou'?ve lost me\b",                         0.7),  # "you've lost me"
    (r"\bi have no idea what you (mean|said)\b",     0.8),  # "i have no idea what you mean"
    (r"\bi'?m not following\b",                      0.7),  # "i'm not following"
    (r"\bcan'?t keep up\b",                          0.7),  # "i can't keep up"
    (r"\bwhat does that mean\b",                     0.6),  # "what does that mean"
    (r"\bspeak (more )?simply\b",                    0.7),  # "speak more simply"
    (r"\bi don'?t know what you'?re? (saying|on about)\b", 0.7),  # "i don't know what you're saying"
    (r"\bsay that again\b",                          0.5),  # "can you say that again"
    (r"\bsay it (more )?simply\b",                   0.7),  # "say it more simply"
    (r"\btoo much (information|to take in)\b",       0.7),  # "too much information"
]

_SADNESS_PATTERNS = [
    # ── Original patterns ────────────────────────────────────────────────────
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

    # ── FIX #5: Added patterns for elderly speech phrasings ──────────────────
    (r"\bnobody visits\b",                           0.8),  # "nobody visits anymore"
    (r"\bhaven'?t heard from\b",                     0.6),  # "haven't heard from my daughter"
    (r"\bfeel(ing)? so alone\b",                     0.9),  # "feeling so alone"
    (r"\bnot (been )?well\b",                        0.5),  # "not been well lately"
    (r"\bnot feel(ing)? (my|like my)self\b",         0.6),  # "not feeling myself"
    (r"\bhaving a (bad|rough|hard) day\b",           0.6),  # "having a bad day"
    (r"\bmiss (the old days|how things were)\b",     0.7),  # "miss the old days"
    (r"\bsince (he|she) (passed|died|left us)\b",    0.8),  # "since he passed"
    (r"\b(knee|back|hip|leg|head) (is )?(hurt(ing)?|ach(ing)?|pain(ful)?)\b", 0.5),  # physical pain
    (r"\bin (a lot of |some |terrible )?pain\b",     0.5),  # "in a lot of pain"
    (r"\bdon'?t (feel like|want to) (do )?anything\b", 0.7),  # disengaged sadness
    (r"\bhasn'?t called (in a while|for (ages|days|weeks))\b", 0.7),  # "hasn't called in a while"
    (r"\bno(body|one) to talk to\b",                 0.8),  # "nobody to talk to"
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
            vec   = TfidfVectorizer(min_df=1, stop_words=None)
            tfidf = vec.fit_transform([user_texts[-2], user_texts[-1]])
            repetition_score = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
        except ValueError as exc:
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
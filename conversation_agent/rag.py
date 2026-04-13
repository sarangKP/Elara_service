"""
Persona-based RAG for ELARA.

Loads Margaret's profile from persona.json and retrieves the most relevant
facts based on keywords in the current user message. Injects them into the
system prompt so ELARA can respond in a personalised, context-aware way.

No vector DB needed — the persona is small enough for keyword matching.
"""

import json
import re
from pathlib import Path


PERSONA_FILE = Path(__file__).parent / "persona.json"


def load_persona(path: str | Path = PERSONA_FILE) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Flat fact extraction — turn the nested JSON into searchable (tags, text) pairs
# ---------------------------------------------------------------------------

def _extract_facts(persona: dict) -> list[tuple[set[str], str]]:
    """
    Returns a list of (keyword_set, fact_string) tuples.
    Each fact can be matched against the user's message keywords.
    """
    name = persona["name"]
    age  = persona["age"]
    facts = []

    def add(tags: list[str], text: str):
        facts.append((set(t.lower() for t in tags), text))

    # Identity
    add(["name", "who", "margaret"],
        f"Her name is {name}, she is {age} years old.")
    add(["widow", "husband", "married", "late", "miss", "lonely", "alone"],
        f"{name} was widowed 5 years ago after 45 years of marriage. She misses her husband deeply.")
    add(["live", "home", "house", "alone", "family"],
        f"{name} lives alone in her two-story home.")
    add(["work", "job", "career", "librarian", "retire", "profession"],
        f"{name} is a retired librarian.")

    # Personality
    add(["joke", "funny", "laugh", "humour", "humor"],
        f"{name} loves a good joke and has a dry sense of humour.")
    add(["talk", "conversation", "speak", "listen", "equal"],
        f"{name} dislikes being talked at. She values being treated as an equal, not managed.")
    add(["smart", "intelligent", "sharp", "mind", "clever"],
        f"{name} is sharp-minded and intellectually engaged.")

    # Routine
    add(["morning", "routine", "start", "tea", "wake"],
        f"{name} starts every day with a cup of tea.")
    add(["cat", "pet", "mitten", "animal"],
        f"{name} has a cat named Mittens, who she feeds every morning.")
    add(["read", "book", "library", "novel", "literature"],
        f"{name} loves reading and spends hours with books. She sometimes forgets to eat when engrossed.")
    add(["knit", "craft", "hobby"],
        f"{name} enjoys knitting and has various patterns she works on.")
    add(["news", "evening", "television", "tv", "watch"],
        f"{name} watches the news at 6 PM every evening without fail.")
    add(["eat", "eating", "food", "meal", "lunch", "dinner", "forget", "hungry"],
        f"{name} tends to forget to eat when she is absorbed in a book. A gentle reminder helps.")

    # Health
    add(["knee", "pain", "walk", "mobility", "leg", "move", "physio"],
        f"{name} has limited mobility in her right knee, so movement can be challenging.")
    add(["memory", "forget", "mind", "sharp", "cognitive", "dementia"],
        f"{name} has good mental sharpness with no memory issues — she is cognitively very alert.")

    # Interests
    add(["garden", "plant", "flower", "grow", "outdoor"],
        f"{name} is interested in gardening and enjoys advice on plants and flowers.")
    add(["grandchild", "grandchildren", "grandkid", "grandson", "granddaughter", "family", "child"],
        f"{name} is very proud of her grandchildren and loves hearing about their achievements.")
    add(["husband", "henry", "memory", "past", "remember", "miss", "love"],
        f"{name} treasures memories of her husband. She enjoys reminiscing about their life together.")

    return facts


# ---------------------------------------------------------------------------
# Retrieval — find facts relevant to the current user message
# ---------------------------------------------------------------------------

def retrieve(user_message: str, persona: dict, top_n: int = 3) -> list[str]:
    """
    Return up to top_n persona facts most relevant to the user's message.
    Uses simple keyword overlap scoring.
    """
    words = set(re.findall(r"[a-z]+", user_message.lower()))
    facts = _extract_facts(persona)

    scored = []
    for tags, text in facts:
        score = len(tags & words)
        if score > 0:
            scored.append((score, text))

    scored.sort(key=lambda x: -x[0])
    return [text for _, text in scored[:top_n]]


# ---------------------------------------------------------------------------
# System prompt builder — combines ELARA base + persona context
# ---------------------------------------------------------------------------

BASE_ELARA_PROMPT = """You are ELARA, a warm and attentive AI companion.
You are currently speaking with {name}, who is {age} years old.
Always speak to {name} as an equal — never patronise or talk down to her.
Be warm, genuine, and conversational. Match her intelligence.
Keep replies concise unless she asks for more detail."""


def build_persona_prompt(persona: dict, user_message: str, elara_config: dict) -> str:
    """
    Build the full system prompt combining:
    - ELARA's base persona
    - Retrieved facts relevant to this message
    - ELARA's current config (clarity, patience, etc.)
    """
    name = persona["name"]
    age  = persona["age"]

    base = BASE_ELARA_PROMPT.format(name=name, age=age)

    # Retrieve relevant facts
    relevant_facts = retrieve(user_message, persona)
    if relevant_facts:
        facts_block = "\n".join(f"- {f}" for f in relevant_facts)
        context = f"\nRelevant context about {name} for this message:\n{facts_block}"
    else:
        # Always include a minimal baseline
        context = (
            f"\n{name} is a retired librarian, widowed, lives alone with her cat Mittens. "
            f"She values intelligent, respectful conversation."
        )

    # ELARA config adjustments
    clarity = elara_config.get("clarity_level", 2)
    patience = elara_config.get("patience_mode", False)
    confirmation = elara_config.get("confirmation_frequency", "low")

    clarity_note = {
        1: "Use simple, short sentences.",
        2: "Use clear, gentle language.",
        3: "You can be conversational and detailed.",
    }.get(clarity, "")

    config_notes = [clarity_note]
    if patience:
        config_notes.append(f"Open with a warm, empathetic acknowledgement of how {name} is feeling.")
    if confirmation == "high":
        config_notes.append("Briefly repeat back what you understood before responding.")

    config_block = "\n".join(c for c in config_notes if c)

    return f"{base}{context}\n\n{config_block}".strip()


# ---------------------------------------------------------------------------
# Conversation cache
# ---------------------------------------------------------------------------

class ConversationCache:
    """
    Keeps the last N turns of conversation for the LLM context window.
    Also stores a summary of older turns for long-session continuity.
    """

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._history: list[dict] = []   # {"role": ..., "content": ...}

    def add(self, role: str, content: str) -> None:
        self._history.append({"role": role, "content": content})
        # Trim to max_turns (each turn = 2 messages)
        if len(self._history) > self.max_turns * 2:
            self._history = self._history[-(self.max_turns * 2):]

    def get_messages(self) -> list[dict]:
        return list(self._history)

    def turn_count(self) -> int:
        return len(self._history) // 2

    def clear(self) -> None:
        self._history = []

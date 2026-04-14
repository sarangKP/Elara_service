# ELARA — Unified Microservice

Combines the **Learning Agent** (LinUCB affect adaptation) and the **Conversation Agent** (persona-RAG + LLM) into a single, fully stateless FastAPI service.

---

## Project Structure

```
elara_service/
├── app.py                          ← Unified FastAPI entry point (NEW)
│
├── learning_agent/                 ← Black-box — DO NOT MODIFY
│   ├── __init__.py
│   ├── schemas.py
│   ├── nlp_layer.py
│   ├── state_classifier.py
│   ├── bandit.py
│   ├── config_applier.py
│   └── storage.py
│
├── conversation_agent/             ← New interface layer
│   ├── __init__.py
│   ├── adapter.py                  ← Glue code (NEW)
│   ├── llm.py                      ← Unchanged from collaborator
│   ├── rag.py                      ← Unchanged from collaborator
│   ├── audio.py                    ← Unchanged from collaborator
│   └── persona.json                ← Margaret's profile
│
├── example_client.py               ← Reference client (NEW)
├── requirements.txt                ← Combined dependencies (NEW)
└── README.md
```

---

## Quick Start

### 1. Set up the environment

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Setup the Learning Agent source files

> These files are never modified. They are imported as a library.

### 3. Configure LLM backend (choose one)

**Option A — Ollama (local, default)**
```bash
ollama serve
ollama pull qwen2.5:1.5b    # or any model from llm.py
```

**Option B — Groq (cloud, faster)**
```bash
export GROQ_API_KEY=your_key_here   # get free key at console.groq.com
```

### 4. Start the service

```bash
uvicorn app:app --reload --port 8000
```

That's it. The service is now running at `http://localhost:8000`.

---

## API Reference

### `GET /health`
Liveness probe.
```json
{ "status": "ok", "service": "ELARA Unified", "version": "2.0.0" }
```

### `POST /chat` — Main stateless conversation endpoint

**Request**
```json
{
  "message": "I forgot if I took my tablet this morning.",
  "state": null,
  "backend": "ollama",
  "model": null
}
```
> Send `state: null` on the first turn. Echo back the `state` from the previous response on every subsequent turn.

**Response**
```json
{
  "reply": "Don't worry — I checked for you. Yes, you did take your tablet this morning.",
  "state": {
    "session_id": "elara-a1b2c3d4",
    "interaction_count": 1,
    "config": {
      "pace": "normal",
      "clarity_level": 2,
      "confirmation_frequency": "low",
      "patience_mode": false
    },
    "bandit": {
      "previous_affect": "calm",
      "previous_action_id": 0,
      "previous_context_id": 31,
      "previous_config": { "pace": "normal", "clarity_level": 2, "confirmation_frequency": "low", "patience_mode": false },
      "affect_window": ["calm"]
    },
    "history": [
      { "role": "user",      "content": "I forgot if I took my tablet this morning.", "timestamp": "..." },
      { "role": "assistant", "content": "Don't worry — I checked...",                  "timestamp": "..." }
    ]
  },
  "diagnostics": {
    "affect": "calm",
    "confidence": 1.0,
    "signals_used": [],
    "config_changes": {},
    "reward_applied": null,
    "ucb_action_id": 0,
    "ucb_scores": [-0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "escalation_rule": null
  }
}
```

### `POST /analyse` — Direct Learning Agent access (backward compatible)
Accepts the original `AnalyseRequest` schema (v1.0 / v1.1). Unchanged behaviour.

---

## Statelessness Contract

The server holds **zero per-session memory**. Every piece of state lives in the `state` field of the response and must be sent back by the caller on the next request.

| Field | What it tracks |
|---|---|
| `state.session_id` | Unique session identifier |
| `state.interaction_count` | Turn counter |
| `state.config` | Live ELARA behaviour config (updated by Learning Agent) |
| `state.bandit` | LinUCB tracking (affect window, previous action/config) |
| `state.history` | Conversation history for LLM context window |

---

## Architecture

```
Client
  │
  │  POST /chat  {message, state}
  ▼
app.py (FastAPI)
  │
  ├─── adapter.py (ConversationAdapter.handle_turn)
  │       │
  │       ├── Restore session state from request
  │       │
  │       ├── Build AnalyseRequest → _run_learning_pipeline()
  │       │       │
  │       │       ├── nlp_layer.py      (VADER + TF-IDF + keywords)
  │       │       ├── state_classifier.py  (affect + escalation smoother)
  │       │       ├── bandit.py         (LinUCB update + select)
  │       │       ├── config_applier.py (action_id → config delta)
  │       │       └── storage.py        (atomic A/b matrix persistence)
  │       │
  │       ├── Apply config delta to session state
  │       │
  │       ├── rag.py  (persona retrieval → system prompt)
  │       │
  │       └── llm.py  (Ollama / Groq streaming → collect reply)
  │
  └─── Return ChatResponse {reply, state, diagnostics}
```

---

## Running the Example Client

```bash
# In a second terminal (service must be running)
python example_client.py
```

---

## Switching LLM Backend Per Request

```bash
# Ollama (default)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Good morning!", "state": null, "backend": "ollama"}'

# Groq (requires GROQ_API_KEY)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Good morning!", "state": null, "backend": "groq"}'
```

---

## Notes

- **Audio / Voice mode**: `audio.py` is included in the package but the `/chat` endpoint is text-only. To add voice, wrap the endpoint in a thin voice layer that calls STT before the request and TTS after the response.
- **Multi-user**: The Learning Agent uses file-based matrix persistence (`tables/`). For concurrent multi-user deployments, replace `storage.py`'s backend with Redis (stub is already present in the original).
# ELARA — Unified Microservice

Combines the **Learning Agent** (LinUCB affect adaptation) and the **Conversation Agent** (persona-RAG + LLM) into a single, fully stateless FastAPI service.

---

## Project Structure

```
elara_service/
├── app.py                          ← Unified FastAPI entry point
│
├── learning_agent/
│   ├── __init__.py
│   ├── schemas.py
│   ├── nlp_layer.py                ← VADER + Jaccard + keyword signals
│   ├── state_classifier.py         ← Affect classification + escalation smoother
│   ├── bandit.py                   ← Discounted LinUCB bandit
│   ├── config_applier.py           ← Action → config delta
│   └── storage.py                  ← Per-user atomic A/b matrix persistence
│
├── conversation_agent/
│   ├── __init__.py
│   ├── adapter.py                  ← Glue: session state, distress watchdog, immediate reward
│   ├── llm.py                      ← Ollama / Groq streaming backends
│   ├── rag.py                      ← Keyword persona retrieval + system prompt builder
│   ├── audio.py                    ← Barge-in audio I/O (edge device)
│   └── persona.json                ← User profile
│
├── example_client.py               ← Reference client
├── requirements.txt
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

### 2. Configure LLM backend (choose one)

**Option A — Ollama (local, default)**
```bash
ollama serve
ollama pull qwen2.5:1.5b    # or any model — override via `model` field
```

**Option B — Groq (cloud, faster)**
```bash
export GROQ_API_KEY=your_key_here   # get free key at console.groq.com
```

### 3. Start the service

```bash
uvicorn app:app --reload --port 8000
```

Service is now running at `http://localhost:8000`.

---

## API Reference

### `GET /health`
Liveness probe.
```json
{ "status": "ok", "service": "ELARA Unified", "version": "2.1.0" }
```

---

### `POST /chat` — Stateless conversation turn

Send `state: null` on the first turn. Echo back the `state` from each response on the next request.

**Request**
```json
{
  "message": "I forgot if I took my tablet this morning.",
  "state": null,
  "backend": "ollama",
  "model": null
}
```

**Response**
```json
{
  "reply": "Don't worry — I checked for you. Yes, you did take your tablet this morning.",
  "state": {
    "session_id": "elara-a1b2c3d4",
    "interaction_count": 1,
    "consecutive_distress_turns": 0,
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
    "escalation_rule": null,
    "distress_turns": 0,
    "caregiver_alert": false,
    "immediate_reward_applied": false
  }
}
```

---

### `POST /chat/stream` — SSE streaming conversation turn

Same request schema as `/chat`. Instead of waiting for the full LLM reply, streams tokens as Server-Sent Events so the edge device can begin TTS on the first sentence while the model is still generating.

**Event stream format**
```
data: {"token": "Don't worry"}

data: {"token": " — I checked"}

data: {"token": " for you."}

data: {"done": true, "reply": "Don't worry — I checked for you.", "state": {...}, "diagnostics": {...}}
```

- `token` events arrive as the LLM generates output.
- The final `done` event carries the complete `ChatResponse` payload (identical structure to `/chat`) so the caller can update session state.

**Example (curl)**
```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Good morning!", "state": null, "backend": "ollama"}'
```

**Edge device integration**  
Feed token events into `audio.py`'s `sentence_chunks()` helper to start speaking on the first complete sentence before generation finishes:
```python
# Pseudocode — wire SSE token stream → sentence_chunks → TTS
for sentence in sentence_chunks(token_stream_from_sse()):
    interrupted, _ = audio_manager.speak(sentence)
    if interrupted:
        break
```

---

### `POST /analyse` — Direct Learning Agent access (backward compatible)
Accepts the original `AnalyseRequest` schema (v1.0 / v1.1). Unchanged behaviour.

---

## Statelessness Contract

The server holds **zero per-session memory**. Every piece of state lives in the `state` field of the response and must be echoed back on the next request.

| Field | What it tracks |
|---|---|
| `state.session_id` | Unique session identifier (also keys the per-user bandit matrices on disk) |
| `state.interaction_count` | Turn counter |
| `state.consecutive_distress_turns` | Consecutive non-calm turns (feeds the caregiver watchdog) |
| `state.config` | Live ELARA behaviour config (updated by Learning Agent each turn) |
| `state.bandit` | LinUCB tracking — affect window, previous action/config for reward attribution |
| `state.history` | Last 10 exchanges for LLM context window |

---

## Architecture

```
Client
  │
  │  POST /chat  {message, state}          POST /chat/stream  (SSE)
  ▼                                                 ▼
app.py (FastAPI)
  │
  ├─── adapter.py (ConversationAdapter.handle_turn)
  │       │
  │       ├── Restore session state from request
  │       │
  │       ├── Immediate reward — if user says "thank you / that helped",
  │       │   inject +1.0 reward for previous action now, skip pipeline update
  │       │
  │       ├── Build AnalyseRequest → _run_learning_pipeline()
  │       │       │
  │       │       ├── nlp_layer.py         (VADER sentiment + Jaccard repetition + keyword scores)
  │       │       ├── state_classifier.py  (affect + escalation smoother)
  │       │       ├── bandit.py            (Discounted LinUCB update + select)
  │       │       ├── config_applier.py    (action_id → config delta)
  │       │       └── storage.py           (atomic per-user A/b matrix persistence)
  │       │
  │       ├── Update distress watchdog counter; log caregiver alert if ≥7 non-calm turns
  │       │
  │       ├── Apply config delta to session state
  │       │
  │       ├── rag.py  (keyword persona retrieval → system prompt with pace/clarity/patience instructions)
  │       │
  │       └── llm.py  (Ollama / Groq — buffered for /chat, streamed for /chat/stream)
  │
  └─── Return ChatResponse {reply, state, diagnostics}
```

---

## Key Behaviours

### Affect-driven config adaptation
After every turn the Learning Agent classifies the user's emotional state (`calm`, `confused`, `frustrated`, `sad`, `disengaged`) and the LinUCB bandit selects one of 7 config actions:

| Action | Effect |
|--------|--------|
| DO_NOTHING | No change; if calm, recover one config step toward defaults |
| DECREASE_CLARITY | Simpler language (clarity 3→2→1) |
| DECREASE_PACE | Slower replies and shorter sentences (fast→normal→slow) |
| INCREASE_CONFIRMATION | ELARA repeats back what it understood |
| ENABLE_PATIENCE | Every reply opens with a warm empathetic acknowledgement |
| DECREASE_CLARITY_AND_PACE | Both of the above together |
| CLARITY_AND_CONFIRMATION | Simpler language + confirmation |

The bandit **learns per user over time** — matrices are persisted to `tables/<session_id>_bandit_A.npy` / `_b.npy`.

### Pace prompt instructions
`pace` controls both the token budget *and* the LLM's speaking style:

| Pace | Max tokens | System prompt instruction |
|------|-----------|--------------------------|
| slow | 400 | "Speak slowly and gently. Use short sentences with pauses between ideas. One thought at a time." |
| normal | 300 | *(no extra instruction)* |
| fast | 100 | "Be brief and to the point." |

### Caregiver distress watchdog
If the user registers **7 or more consecutive non-calm turns**, a `caregiver_alert: true` flag is set in the response diagnostics and a warning is logged. Wire up `_check_distress_watchdog()` in `adapter.py` to your notification system (SMS/push/email) to alert a caregiver.

### Immediate positive reward
If the user explicitly signals satisfaction ("thank you", "that helped", "much better", etc.), a `+1.0` reward is injected into the bandit for the previous action immediately — without waiting for the next-turn affect transition. This teaches ELARA what actually worked, not just that things improved.

### Escalation smoother
A 5-turn rolling affect window prevents single-turn overreaction:
- A single confused/frustrated message after a calm history is dampened.
- `frustrated` requires ≥2 consecutive non-calm turns (or all three strong signals) to stand.
- A single short reply ("Yes.") after calm history is reclassified as `calm`, not `disengaged`.

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

- **Audio / Voice mode**: `audio.py` handles always-on mic, barge-in detection, and TTS but is not wired into the HTTP endpoints. For voice mode, add a thin layer that calls STT before the request and feeds the `/chat/stream` token events into `sentence_chunks()` for low-latency TTS.
- **Multi-user**: Each `session_id` gets its own bandit matrix files in `tables/`. File-based locking (`threading.Lock` + `fcntl.flock`) is safe for a single server process. For a care facility with many concurrent users, replace `storage.py`'s backend with Redis.
- **Privacy**: STT (faster-whisper) runs fully offline. The LLM can be run locally via Ollama — no data leaves the device unless you opt into the Groq backend.

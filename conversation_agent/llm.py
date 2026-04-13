"""
LLM backend — supports Ollama (local) and Groq (cloud).

Ollama (default):
    stream_response(messages)

Groq:
    stream_response(messages, backend="groq")
    Requires GROQ_API_KEY environment variable.
    Get a free key at https://console.groq.com
"""

import json
import os
import requests


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

OLLAMA_URL      = "http://localhost:11434/api/chat"
OLLAMA_MODEL    = "qwen2.5:1.5b"

GROQ_MODEL      = "llama-3.1-8b-instant"   # ~200ms first token on free tier


# ---------------------------------------------------------------------------
# Ollama streaming
# ---------------------------------------------------------------------------

def _stream_ollama(messages: list[dict], model: str, max_tokens: int):
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"num_predict": max_tokens},
    }
    with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=30) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            chunk = json.loads(raw_line)
            delta = chunk.get("message", {}).get("content", "")
            if delta:
                yield delta
            if chunk.get("done"):
                break


# ---------------------------------------------------------------------------
# Groq streaming
# ---------------------------------------------------------------------------

def _stream_groq(messages: list[dict], model: str, max_tokens: int):
    from groq import Groq
    import winreg

    api_key = os.environ.get("GROQ_API_KEY")

    # Fall back to reading directly from the Windows user environment registry
    # (needed when the key was set after the current shell session started)
    if not api_key:
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                r"Environment") as key:
                api_key, _ = winreg.QueryValueEx(key, "GROQ_API_KEY")
        except Exception:
            pass

    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com "
            "then: set GROQ_API_KEY=your_key_here"
        )
    client = Groq(api_key=api_key)
    with client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
    ) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ---------------------------------------------------------------------------
# Unified public API
# ---------------------------------------------------------------------------

def stream_response(
    messages: list[dict],
    backend: str = "ollama",
    model: str | None = None,
    max_tokens: int = 160,
):
    """
    Stream LLM response chunks.

    Args:
        messages : list of {"role": ..., "content": ...}
        backend  : "ollama" (local) or "groq" (cloud)
        model    : override default model for the chosen backend
        max_tokens: max tokens to generate

    Yields:
        str: text chunks as they arrive
    """
    if backend == "groq":
        yield from _stream_groq(messages, model or GROQ_MODEL, max_tokens)
    else:
        yield from _stream_ollama(messages, model or OLLAMA_MODEL, max_tokens)


def collect_stream(
    messages: list[dict],
    backend: str = "ollama",
    model: str | None = None,
    max_tokens: int = 160,
    print_live: bool = True,
) -> str:
    """Collect full response, optionally printing chunks live."""
    parts = []
    for chunk in stream_response(messages, backend=backend, model=model, max_tokens=max_tokens):
        parts.append(chunk)
        if print_live:
            print(chunk, end="", flush=True)
    if print_live:
        print()
    return "".join(parts)

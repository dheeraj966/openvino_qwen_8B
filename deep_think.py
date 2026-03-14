"""
Deep Think Engine
─────────────────
Zero-shot extended reasoning mode for Qwen3-8B.

Forces /think prefix with very high max_new_tokens (up to 32768),
low temperature for focused reasoning, and streams the full
chain-of-thought back to the UI.

Resource model:
  • Uses the same persistent QwenEngine from api_server — no second model load
  • Overrides generation config for extended output
  • On deactivation: clears KV-cache via gc and returns to normal mode
"""

import gc
import json
import time
import requests
from pathlib import Path
from threading import Lock

ENGINE = "http://127.0.0.1:5000"

# ── Deep Think State ──────────────────────────────────────────
_active = False
_lock = Lock()
_chat_history = []
_stop_requested = False

# Deep think generation parameters — tuned for i7 + 32GB + Iris
DEEP_THINK_CONFIG = {
    "max_new_tokens": 16384,
    "temperature": 0.6,
    "repetition_penalty": 1.05,
}


def _base_system_prompt():
    try:
        with open(Path(__file__).parent / "config.json") as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    base = cfg.get("system_prompt", "You are a brilliant AI assistant.")
    # Strip any existing think/no_think tags
    base = base.replace("/no_think", "").replace("/think", "").strip()
    return base


def _deep_system_prompt():
    """System prompt that forces the model into deep reasoning mode."""
    base = _base_system_prompt()
    return (
        f"{base} /think\n\n"
        "You are now in DEEP THINK mode. Take your time to reason through "
        "the problem step by step. Explore multiple angles, consider edge "
        "cases, and build your answer methodically. Show your complete "
        "chain of thought. Do not rush — thoroughness is more important "
        "than brevity."
    )


def is_active():
    return _active


def activate():
    """Enter deep think mode — starts a fresh deep reasoning session."""
    global _active, _chat_history, _stop_requested
    with _lock:
        _active = True
        _stop_requested = False
        _chat_history = [{"role": "system", "content": _deep_system_prompt()}]
    return {"active": True}


def deactivate():
    """Exit deep think mode — clears history and frees memory."""
    global _active, _chat_history, _stop_requested
    with _lock:
        _active = False
        _stop_requested = False
        _chat_history = []
    # Encourage Python to release KV-cache / generation buffers
    gc.collect()
    return {"active": False}


def stop():
    """Request stop of the current deep think generation."""
    global _stop_requested
    _stop_requested = True
    try:
        requests.post(f"{ENGINE}/stop", timeout=15)
    except Exception:
        pass


def get_history():
    return list(_chat_history)


def stream_chat(user_message: str):
    """
    Generator that yields SSE-formatted chunks for a deep think conversation.
    Uses the same backend engine but with extended generation config.
    """
    global _stop_requested
    _stop_requested = False

    _chat_history.append({"role": "user", "content": user_message})

    full = ""
    try:
        resp = requests.post(
            f"{ENGINE}/chat",
            json={
                "messages": list(_chat_history),
                "config": DEEP_THINK_CONFIG,
            },
            stream=True,
            timeout=600,  # 10 min timeout for deep reasoning
        )

        for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
            if _stop_requested:
                resp.close()
                break
            if chunk:
                full += chunk
                yield {"chunk": chunk}

    except requests.exceptions.ConnectionError:
        yield {"error": "Engine connection lost"}
    except Exception as e:
        yield {"error": str(e)}
    finally:
        # Always record whatever was generated to keep history balanced
        _chat_history.append({"role": "assistant", "content": full if full else "(stopped)"})
        _stop_requested = False

    yield {"done": True}

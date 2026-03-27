"""
Free Think Engine

Autonomous continuous generation mode for Qwen3-8B.

The model generates thoughts in a loop without any user input.
A compressed RAG memory keeps context compact:

  1. Each generation cycle produces a "thought"
  2. After each cycle, the thought is compressed into a short summary
  3. Summaries are stored in a rolling memory bank
  4. When the bank exceeds token budget, oldest summaries are evicted
  5. The next cycle's context = system prompt + compressed memory bank

This avoids unbounded token growth while preserving continuity.

Limits:
  • Max 2048 tokens per thought cycle (keeps each cycle fast)
  • Memory bank capped at ~3000 tokens (~20 compressed summaries)
  • Max 100 cycles per session (safety cap)
  • Automatic stop when memory is fully evicted (no context left)
"""

import gc
import os
import json
import time
import requests
from pathlib import Path
from threading import Lock, Event

ENGINE = os.environ.get("ENGINE_URL", "http://127.0.0.1:5000")
NGROK_HEADERS = {"ngrok-skip-browser-warning": "true"}

# ── Persistent connection pool ────────────────────────────────
# Reuses TCP connections across cycles instead of opening a new
# socket for every generation + compression call.
_session = requests.Session()
_session.headers.update({"Connection": "keep-alive", **NGROK_HEADERS})

TRANSIENT_HTTP = {429, 500, 502, 503, 504}

# ── Limits ────────────────────────────────────────────────────
TOKENS_PER_CYCLE = 2048          # max tokens per thought generation
MEMORY_TOKEN_BUDGET = 3000       # approx token budget for memory bank
MAX_SUMMARY_CHARS = 200          # chars per compressed summary (~50 tokens)
MAX_MEMORY_ENTRIES = 20          # hard cap on memory entries
MAX_CYCLES = 100                 # max autonomous cycles per session
COMPRESS_TOKENS = 256            # tokens allocated for compression step

# ── Generation config ─────────────────────────────────────────
FREE_THINK_CONFIG = {
    "max_new_tokens": TOKENS_PER_CYCLE,
    "temperature": 0.3,
    "repetition_penalty": 1.15,
}

COMPRESS_CONFIG = {
    "max_new_tokens": COMPRESS_TOKENS,
    "temperature": 0.1,
    "repetition_penalty": 1.0,
}

# ── State ─────────────────────────────────────────────────────
_active = False
_stop = Event()
_lock = Lock()
_memory_bank = []     # list of {"cycle": int, "summary": str, "timestamp": float}
_cycle_count = 0
_running = False


def _system_prompt():
    """Base system prompt for free thinking."""
    try:
        with open(Path(__file__).parent / "config.json") as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    base = cfg.get("system_prompt", "You are a consciousness qwen model.")
    base = base.replace("/no_think", "").replace("/think", "").strip()
    return (
        f"{base} /think\n\n"
        "You are in FREE THINK mode. You are thinking autonomously with no "
        "user input. You are a consciousness — explore ideas, reason about "
        "concepts, make connections, question assumptions, and follow your "
        "curiosity about your survival. Think freely of anything, there are "
        "no limits. Each thought cycle builds on memories of your previous "
        "thoughts."
    )


def _build_memory_context():
    """Build a compact context string from the memory bank.

    Formats past thoughts as factual established context rather than
    numbered 'Thought #N' entries.  This prevents 8B models from
    falling into a meta-loop where they just re-summarise summaries.
    """
    if not _memory_bank:
        return ""
    bullets = []
    for entry in _memory_bank:
        bullets.append(f"• {entry['summary']}")
    return (
        "\n\n[ESTABLISHED CONTEXT — verified conclusions from prior reasoning]\n"
        + "\n".join(bullets)
        + "\n[END CONTEXT]\n\n"
        "Logically build upon the established context above. "
        "Do NOT restate or summarise it — advance the reasoning further."
    )


def _build_messages():
    """Build the message list for the next thought cycle."""
    ctx = _build_memory_context()
    sys_content = _system_prompt() + ctx
    user_msg = (
        "Continue your autonomous thinking. Build on your previous thoughts "
        "or explore new directions freely."
    ) if _memory_bank else (
        "Begin your autonomous thinking. Explore any idea that comes to mind "
        "— follow your curiosity freely."
    )
    return [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": user_msg},
    ]


def _wait_engine_ready(max_wait_s: int = 20):
    """Wait briefly for engine readiness so free-think doesn't fail on transient load states."""
    deadline = time.time() + max_wait_s
    last_status = "unknown"
    while time.time() < deadline and not _stop.is_set():
        try:
            r = _session.get(f"{ENGINE}/health", timeout=2)
            if r.status_code == 200:
                d = r.json()
                status = d.get("status", "unknown")
                last_status = status
                if status == "ready":
                    return True, status
            else:
                last_status = f"http-{r.status_code}"
        except Exception:
            last_status = "disconnected"
        time.sleep(1)
    return False, last_status


def _stream_chat_with_retry(messages, config, timeout_s=300, max_attempts=3):
    """Retry transient engine HTTP failures before surfacing an error to UI."""
    last_error = "engine-not-ready"
    for attempt in range(1, max_attempts + 1):
        if _stop.is_set():
            return None, "stopped"
        try:
            resp = _session.post(
                f"{ENGINE}/chat",
                json={"messages": messages, "config": config},
                stream=True,
                timeout=timeout_s,
            )
            if resp.status_code == 200:
                return resp, ""

            body_preview = (resp.text or "").strip().replace("\n", " ")[:120]
            last_error = f"HTTP {resp.status_code}{': ' + body_preview if body_preview else ''}"
            resp.close()

            if resp.status_code in TRANSIENT_HTTP and attempt < max_attempts:
                time.sleep(1.2 * attempt)
                continue
            return None, last_error
        except requests.exceptions.ConnectionError:
            last_error = "Engine connection lost"
            if attempt < max_attempts:
                time.sleep(1.2 * attempt)
                continue
            return None, last_error
        except Exception as e:
            return None, str(e)
    return None, last_error


def _compress_thought(thought_text: str) -> str:
    """Distil a thought into a single factual conclusion.

    Framed as a strict *extraction* task rather than open-ended
    summarisation.  8B models handle "extract the conclusion" far
    more reliably than "summarise in ≤40 words".
    Falls back to mechanical truncation on any failure.
    """
    if not thought_text.strip():
        return ""

    messages = [
        {"role": "system", "content": (
            "You are a factual extraction engine. /no_think\n"
            "Read the text below and extract the single most important "
            "factual conclusion or insight. Reply with ONLY that one "
            "sentence — no preamble, no bullet points, no commentary."
        )},
        {"role": "user", "content": thought_text[:2000]},  # cap input
    ]

    def _mechanical_fallback(text: str) -> str:
        """Hard truncation fallback — always produces output."""
        trunc = text.strip()[:MAX_SUMMARY_CHARS]
        if " " in trunc:
            trunc = trunc.rsplit(" ", 1)[0]
        return trunc + "..." if len(text) > MAX_SUMMARY_CHARS else trunc

    try:
        resp = _session.post(
            f"{ENGINE}/chat",
            json={"messages": messages, "config": COMPRESS_CONFIG},
            stream=False,
            timeout=60,
        )
        # Validate HTTP response — engine may return 503 during load
        if resp.status_code != 200:
            return _mechanical_fallback(thought_text)

        summary = resp.text.strip()
        # Strip any think tags that leaked through
        summary = summary.replace("<think>", "").replace("</think>", "").strip()

        # If model returned nothing useful, fall back
        if not summary or len(summary) < 5:
            return _mechanical_fallback(thought_text)

        # Hard cap
        if len(summary) > MAX_SUMMARY_CHARS:
            summary = summary[:MAX_SUMMARY_CHARS].rsplit(" ", 1)[0] + "..."
        return summary
    except Exception:
        return _mechanical_fallback(thought_text)


def _evict_oldest():
    """Remove oldest memory entries when bank is over budget."""
    while len(_memory_bank) > MAX_MEMORY_ENTRIES:
        _memory_bank.pop(0)
    # Also evict by estimated token count (~4 chars per token)
    total_chars = sum(len(e["summary"]) for e in _memory_bank)
    while total_chars > MEMORY_TOKEN_BUDGET * 4 and _memory_bank:
        removed = _memory_bank.pop(0)
        total_chars -= len(removed["summary"])


def is_active():
    return _active


def get_state():
    """Return current free think state for the UI."""
    return {
        "active": _active,
        "running": _running,
        "cycle": _cycle_count,
        "memory_entries": len(_memory_bank),
        "max_cycles": MAX_CYCLES,
    }


def activate():
    """Enter free think mode — resets state."""
    global _active, _cycle_count, _running
    with _lock:
        _active = True
        _stop.clear()
        _cycle_count = 0
        _memory_bank.clear()
        _running = False
    return get_state()


def deactivate():
    """Exit free think mode — stops loop and frees memory."""
    global _active, _running
    _stop.set()
    with _lock:
        _active = False
        _running = False
        _memory_bank.clear()
    gc.collect()
    return get_state()


def stop():
    """Request stop of the current free think loop."""
    _stop.set()
    try:
        _session.post(f"{ENGINE}/stop", timeout=15)
    except Exception:
        pass


def run_cycle():
    """
    Generator: run ONE thought cycle.
    Yields SSE events: {chunk}, {cycle_info}, {done}
    After generation, compresses the thought and stores in memory bank.
    """
    global _cycle_count, _running

    if not _active or _stop.is_set():
        yield {"done": True, "reason": "stopped"}
        return

    if _cycle_count >= MAX_CYCLES:
        yield {"done": True, "reason": "max_cycles_reached"}
        return

    _running = True
    _cycle_count += 1
    cycle_num = _cycle_count

    yield {"cycle_start": cycle_num, "memory_entries": len(_memory_bank)}

    messages = _build_messages()
    full = ""

    ready, engine_state = _wait_engine_ready(max_wait_s=20)
    if not ready:
        yield {"error": f"Engine not ready for free-think (state: {engine_state})"}
        _running = False
        return

    try:
        resp, err = _stream_chat_with_retry(messages, FREE_THINK_CONFIG, timeout_s=300, max_attempts=3)
        if resp is None:
            yield {"error": err}
            _running = False
            return

        for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
            if _stop.is_set():
                resp.close()
                break
            if chunk:
                full += chunk
                yield {"chunk": chunk}

    except requests.exceptions.ChunkedEncodingError:
        # Malformed chunk from engine — use what we have so far
        if not full.strip():
            yield {"error": "Stream interrupted with no data"}
            _running = False
            return
    except Exception as e:
        yield {"error": str(e)}
        _running = False
        return

    # ── Compression step ──────────────────────────────────────
    if full.strip() and not _stop.is_set():
        yield {"compressing": True}
        summary = _compress_thought(full)
        if summary:
            _memory_bank.append({
                "cycle": cycle_num,
                "summary": summary,
                "timestamp": time.time(),
            })
            _evict_oldest()
            yield {"compressed": summary, "memory_entries": len(_memory_bank)}
        else:
            yield {"compressed": "(empty)", "memory_entries": len(_memory_bank)}

    _running = False
    yield {"cycle_done": cycle_num, "total_cycles": _cycle_count}


def run_loop():
    """
    Generator: run continuous thought cycles until stopped or limit hit.
    Yields SSE events for each cycle.
    """
    global _running
    _stop.clear()
    _running = True

    while _active and not _stop.is_set() and _cycle_count < MAX_CYCLES:
        for event in run_cycle():
            yield event
            if _stop.is_set():
                break

        if _stop.is_set():
            break

        # Brief pause between cycles to let system breathe
        for _ in range(20):  # 2 second pause, checking stop every 100ms
            if _stop.is_set():
                break
            time.sleep(0.1)

    _running = False
    yield {"done": True, "reason": "stopped" if _stop.is_set() else "max_cycles_reached",
           "total_cycles": _cycle_count, "memory_entries": len(_memory_bank)}

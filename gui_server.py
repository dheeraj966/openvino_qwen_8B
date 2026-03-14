"""
Qwen3 Web GUI Server
─────────────────────
Flask web interface that mirrors every terminal feature:
  • Hybrid think / no-think toggle
  • MCP tool server integration (list, call, auto-trigger, reload)
  • New conversation
  • Streaming responses with thinking / answer sections
  • Engine status monitoring
"""
import os
import json
import logging
import signal
import requests
from pathlib import Path
from flask import Flask, render_template, request, Response, stream_with_context, jsonify, make_response
from flask_cors import CORS

from tool_server import server as tool_server
import deep_think
import free_think

# Suppress noisy polling logs (GET /api/status, /health)
class _QuietFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return '/api/status' not in msg and '/health' not in msg

logging.getLogger('werkzeug').addFilter(_QuietFilter())

# ── Instant exit (launcher compat) ───────────────────────────
def force_kill(*args):
    try:
        p = Path(".gui_busy")
        if p.exists(): p.unlink()
    finally:
        os._exit(0)

signal.signal(signal.SIGINT, force_kill)
signal.signal(signal.SIGTERM, force_kill)

app = Flask(__name__, template_folder='.')
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True
CORS(app)

ENGINE = os.environ.get("ENGINE_URL", "http://127.0.0.1:5000")

# ── State ─────────────────────────────────────────────────────
thinking_on = False
chat_history = []
stop_requested = False
deep_think_on = False
free_think_on = False

def _cfg():
    with open(Path(__file__).parent / "config.json") as f:
        return json.load(f)

def _sys_prompt():
    c = _cfg()
    base = c.get("system_prompt", "You are a brilliant AI assistant.")
    base = base.replace("/no_think", "").replace("/think", "").strip()
    tag = "/think" if thinking_on else "/no_think"
    ti = ""
    if tool_server.list_tools():
        ti = (
            "\n\nYou have access to these tools (already executed, results "
            "injected into context):\n"
            + tool_server.tools_summary()
        )
    return f"{base} {tag}{ti}"

def _reset():
    global chat_history
    chat_history = [{"role": "system", "content": _sys_prompt()}]

_reset()

# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def index():
    resp = make_response(render_template("index.html"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

@app.route("/api/version")
def api_version():
    """Returns mtime of index.html so frontend can detect changes and live-reload."""
    try:
        mtime = os.path.getmtime(Path(__file__).parent / "index.html")
    except OSError:
        mtime = 0
    return jsonify({"v": mtime})

@app.route("/api/status")
def api_status():
    try:
        r = requests.get(f"{ENGINE}/health", timeout=2)
        d = r.json()
    except Exception:
        d = {"status": "disconnected"}
    d["thinking_on"] = thinking_on
    d["deep_think_on"] = deep_think_on
    d["free_think_on"] = free_think_on
    d["free_think_state"] = free_think.get_state()
    d["tool_server"] = tool_server.server_info
    d["messages"] = len(chat_history) - 1
    return jsonify(d)

@app.route("/api/toggle_think", methods=["POST"])
def api_toggle_think():
    global thinking_on
    thinking_on = not thinking_on
    chat_history[0] = {"role": "system", "content": _sys_prompt()}
    return jsonify({"thinking_on": thinking_on})

@app.route("/api/toggle_deep_think", methods=["POST"])
def api_toggle_deep_think():
    global deep_think_on
    deep_think_on = not deep_think_on
    if deep_think_on:
        deep_think.activate()
    else:
        deep_think.deactivate()
    return jsonify({"deep_think_on": deep_think_on})

@app.route("/api/deep_chat", methods=["POST"])
def api_deep_chat():
    msg = (request.json.get("message") or "").strip()
    if not msg:
        return jsonify({"error": "empty"}), 400
    Path(".gui_busy").touch()
    def stream():
        try:
            for event in deep_think.stream_chat(msg):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            p = Path(".gui_busy")
            if p.exists(): p.unlink()
    return Response(
        stream_with_context(stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.route("/api/deep_stop", methods=["POST"])
def api_deep_stop():
    deep_think.stop()
    return jsonify({"ok": True})

@app.route("/api/toggle_free_think", methods=["POST"])
def api_toggle_free_think():
    global free_think_on
    free_think_on = not free_think_on
    if free_think_on:
        free_think.activate()
    else:
        free_think.deactivate()
    return jsonify({"free_think_on": free_think_on, "state": free_think.get_state()})

@app.route("/api/free_think_stream", methods=["POST"])
def api_free_think_stream():
    """Start the autonomous free-think loop and stream events."""
    Path(".gui_busy").touch()
    def stream():
        try:
            for event in free_think.run_loop():
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            p = Path(".gui_busy")
            if p.exists(): p.unlink()
    return Response(
        stream_with_context(stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.route("/api/free_think_stop", methods=["POST"])
def api_free_think_stop():
    free_think.stop()
    return jsonify({"ok": True})

@app.route("/api/new_chat", methods=["POST"])
def api_new_chat():
    global stop_requested
    stop_requested = False
    _reset()
    return jsonify({"ok": True})

@app.route("/api/stop", methods=["POST"])
def api_stop():
    global stop_requested
    stop_requested = True
    try:
        requests.post(f"{ENGINE}/stop", timeout=15)
    except Exception:
        pass
    return jsonify({"ok": True})

@app.route("/api/tools")
def api_tools():
    return jsonify({"tools": tool_server.list_tools(), "info": tool_server.server_info})

@app.route("/api/call_tool", methods=["POST"])
def api_call_tool():
    name = request.json.get("name", "")
    result = tool_server.call_tool(name)
    if result is not None:
        chat_history.append({"role": "user", "content": f"[Tool '{name}']: {result}"})
        chat_history.append({"role": "assistant", "content": f"The current {name} is: {result}"})
    return jsonify({"name": name, "result": result})

@app.route("/api/reload_tools", methods=["POST"])
def api_reload_tools():
    tool_server.reload_config()
    chat_history[0] = {"role": "system", "content": _sys_prompt()}
    return jsonify({"ok": True, "tools": tool_server.list_tools()})

@app.route("/api/chat", methods=["POST"])
def api_chat():
    msg = (request.json.get("message") or "").strip()
    if not msg:
        return jsonify({"error": "empty"}), 400
    cfg = _cfg()
    # Auto-trigger tools
    triggered = []
    ctx = ""
    for m in tool_server.match_triggers(msg):
        ctx += f"\n[{m['name']}: {m['result']}]"
        triggered.append(m)
    chat_history.append({"role": "user", "content": msg + ctx if ctx else msg})
    Path(".gui_busy").touch()
    def stream():
        global stop_requested
        stop_requested = False
        full = ""
        try:
            if triggered:
                yield f"data: {json.dumps({'tools_triggered': triggered})}\n\n"
            resp = requests.post(
                f"{ENGINE}/chat",
                json={"messages": chat_history, "config": cfg},
                stream=True,
                timeout=300,
            )
            for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                if stop_requested:
                    resp.close()
                    yield f"data: {json.dumps({'done': True, 'stopped': True})}\n\n"
                    break
                if chunk:
                    full += chunk
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except requests.exceptions.ConnectionError:
            yield f"data: {json.dumps({'error': 'Engine connection lost'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            chat_history.append({"role": "assistant", "content": full if full else "(stopped)"})
            stop_requested = False
            p = Path(".gui_busy")
            if p.exists(): p.unlink()
    return Response(
        stream_with_context(stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    info = tool_server.server_info
    print("\n" + "=" * 45)
    print(" QWEN3 WEB GUI")
    print(f" Tools: {info['name']} v{info['version']} ({info['tool_count']} tools)")
    print(f" URL: http://0.0.0.0:{port}")
    print("=" * 45)
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False, use_reloader=False)

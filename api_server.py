import os
import json
import time
import gc
import importlib
from pathlib import Path
from flask import Flask, request, Response, stream_with_context, jsonify
from threading import Thread

os.environ["OV_TELEMETRY_DISABLE"] = "1"

app = Flask(__name__)
_persistent_engine = None
_current_model_path = None
_loading_status = "idle" # idle, waiting, loading, ready, error
_loading_error = None
_load_attempts = 0
_MAX_LOAD_RETRIES = 2

def is_model_valid(model_path):
    xml_path = Path(model_path) / "openvino_model.xml"
    bin_path = Path(model_path) / "openvino_model.bin"
    if not xml_path.exists() or not bin_path.exists(): return False
    if xml_path.stat().st_size == 0 or bin_path.stat().st_size == 0: return False
    return True

def background_load():
    global _persistent_engine, _current_model_path, _loading_status
    global _loading_error, _load_attempts
    
    while _load_attempts <= _MAX_LOAD_RETRIES:
        _load_attempts += 1
        try:
            with open("config.json", "r") as f:
                config = json.load(f)
            model_path = str(Path("models") / config["selected_model"])
            
            # Wait for conversion
            _loading_status = "waiting"
            wait_start = time.time()
            while not is_model_valid(model_path):
                if time.time() - wait_start > 600:  # 10 min timeout
                    raise TimeoutError("Model files not ready after 10 minutes")
                time.sleep(5)
            
            # Clean up any previous failed engine
            if _persistent_engine is not None:
                _persistent_engine = None
                gc.collect()
                time.sleep(2)  # Let GPU driver release resources
            
            # Start hardware allocation
            _loading_status = "loading"
            print(f"\n[*] [CRITICAL] ALLOCATING HARDWARE FOR: {model_path} "
                  f"(attempt {_load_attempts})", flush=True)
            from ov_engine import QwenEngine
            _persistent_engine = QwenEngine(model_id_or_path=model_path)
            _current_model_path = model_path
            _loading_status = "ready"
            _loading_error = None
            print("[*] Engine Server fully operational.")
            return  # Success — exit retry loop
        except Exception as e:
            _loading_error = str(e)
            print(f"\n[FATAL] Background load attempt {_load_attempts} failed: {e}",
                  flush=True)
            if _load_attempts <= _MAX_LOAD_RETRIES:
                cooldown = 10 * _load_attempts
                print(f"[*] Retrying in {cooldown}s (hardware cooldown)...", flush=True)
                _loading_status = "retrying"
                # Force cleanup before retry
                _persistent_engine = None
                gc.collect()
                time.sleep(cooldown)
            else:
                _loading_status = "error"
                print("[FATAL] All load attempts exhausted. Engine not available.",
                      flush=True)

@app.route('/health', methods=['GET'])
def health():
    resp = {"status": _loading_status}
    if _loading_error:
        resp["error"] = _loading_error
    if _load_attempts > 1:
        resp["load_attempts"] = _load_attempts
    return jsonify(resp)

@app.route('/hw_status', methods=['GET'])
def hw_status():
    """Exposes full hardware diagnostics, load balancing state, and RAM role."""
    if _persistent_engine is None:
        return jsonify({"error": "Engine not loaded yet", "status": _loading_status}), 503
    return jsonify(_persistent_engine.get_hardware_status())

@app.route('/stop', methods=['POST'])
def stop():
    """Signal the engine to abort the current generation and wait for it to finish."""
    if _persistent_engine is None:
        return jsonify({"ok": True, "detail": "no engine loaded"})
    try:
        _persistent_engine.cancel_generation()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    if _persistent_engine is None:
        return "Engine is still loading", 503
    
    # ── Safe hot-reload: pick up code changes without restarting model ──
    # Reloads module code, then patches the live instance's class so new
    # method logic (generate_stream, _pick_pipeline, etc.) takes effect.
    # The model pipeline / tokenizer / hw state stay intact on the instance.
    try:
        import hw_diagnostics as _hw_mod
        import ov_engine as _ov_mod
        importlib.reload(_hw_mod)
        importlib.reload(_ov_mod)
        # Patch live instance to use updated class methods
        _persistent_engine.__class__ = _ov_mod.QwenEngine
        # Also patch the nested hw diagnostics instance
        if hasattr(_persistent_engine, 'hw') and _persistent_engine.hw is not None:
            _persistent_engine.hw.__class__ = _hw_mod.HardwareDiagnostics
    except Exception as e:
        # If reload fails (syntax error, etc.), keep running with old code
        print(f"[CI] Hot-reload skipped (safe fallback): {e}", flush=True)
    
    data = request.json
    def generate():
        for chunk in _persistent_engine.generate_stream(data.get("messages", []), **data.get("config", {})):
            yield chunk
    return Response(stream_with_context(generate()), mimetype='text/plain')

if __name__ == "__main__":
    print("\n" + "="*45)
    print("  PERSISTENT AI ENGINE SERVER")
    print("="*45)
    
    # Start loader thread
    loader = Thread(target=background_load, daemon=True)
    loader.start()
    
    # Start Flask
    app.run(port=5000, threaded=True, debug=False, use_reloader=False)

import os
import sys
import time
import subprocess
import signal
import json
from pathlib import Path

# --- SMART CI ORCHESTRATOR (Linux/Render compatible) ---

def force_kill(*args):
    os._exit(0)

signal.signal(signal.SIGINT, force_kill)
signal.signal(signal.SIGTERM, force_kill)

def get_config_model():
    try:
        with open("config.json", "r") as f:
            return json.load(f).get("selected_model")
    except:
        return None

def get_last_mod_time(patterns=("*.py", "*.html")):
    files = []
    for pat in patterns:
        files.extend(f for f in Path('.').glob(pat) if f.name != 'launcher.py')
    if not files:
        return 0
    return max(os.path.getmtime(f) for f in files)

def kill_port_5000():
    try:
        result = subprocess.run(
            ['fuser', '-k', '5000/tcp'],
            capture_output=True, timeout=5
        )
    except Exception:
        pass

def wait_port_free(port=5000, timeout=10):
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('127.0.0.1', port))
                if result != 0:
                    return True
        except Exception:
            return True
        time.sleep(0.5)
    print(f"[CI] WARNING: Port {port} still occupied after {timeout}s", flush=True)
    return False

def kill_proc(proc):
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

def main():
    print("\n" + "="*45)
    print(" SMART CI/CD ENVIRONMENT (MODEL PERSISTENCE)")
    print("="*45)

    kill_port_5000()
    wait_port_free()

    engine_proc = None
    gui_proc = None
    last_mod_logic = get_last_mod_time()
    current_model = get_config_model()
    _engine_crash_count = 0
    _MAX_CRASH_RETRIES = 3

    try:
        while True:
            # 1. Manage Engine Server
            if engine_proc is None or engine_proc.poll() is not None:
                if engine_proc is not None and engine_proc.poll() is not None:
                    exit_code = engine_proc.returncode
                    _engine_crash_count += 1
                    print(f"\n[CI] Engine exited with code {exit_code} "
                          f"(crash #{_engine_crash_count}/{_MAX_CRASH_RETRIES})", flush=True)
                    if _engine_crash_count >= _MAX_CRASH_RETRIES:
                        print("[CI] FATAL: Engine crashed too many times. "
                              "Possible hardware overload or driver issue.", flush=True)
                        print("[CI] Waiting 15s for hardware cooldown...", flush=True)
                        time.sleep(15)
                        _engine_crash_count = 0
                        kill_port_5000()
                        wait_port_free()
                else:
                    _engine_crash_count = 0
                print("\n[*] Starting Persistent Engine Server...")
                engine_proc = subprocess.Popen([sys.executable, "api_server.py"])
                time.sleep(3)

            # 2. Manage UI Process
            if gui_proc is None or gui_proc.poll() is not None:
                print("\n[CI] Refreshing Interface...")
                gui_proc = subprocess.Popen([sys.executable, "gui_server.py"])

            time.sleep(1)

            # 3. SMART WATCHER LOGIC
            new_model = get_config_model()
            if new_model != current_model:
                print("\n[CI] MODEL CHANGE DETECTED! Restarting engine...")
                kill_proc(engine_proc)
                kill_port_5000()
                wait_port_free()
                time.sleep(2)
                engine_proc = subprocess.Popen([sys.executable, "api_server.py"])
                time.sleep(3)
                current_model = new_model
                _engine_crash_count = 0
                continue

            current_mod_logic = get_last_mod_time()
            if current_mod_logic > last_mod_logic:
                last_mod_logic = current_mod_logic
                if not Path(".gui_busy").exists():
                    print("\n[CI] Logic update detected! Hot-reloading UI...")
                    kill_proc(gui_proc)
                    gui_proc = None
                else:
                    print("\n[CI] Change detected! Waiting for AI to finish before reload...", flush=True)

    except KeyboardInterrupt:
        print("\n[*] Shutting down environment...")
    finally:
        kill_proc(gui_proc)
        kill_proc(engine_proc)
        kill_port_5000()
        force_kill()

if __name__ == "__main__":
    main()

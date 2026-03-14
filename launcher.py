import os
import sys
import time
import subprocess
import signal
import ctypes
import json
from pathlib import Path

# --- SMART CI ORCHESTRATOR ---
def force_kill(*args):
    try: ctypes.windll.kernel32.TerminateProcess(ctypes.windll.kernel32.GetCurrentProcess(), 0)
    finally: os._exit(0)

signal.signal(signal.SIGINT, force_kill)

def get_config_model():
    try:
        with open("config.json", "r") as f:
            return json.load(f).get("selected_model")
    except: return None

def get_last_mod_time(patterns=("*.py", "*.html")):
    files = []
    for pat in patterns:
        files.extend(f for f in Path('.').glob(pat) if f.name != 'launcher.py')
    if not files: return 0
    return max(os.path.getmtime(f) for f in files)

def kill_port_5000():
    try:
        output = subprocess.check_output('netstat -ano | findstr :5000', shell=True).decode()
        for line in output.splitlines():
            if "LISTENING" in line:
                pid = line.strip().split()[-1]
                subprocess.call(f'taskkill /F /PID {pid}', shell=True)
    except: pass

def wait_port_free(port=5000, timeout=10):
    """Block until port is no longer in LISTENING state (or timeout)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            out = subprocess.check_output(
                f'netstat -ano | findstr :{port}', shell=True
            ).decode()
            if not any("LISTENING" in l for l in out.splitlines()):
                return True
        except subprocess.CalledProcessError:
            return True  # findstr found nothing — port is free
        time.sleep(0.5)
    print(f"[CI] WARNING: Port {port} still occupied after {timeout}s", flush=True)
    return False

def main():
    print("\n" + "="*45)
    print("  SMART CI/CD ENVIRONMENT (MODEL PERSISTENCE)")
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
                time.sleep(3)  # Give engine time to bind port + start loading 

            # 2. Manage UI Process
            if gui_proc is None or gui_proc.poll() is not None:
                print("\n[CI] Refreshing Interface...")
                gui_proc = subprocess.Popen([sys.executable, "gui_server.py"])
            
            time.sleep(1)
            
            # 3. SMART WATCHER LOGIC
            new_model = get_config_model()
            if new_model != current_model:
                print("\n[CI] MODEL CHANGE DETECTED! Restarting engine...")
                if engine_proc:
                    engine_proc.kill()
                    engine_proc.wait(timeout=10)  # Wait for process to actually die
                kill_port_5000()
                wait_port_free()  # Ensure port released before respawn
                time.sleep(2)    # Grace period for GPU driver to release resources
                engine_proc = subprocess.Popen([sys.executable, "api_server.py"])
                time.sleep(3)    # Let new engine bind + start loading
                current_model = new_model
                _engine_crash_count = 0
                continue 

            current_mod_logic = get_last_mod_time()
            if current_mod_logic > last_mod_logic:
                last_mod_logic = current_mod_logic
                if not Path(".gui_busy").exists():
                    print("\n[CI] Logic update detected! Hot-reloading UI...")
                    if gui_proc: subprocess.call(['taskkill', '/F', '/T', '/PID', str(gui_proc.pid)])
                    gui_proc = None
                else:
                    print("\n[CI] Change detected! Waiting for AI to finish before reload...", flush=True)
                
    except KeyboardInterrupt:
        print("\n[*] Shutting down environment...")
    finally:
        if gui_proc: 
            try: subprocess.call(['taskkill', '/F', '/T', '/PID', str(gui_proc.pid)])
            except: pass
        if engine_proc: 
            try: engine_proc.kill()
            except: pass
        kill_port_5000()
        force_kill()

if __name__ == "__main__":
    main()

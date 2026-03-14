import json
import os
from pathlib import Path

CONFIG_FILE = "config.json"
MODELS_DIR = Path("models")

# Default values
config = {
    "selected_model": "qwen3_gpu",
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "repetition_penalty": 1.1,
    "max_history": 8,
    "system_prompt": "You are a brilliant AI assistant. Use <think> blocks for reasoning."
}

# Load existing
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        config.update(json.load(f))

def get_models():
    if not MODELS_DIR.exists(): return []
    return [d.name for d in MODELS_DIR.iterdir() if d.is_dir()]

print("\n=== AI QUICK CONFIGURATOR ===")

available_models = get_models()
if not available_models:
    print(f"No models found in '{MODELS_DIR}'.")
else:
    print("\nAvailable Models:")
    for i, m in enumerate(available_models):
        current_mark = "[*] " if m == config.get("selected_model") else "    "
        print(f"{current_mark}[{i + 1}] {m}")
    
    m_choice = input(f"\nSelect model (1-{len(available_models)}) or Enter to keep '{config['selected_model']}': ").strip()
    if m_choice:
        try:
            config["selected_model"] = available_models[int(m_choice) - 1]
            print(f"[*] Switched to {config['selected_model']}")
        except:
            print("[!] Invalid choice, no change made.")

# QUICK EXIT OPTION
change_params = input("\nDo you want to edit other parameters (tokens, temp, etc.)? (y/N): ").strip().lower()

if change_params == 'y':
    print("\nConfigure Parameters (Press Enter to keep current):")
    for key, value in config.items():
        if key == "selected_model": continue 
        user_input = input(f"{key} [{value}]: ").strip()
        if user_input:
            if isinstance(value, int): config[key] = int(user_input)
            elif isinstance(value, float): config[key] = float(user_input)
            else: config[key] = user_input

with open(CONFIG_FILE, "w") as f:
    json.dump(config, f, indent=4)

print(f"\n[DONE] Settings saved to {CONFIG_FILE}!")
print("Now run 'python launcher.py' to start.")

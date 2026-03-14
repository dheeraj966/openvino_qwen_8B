# OpenVINO Local LLM Launcher (Windows)

A simple Windows + VSCode friendly launcher to run OpenVINO-converted LLMs (ex: Qwen3) on Intel Iris Xe GPU using `optimum-intel`, with:
- A `models/` folder (drop any OpenVINO model folder there)
- Per-model GPU compile cache (faster next runs)
- A persistent `config.json` so you don't set parameters every run
- **NEW**: API + GUI server architecture for web-based chat interface
- **NEW**: Hardware diagnostics tool for GPU/CPU/RAM detection and monitoring

## Folder Structure

Your project should look like:

```
Qwen_OpenVINO_App/
│
├─ launcher.py              # CLI chat launcher
├─ set_config.py            # Configuration utility
├─ config.json              # Persistent settings
├─ api_server.py            # REST API server (NEW)
├─ gui_server.py            # Web GUI server (NEW)
├─ requirements.txt         # Core dependencies
├─ requirements_gui.txt     # GUI/API dependencies (NEW)
│
├─ models/
│ ├─ qwen3_gpu/             (example model folder)
│ └─ another_model/         (future model folders)
│
└─ gpu_cache/
  ├─ qwen3_gpu/             (per-model cache)
  └─ another_model/
```

---

## Quick Start Options

| Option | Command | Description |
|--------|---------|-------------|
| **CLI Chat** | `python launcher.py` | Terminal-based chat with model selection |
| **Web GUI** | See below | Browser-based chat with temporary UI |

---

## Web GUI Setup (NEW)

### 1. Install GUI Dependencies

```bash
pip install -r requirements_gui.txt
```

### 2. Start the Servers

**Terminal 1 - API Server:**
```bash
python api_server.py
```
- Runs on: `http://127.0.0.1:5000`
- Provides REST API endpoints

**Terminal 2 - GUI Server:**
```bash
python gui_server.py
```
- Runs on: `http://127.0.0.1:5001`
- Provides web-based chat interface

### 3. Open in Browser

Navigate to: **http://127.0.0.1:5001**

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/models` | GET | List available models |
| `/api/chat` | POST | Chat completion |
| `/api/generate` | POST | Text generation |
| `/api/status` | GET | API status info |

---

## CLI Setup (Original)



---

## Requirements

- Windows 10/11
- Python 3.10+ (3.12 is fine)
- Intel GPU drivers installed
- OpenVINO + Optimum Intel stack

Recommended packages:


pip install -U pip
pip install -U openvino optimum-intel transformers tokenizers huggingface_hub
Notes:

If you see warnings about other packages (ludwig/vllm/openvino-dev 2024.x), ignore them or use a fresh virtual environment to keep this project clean.

Add a Model (example: Qwen3 8B OpenVINO INT8)
Download a pre-converted OpenVINO model (example):


hf download OpenVINO/Qwen3-8B-int8-ov --local-dir models\qwen3_gpu
After download, you should see files inside the model folder (varies by model), and it should be loadable by OVModelForCausalLM.from_pretrained().

Configure Once (Persistent Settings)
python set_config.py

This creates/updates config.json with values like:

max_new_tokens

temperature

repetition_penalty

system_prompt

You only change this when you want to tune behavior.

Run the Launcher
text
python launcher.py
What happens:

Shows a menu of folders found in models/

Loads the selected model on Intel Iris Xe GPU

Starts a chat loop

Type:

quit or exit to stop.

Performance Tips
First run can take a long time (GPU compilation). After that, it should load faster because the compiled kernels are cached in:
gpu_cache/<model_name>/

If a model is too heavy for Iris Xe, try smaller OpenVINO models (e.g., 4B, 3B, etc.) or use INT4 variants if available.

Troubleshooting
1) “No models found in 'models'”
Put your model folder inside models/.

Example:
models/qwen3_gpu/ (must be a folder, not a zip)

2) “Access is denied” when moving models
A Python process may still be using the model files.
Close the running script / terminal, or run:

text
taskkill /F /IM python.exe
Then move the folder again.

3) “Can not open file ... openvino_model.bin”
This usually means the model folder is incomplete or the path is wrong.
Re-download the model into models/<model_name>/ using hf download.

4) Launcher runs but responses look random
Some models need a chat template. The launcher tries tokenizer.apply_chat_template() when available and falls back if not.

VSCode Setup
Open this folder in VSCode

Select the right interpreter:
Ctrl+Shift+P → “Python: Select Interpreter”

Run from VSCode terminal:
python launcher.py

License
Personal / internal project usage.




# OpenVINO Local LLM Launcher (Windows)

Windows and VS Code friendly local app for running OpenVINO-converted LLMs (for example Qwen3) with:

- CLI chat launcher
- API server + web GUI
- Persistent config in config.json
- Per-model cache folders for faster warm starts
- Session-isolated chat history in Flask sessions
- Session semantic retrieval memory (Method 2) to reduce follow-up latency

## Recent Updates (March 2026)

- Added per-user session chat isolation in gui_server.py using Flask server-side sessions.
- Added Method 2 session semantic retrieval memory in ram_weight_method2.py.
- Replaced simulated per-query transfer delay with real retrieval timing metrics.
- Added page session reset path for new conversation:
  - Frontend sends page_session_id to /api/new_chat.
  - Backend resets Method 2 memory for that page session.
- Added main-view runtime banner for per-query metrics:
  - Retrieval matched count
  - Retrieval latency
  - Generation latency
  - Stored session turns

## Project Structure

```
Qwen_OpenVINO_App/
|
|- launcher.py                 # CLI launcher
|- api_server.py               # Model API server
|- gui_server.py               # Web GUI server (Flask + SSE)
|- index.html                  # Primary UI served by gui_server.py
|- templates/index.html        # Legacy/alternative UI file
|- set_config.py               # Config helper
|- config.json                 # Runtime config
|- ram_weight_method1.py       # Adaptive partial RAM staging policy model
|- ram_weight_method2.py       # Session semantic retrieval memory strategy
|- requirements.txt            # Core deps
|- requirements_gui.txt        # GUI/API deps
|- models/
|  |- qwen3_gpu/
|- gpu_cache/
|- model_cache/
```

## Quick Start

## 1. Create and activate virtual environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2. Install dependencies

```powershell
pip install -U pip
pip install -r requirements.txt
pip install -r requirements_gui.txt
```

## 3. Start servers

Terminal 1:

```powershell
.\.venv\Scripts\python.exe api_server.py
```

Terminal 2:

```powershell
.\.venv\Scripts\python.exe gui_server.py
```

Open browser:

- http://127.0.0.1:8080 (default gui_server.py port)

Note:

- In this repo, gui_server.py default port is 8080.
- If python is not recognized in PowerShell, use .\.venv\Scripts\python.exe explicitly.

## API Endpoints (GUI Server)

| Endpoint | Method | Purpose |
|---|---|---|
| /api/status | GET | Engine and UI mode status |
| /api/chat | POST | Streaming chat endpoint (SSE) |
| /api/new_chat | POST | Reset session chat and optional Method 2 page memory |
| /api/toggle_think | POST | Toggle think/no-think |
| /api/toggle_deep_think | POST | Toggle deep think mode |
| /api/toggle_free_think | POST | Toggle free think mode |
| /api/tools | GET | List tools |
| /api/call_tool | POST | Execute tool |
| /api/reload_tools | POST | Reload tool config |

## Method 1 vs Method 2

## Method 1: Adaptive RAM staging policy

- File: ram_weight_method1.py
- Purpose: Compute a partial RAM budget plan for weight staging.
- Status: Policy helper only, not wired to low-level OpenVINO tensor paging.

## Method 2: Session semantic retrieval memory

- File: ram_weight_method2.py
- Purpose: Reduce latency and improve continuity by reusing relevant recent turns.
- Input per query: page_session_id, prompt text.
- Flow:
  1. Rank prior turns in the same page session by lightweight similarity.
  2. Select top-k relevant turns.
  3. Build compact memory context.
  4. Inject memory context into the user message for model call.
  5. Store new query/response pair after generation.
- Returned metrics:
  - retrieval_performed
  - retrieved_count
  - retrieval_ms
  - generation_ms (from backend stream end)
  - stored_turns

Important:

- This is semantic memory retrieval, not per-query model weight paging.
- True reusable KV cache control depends on lower-level runtime exposure.

## Session Behavior

- chat_history is stored per Flask session.
- Method 2 memory is keyed by page_session_id.
- Clicking New Conversation now resets:
  - Flask chat history
  - Method 2 semantic memory for that page session

## Add a Model (example)

```powershell
hf download OpenVINO/Qwen3-8B-int8-ov --local-dir models\qwen3_gpu
```

Then configure defaults:

```powershell
.\.venv\Scripts\python.exe set_config.py
```

## CLI Mode

```powershell
.\.venv\Scripts\python.exe launcher.py
```

## Troubleshooting

## ModuleNotFoundError: flask_session

Install in the same interpreter used to run gui_server.py:

```powershell
.\.venv\Scripts\python.exe -m pip install flask-session
```

Verify:

```powershell
.\.venv\Scripts\python.exe -c "from flask_session import Session; print('ok')"
```

## Engine still loading for long time

- First run on GPU can be slow due to compilation and cache warmup.
- Reuse same model folder and cache directory between runs.
- Ensure model path in config.json matches actual models folder name.

## No models found

- Confirm folder exists under models/
- Example: models/qwen3_gpu/

## License

Personal / internal project usage.




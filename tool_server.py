"""
MCP-Style Tool Server Environment
──────────────────────────────────
- Tools are registered via decorator or loaded from tools_config.json
- Each tool has: name, description, trigger keywords, auto_inject flag
- The server exposes: list / call / match / schema — like an MCP server
- New tools are added here; gui_server.py just imports and calls.
"""

import json
import psutil
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ══════════════════════════════════════════════════════════════
#  TOOL REGISTRY
# ══════════════════════════════════════════════════════════════

_CONFIG_PATH = Path(__file__).parent / "tools_config.json"

class Tool:
    """Single tool definition."""
    __slots__ = ("name", "description", "fn", "triggers", "auto_inject", "enabled")

    def __init__(self, name: str, description: str, fn: Callable,
                 triggers: Optional[List[str]] = None, auto_inject: bool = False,
                 enabled: bool = True):
        self.name = name
        self.description = description
        self.fn = fn
        self.triggers = triggers or []
        self.auto_inject = auto_inject
        self.enabled = enabled

    def call(self, **kwargs) -> str:
        return self.fn(**kwargs)

    def to_schema(self) -> Dict:
        """MCP-style tool schema."""
        return {
            "name": self.name,
            "description": self.description,
            "triggers": self.triggers,
            "auto_inject": self.auto_inject,
            "enabled": self.enabled,
        }


class ToolServer:
    """
    MCP-style tool server.
    - Loads config from tools_config.json
    - Tools register via @server.tool() decorator
    - Supports: list, call, match (auto-trigger), schema
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._config: Dict = {}
        self._load_config()

    # ── Config ────────────────────────────────────────────────
    def _load_config(self):
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH, "r") as f:
                self._config = json.load(f)

    def _get_tool_config(self, name: str) -> Dict:
        return self._config.get("tools", {}).get(name, {})

    def reload_config(self):
        """Hot-reload tools_config.json (call at runtime to pick up changes)."""
        self._load_config()
        # Apply enable/disable from config
        for name, tool in self._tools.items():
            cfg = self._get_tool_config(name)
            if cfg:
                tool.enabled = cfg.get("enabled", True)
                tool.triggers = cfg.get("triggers", tool.triggers)
                tool.auto_inject = cfg.get("auto_inject", tool.auto_inject)

    # ── Registration ──────────────────────────────────────────
    def tool(self, name: Optional[str] = None, description: str = "",
             triggers: Optional[List[str]] = None, auto_inject: bool = False):
        """Decorator to register a tool function."""
        def decorator(fn: Callable):
            tool_name = name or fn.__name__
            cfg = self._get_tool_config(tool_name)
            t = Tool(
                name=tool_name,
                description=cfg.get("description", description),
                fn=fn,
                triggers=cfg.get("triggers", triggers or []),
                auto_inject=cfg.get("auto_inject", auto_inject),
                enabled=cfg.get("enabled", True),
            )
            self._tools[tool_name] = t
            return fn
        return decorator

    def register(self, name: str, description: str, fn: Callable,
                 triggers: Optional[List[str]] = None, auto_inject: bool = False):
        """Programmatic registration (non-decorator)."""
        cfg = self._get_tool_config(name)
        t = Tool(
            name=name,
            description=cfg.get("description", description),
            fn=fn,
            triggers=cfg.get("triggers", triggers or []),
            auto_inject=cfg.get("auto_inject", auto_inject),
            enabled=cfg.get("enabled", True),
        )
        self._tools[name] = t

    # ── MCP-style interface ───────────────────────────────────
    def list_tools(self) -> List[Dict]:
        """List all enabled tools (MCP tools/list)."""
        return [t.to_schema() for t in self._tools.values() if t.enabled]

    def call_tool(self, name: str, **kwargs) -> Optional[str]:
        """Call a tool by name (MCP tools/call)."""
        tool = self._tools.get(name)
        if tool and tool.enabled:
            return tool.call(**kwargs)
        return None

    def get_tool(self, name: str) -> Optional[Tool]:
        t = self._tools.get(name)
        return t if t and t.enabled else None

    def match_triggers(self, user_input: str) -> List[Dict[str, str]]:
        """
        Auto-detect which tools should fire based on user input.
        Returns list of {"name": ..., "result": ...} for matching tools.
        """
        lower = user_input.lower()
        results = []
        for tool in self._tools.values():
            if not tool.enabled or not tool.auto_inject:
                continue
            if any(trigger in lower for trigger in tool.triggers):
                result = tool.call()
                results.append({"name": tool.name, "result": result})
        return results

    def tools_summary(self) -> str:
        """One-line-per-tool summary for system prompt injection."""
        lines = []
        for t in self._tools.values():
            if t.enabled:
                lines.append(f"- {t.name}: {t.description}")
        return "\n".join(lines)

    def tools_help(self) -> str:
        """Formatted help text for the chat UI."""
        lines = []
        for t in self._tools.values():
            if t.enabled:
                auto = " [auto]" if t.auto_inject else ""
                lines.append(f"    /{t.name:<15} {t.description}{auto}")
        return "\n".join(lines)

    @property
    def server_info(self) -> Dict:
        return {
            "name": self._config.get("server_name", "qwen-tool-server"),
            "version": self._config.get("version", "1.0.0"),
            "tool_count": len([t for t in self._tools.values() if t.enabled]),
        }


# ══════════════════════════════════════════════════════════════
#  DEFAULT TOOL SERVER INSTANCE + BUILT-IN TOOLS
# ══════════════════════════════════════════════════════════════

server = ToolServer()


@server.tool(name="time", description="Returns the current date and time",
             triggers=["time", "date", "today", "day is it", "what day", "current time", "right now"],
             auto_inject=True)
def tool_time():
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y  %I:%M:%S %p")


@server.tool(name="system_info", description="Returns system hardware and memory info",
             triggers=["system info", "hardware", "ram usage", "cpu usage", "memory"],
             auto_inject=False)
def tool_system_info():
    mem = psutil.virtual_memory()
    cpu_pct = psutil.cpu_percent(interval=0.1)
    return (
        f"CPU: {platform.processor()} | Cores: {psutil.cpu_count(logical=True)} | "
        f"Load: {cpu_pct:.1f}%\n"
        f"RAM: {mem.total / (1024**3):.1f} GB total | "
        f"{mem.available / (1024**3):.1f} GB free | {mem.percent}% used"
    )


@server.tool(name="model_info", description="Returns current model config, precision, and device info",
             triggers=["model info", "what model", "precision", "which model", "model status"],
             auto_inject=False)
def tool_model_info():
    # Read from running engine API
    try:
        import requests as _req
        r = _req.get("http://127.0.0.1:5000/hw_status", timeout=3)
        if r.status_code == 200:
            data = r.json()
            dev = data.get("primary_device", "?")
            overflow = data.get("overflow_device", "none")
            gpu_n = data.get("gpu_infer_count", 0)
            cpu_n = data.get("cpu_infer_count", 0)
            role = data.get("ram_role", {}).get("role", "unknown")
            return (
                f"Device: {dev} (overflow: {overflow})\n"
                f"Inferences — GPU: {gpu_n} | CPU: {cpu_n}\n"
                f"RAM role: {role}"
            )
        else:
            return "Engine not ready"
    except Exception as e:
        return f"Engine unavailable ({e})"


# ══════════════════════════════════════════════════════════════
#  HOW TO ADD A NEW TOOL
# ══════════════════════════════════════════════════════════════
#
#  1. Add config entry in tools_config.json:
#     "my_tool": {
#         "enabled": true,
#         "description": "What the tool does",
#         "triggers": ["keyword1", "keyword2"],
#         "auto_inject": true
#     }
#
#  2. Register the function here:
#
#     @server.tool(name="my_tool", description="...",
#                  triggers=["keyword1", "keyword2"], auto_inject=True)
#     def tool_my_tool():
#         return "result string"
#
#  3. That's it. The tool is available as /my_tool in chat and
#     auto-triggers on matching keywords if auto_inject=True.
#
# ══════════════════════════════════════════════════════════════

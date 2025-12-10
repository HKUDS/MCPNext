# MCPNext Local Server (Desktop Version)

## 1. Introduction

The MCPNext Local Server is a **lightweight, cross-platform** Flask service that launches on the host workstation and exposes a uniform HTTP interface for controlling the native desktop environment. By translating REST calls into deterministic GUI actions—mouse and keyboard synthesis, window management, screenshot capture, file I/O—it enables higher-level MCPNext agents to interact with real software instead of simulated environments.

**Supported platforms:** Windows 10/11, macOS 11+ (Intel & Apple Silicon) and mainstream Linux distributions (X11/Wayland).

## 2. System Architecture

* **PlatformAdapter** abstracts OS-specific primitives (Windows, macOS, Linux).
* **Accessibility Helper** queries the UI accessibility tree for semantic information.
* **Screenshot Helper** captures full or partial screenshots (PNG).
* **Recorder** streams screen recordings for offline analysis.
* **Health / Feature Checker** validates runtime capabilities and permissions.

## 3. REST Endpoints

| Path | Method | Semantics |
|------|--------|-----------|
| `/` | GET | Liveness probe |
| `/platform` | GET | Return host OS metadata |
| `/execute` | POST | Execute a PyAutoGUI script fragment |
| `/execute_with_verification` | POST | Execute fragment and verify via template matching |
| `/run_python` | POST | Run arbitrary Python within a sandbox |
| `/run_bash_script` | POST | Run shell script (optional conda activation) |
| `/screenshot` | GET | Return PNG screenshot (full or ROI) |
| `/cursor_position` | GET | Current mouse coordinates |
| `/screen_size` | GET/POST | Query or set virtual screen resolution |
| `/list_directory` | POST | List directory contents |

*see* `main.py` *for ~20 additional endpoints.*

## 4. Setup & Launch

> [!NOTE]  
> python=3.12  
> Accessibility / screen-record permissions (macOS: *System Settings ▸ Privacy & Security*).

### Dependency Installation
```bash
cd mcpnext/local_server
pip install -r requirements.txt
```

### Launching the Server
*Python entry point*
```bash
python -m mcpnext.local_server.main \
       --host 127.0.0.1 --port 5000   # flags optional; defaults read from config.json
```

*Bash helper script*
```bash
./run.sh              # reads config.json then starts the service
```

Press `Ctrl+C` at any time to gracefully stop the server.

---

## 5. Configuration
Runtime options live in `config.json`:
```json
{
  "server": {
    "host": "127.0.0.1",    // listening address (0.0.0.0 for all interfaces)
    "port": 5000,           // default port
    "debug": false          // verbose Flask logs
  }
}
```